package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"math/rand/v2"
	"net"
	"net/http"
	"net/textproto"
	"strings"
	"time"
)

var errBodyTooLarge = errors.New("request body too large")

// hopHeaders must not be forwarded by a proxy (RFC 7230 搂6.1).
var hopHeaders = map[string]struct{}{
	"Connection":          {},
	"Proxy-Connection":    {},
	"Keep-Alive":          {},
	"Proxy-Authenticate":  {},
	"Proxy-Authorization": {},
	"Te":                  {},
	"Trailer":             {},
	"Transfer-Encoding":   {},
	"Upgrade":             {},
}

// Proxy forwards OpenAI-compatible requests to a pool of vLLM backends.
//
// Two behaviours are the point of this file:
//
//  1. SSE is flushed chunk by chunk instead of being buffered.
//  2. A request is only ever retried before the first response byte reaches the
//     client, because replaying a partially delivered generation would
//     duplicate or corrupt the output.
type Proxy struct {
	cfg      *Config
	pool     *Pool
	metrics  *Metrics
	client   *http.Client
	log      *slog.Logger
	strategy Strategy
}

// attemptResult describes one upstream attempt.
type attemptResult struct {
	status    int
	committed bool // bytes (or at least headers) already sent to the client
	ttft      time.Duration
	bytes     int64
	err       error
}

// apiError mirrors the OpenAI error envelope so clients need no special case.
type apiError struct {
	Error apiErrorBody `json:"error"`
}

type apiErrorBody struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code,omitempty"`
}

// NewProxy builds the proxy with an HTTP client tuned for long-lived streams.
func NewProxy(cfg *Config, pool *Pool, m *Metrics, log *slog.Logger) *Proxy {
	strategy, _ := ParseStrategy(cfg.Strategy) // validated at load time

	transport := &http.Transport{
		Proxy: http.ProxyFromEnvironment,
		DialContext: (&net.Dialer{
			Timeout:   cfg.Timeouts.Connect.D(),
			KeepAlive: 30 * time.Second,
		}).DialContext,
		MaxIdleConns:          512,
		MaxIdleConnsPerHost:   128,
		IdleConnTimeout:       90 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
		// Compression would hide the true byte stream and confuse
		// Content-Length handling; LLM payloads are JSON/SSE and compress poorly.
		DisableCompression: true,
		ForceAttemptHTTP2:  false,
	}

	return &Proxy{
		cfg:      cfg,
		pool:     pool,
		metrics:  m,
		log:      log,
		strategy: strategy,
		client: &http.Client{
			Transport: transport,
			// Deliberately zero: a global Timeout would kill perfectly healthy
			// multi-minute generations. Timeouts are applied per attempt via ctx.
			Timeout: 0,
		},
	}
}

// ServeHTTP implements the retry loop.
func (p *Proxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	p.metrics.IncInflight()
	defer p.metrics.DecInflight()

	body, err := readRequestBody(r, p.cfg.Retry.MaxBodyBytes)
	if err != nil {
		code := http.StatusBadRequest
		if errors.Is(err, errBodyTooLarge) {
			code = http.StatusRequestEntityTooLarge
		}
		p.metrics.ObserveRequest("none", code, false, time.Since(start), noTTFT)
		writeAPIError(w, code, err.Error())
		return
	}

	isStream := wantsStream(body)
	excluded := make(map[string]struct{}, p.cfg.Retry.Max+1)
	attempts := p.cfg.Retry.Max + 1

	var (
		lastStatus int
		lastErr    error
	)

	for attempt := 0; attempt < attempts; attempt++ {
		backend, aerr := p.pool.Acquire(p.strategy, excluded)
		if aerr != nil {
			lastErr = aerr
			break
		}
		p.metrics.SetRunning(backend.Name, backend.Running.Load())

		res := p.forward(w, r, backend, body, isStream, attempt == attempts-1, start)
		backend.Release()
		p.metrics.SetRunning(backend.Name, backend.Running.Load())

		if res.committed {
			p.metrics.ObserveRequest(backend.Name, res.status, isStream, time.Since(start), res.ttft)
			return
		}

		lastStatus, lastErr = res.status, res.err

		// The client hung up while we were waiting on the backend. Do not retry
		// and do not write: the socket is gone, and retrying would only burn GPU.
		if r.Context().Err() != nil {
			p.metrics.ObserveRequest(backend.Name, 499, isStream, time.Since(start), noTTFT)
			return
		}

		excluded[backend.Name] = struct{}{} // never retry the instance that just failed
		if attempt < attempts-1 {
			p.metrics.IncRetry(backend.Name)
			p.log.Warn("upstream attempt failed, retrying",
				"backend", backend.Name,
				"status", res.status,
				"attempt", attempt+1,
				"error", errString(res.err),
			)
			if !sleepContext(r.Context(), p.backoff(attempt)) {
				p.metrics.ObserveRequest(backend.Name, 499, isStream, time.Since(start), noTTFT)
				return
			}
		}
	}

	code := http.StatusBadGateway
	switch {
	case errors.Is(lastErr, ErrNoHealthyBackend):
		code = http.StatusServiceUnavailable
	case errors.Is(lastErr, context.DeadlineExceeded):
		code = http.StatusGatewayTimeout
	case lastStatus > 0 && lastStatus < 500:
		code = lastStatus
	}

	p.metrics.ObserveRequest("none", code, isStream, time.Since(start), noTTFT)
	p.log.Error("request failed", "path", r.URL.Path, "status", code, "error", errString(lastErr))
	writeAPIError(w, code, "all backend attempts failed")
}

// forward performs exactly one upstream attempt.
func (p *Proxy) forward(w http.ResponseWriter, r *http.Request, backend *Backend, body []byte, stream, final bool, start time.Time) attemptResult {
	ctx := r.Context()
	var cancel context.CancelFunc
	var firstByteTimer *time.Timer

	if stream {
		// Streaming gets a first-byte deadline, not a total deadline. Once the
		// first chunk is in, generation may take as long as it needs.
		ctx, cancel = context.WithCancel(ctx)
		firstByteTimer = time.AfterFunc(p.cfg.Timeouts.StreamFirstByte.D(), cancel)
	} else {
		ctx, cancel = context.WithTimeout(ctx, p.cfg.Timeouts.Request.D())
	}
	defer cancel()
	if firstByteTimer != nil {
		defer firstByteTimer.Stop()
	}

	target := *backend.URL
	target.Path = singleJoiningSlash(backend.URL.Path, r.URL.Path)
	target.RawQuery = r.URL.RawQuery

	req, err := http.NewRequestWithContext(ctx, r.Method, target.String(), bytes.NewReader(body))
	if err != nil {
		return attemptResult{err: err}
	}
	copyRequestHeaders(req.Header, r.Header)
	appendForwardedFor(req, r)
	req.Header.Set("X-Gateway-Backend", backend.Name)

	resp, err := p.client.Do(req)
	if err != nil {
		return attemptResult{err: err}
	}
	defer func() {
		_, _ = io.CopyN(io.Discard, resp.Body, 512)
		resp.Body.Close()
	}()

	// 5xx is retryable unless this was the final attempt, in which case the
	// caller wants the upstream's own error body rather than a generic 502.
	if resp.StatusCode >= 500 && !final {
		_, _ = io.CopyN(io.Discard, resp.Body, 4096)
		return attemptResult{status: resp.StatusCode}
	}

	w.Header().Set("X-Gateway-Backend", backend.Name)

	if stream && isEventStream(resp.Header.Get("Content-Type")) {
		return p.streamCopy(w, resp, start, firstByteTimer)
	}
	return p.copyBuffered(w, resp, start)
}

// streamCopy relays an SSE response. It reads the first chunk *before* writing
// response headers: if the backend fails during prefill, nothing has reached the
// client yet, so the caller can safely retry on another backend.
func (p *Proxy) streamCopy(w http.ResponseWriter, resp *http.Response, start time.Time, firstByteTimer *time.Timer) attemptResult {
	flusher, canFlush := w.(http.Flusher)

	// From a gateway that was never handed a Flusher, fall back to a plain copy
	// (loses streaming but keeps the response correct).
	if !canFlush {
		copyResponseHeaders(w.Header(), resp.Header)
		w.WriteHeader(resp.StatusCode)
		n, err := io.CopyBuffer(w, resp.Body, make([]byte, 32*1024))
		return attemptResult{status: resp.StatusCode, committed: true, ttft: time.Since(start), bytes: n, err: err}
	}

	buf := make([]byte, 32*1024)
	var (
		committed bool
		ttft      time.Duration
		total     int64
	)

	for {
		n, rerr := resp.Body.Read(buf)
		if n > 0 {
			if !committed {
				copyResponseHeaders(w.Header(), resp.Header)
				w.Header().Set("Cache-Control", "no-cache")
				// Required when nginx sits in front, otherwise it buffers the
				// whole stream and defeats the point.
				w.Header().Set("X-Accel-Buffering", "no")
				w.WriteHeader(resp.StatusCode)
				committed = true
				ttft = time.Since(start)
				if firstByteTimer != nil {
					firstByteTimer.Stop()
				}
			}
			if _, werr := w.Write(buf[:n]); werr != nil {
				// Client is gone; stop immediately so the backend can be
				// cancelled via the request context.
				return attemptResult{status: resp.StatusCode, committed: true, ttft: ttft, bytes: total}
			}
			flusher.Flush()
			total += int64(n)
		}

		if rerr != nil {
			if !committed {
				if rerr == io.EOF {
					rerr = errors.New("upstream closed stream before first byte")
				}
				return attemptResult{status: resp.StatusCode, err: rerr}
			}
			if rerr == io.EOF {
				rerr = nil
			}
			return attemptResult{status: resp.StatusCode, committed: true, ttft: ttft, bytes: total, err: rerr}
		}
	}
}

// copyBuffered reads a bounded non-streaming response into memory first, so a
// truncated body can still be retried. Responses larger than the limit fall
// back to pass-through streaming.
func (p *Proxy) copyBuffered(w http.ResponseWriter, resp *http.Response, start time.Time) attemptResult {
	limit := p.cfg.Retry.MaxResponseBody
	data, err := io.ReadAll(io.LimitReader(resp.Body, limit+1))
	if err != nil {
		return attemptResult{status: resp.StatusCode, err: err}
	}

	copyResponseHeaders(w.Header(), resp.Header)
	w.WriteHeader(resp.StatusCode)

	if int64(len(data)) > limit {
		if _, werr := w.Write(data); werr != nil {
			return attemptResult{status: resp.StatusCode, committed: true, bytes: int64(len(data)), err: werr}
		}
		n, cerr := io.CopyBuffer(w, resp.Body, make([]byte, 32*1024))
		return attemptResult{status: resp.StatusCode, committed: true, ttft: time.Since(start), bytes: int64(len(data)) + n, err: cerr}
	}

	if _, werr := w.Write(data); werr != nil {
		return attemptResult{status: resp.StatusCode, committed: true, ttft: time.Since(start), bytes: int64(len(data)), err: werr}
	}
	return attemptResult{status: resp.StatusCode, committed: true, ttft: time.Since(start), bytes: int64(len(data))}
}

func (p *Proxy) backoff(attempt int) time.Duration {
	d := p.cfg.Retry.Backoff.D() << attempt
	if max := p.cfg.Retry.MaxBackoff.D(); d > max || d <= 0 {
		d = max
	}
	// Full jitter: avoids a thundering herd of retries against the same backend.
	return time.Duration(rand.Int64N(int64(d) + 1))
}

func readRequestBody(r *http.Request, limit int64) ([]byte, error) {
	if r.Body == nil || r.Body == http.NoBody {
		return nil, nil
	}
	defer r.Body.Close()

	body, err := io.ReadAll(io.LimitReader(r.Body, limit+1))
	if err != nil {
		return nil, fmt.Errorf("read request body: %w", err)
	}
	if int64(len(body)) > limit {
		return nil, errBodyTooLarge
	}
	return body, nil
}

// wantsStream inspects the JSON body for {"stream": true}. It never returns an
// error: an unparsable body is simply treated as non-streaming and left for the
// backend to reject with its own 400.
func wantsStream(body []byte) bool {
	if len(body) == 0 {
		return false
	}
	var payload struct {
		Stream bool `json:"stream"`
	}
	if err := json.Unmarshal(body, &payload); err != nil {
		return false
	}
	return payload.Stream
}

func isEventStream(contentType string) bool {
	return strings.HasPrefix(strings.ToLower(contentType), "text/event-stream")
}

func copyRequestHeaders(dst, src http.Header) {
	for k, vv := range src {
		ck := textproto.CanonicalMIMEHeaderKey(k)
		if _, hop := hopHeaders[ck]; hop {
			continue
		}
		// Let net/http compute framing; forwarding these confuses the transport.
		if ck == "Host" || ck == "Content-Length" || ck == "Accept-Encoding" {
			continue
		}
		for _, v := range vv {
			dst.Add(ck, v)
		}
	}
}

func copyResponseHeaders(dst, src http.Header) {
	for k, vv := range src {
		ck := textproto.CanonicalMIMEHeaderKey(k)
		if _, hop := hopHeaders[ck]; hop {
			continue
		}
		for _, v := range vv {
			dst.Add(ck, v)
		}
	}
}

func appendForwardedFor(req *http.Request, r *http.Request) {
	ip, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		ip = r.RemoteAddr
	}
	if prior := r.Header.Get("X-Forwarded-For"); prior != "" {
		req.Header.Set("X-Forwarded-For", prior+", "+ip)
	} else {
		req.Header.Set("X-Forwarded-For", ip)
	}
}

func singleJoiningSlash(a, b string) string {
	aslash := strings.HasSuffix(a, "/")
	bslash := strings.HasPrefix(b, "/")
	switch {
	case aslash && bslash:
		return a + b[1:]
	case !aslash && !bslash:
		return a + "/" + b
	}
	return a + b
}

func sleepContext(ctx context.Context, d time.Duration) bool {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return true
	case <-ctx.Done():
		return false
	}
}

func writeAPIError(w http.ResponseWriter, code int, message string) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(apiError{Error: apiErrorBody{
		Message: message,
		Type:    "gateway_error",
	}})
}

func errString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}
