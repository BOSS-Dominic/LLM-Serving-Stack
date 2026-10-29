package main

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	dto "github.com/prometheus/client_model/go"
)

const chatBody = `{"model":"qwen","messages":[{"role":"user","content":"hi"}],"stream":false}`

// newGateway builds the full handler chain (access log + proxy + metrics) so
// tests exercise the same wrapping the real binary uses.
func newGateway(t *testing.T, cfg *Config) (*httptest.Server, *Pool, *Metrics) {
	t.Helper()
	pool, err := NewPool(cfg, testLogger())
	if err != nil {
		t.Fatalf("NewPool: %v", err)
	}
	for _, b := range pool.Backends() {
		b.Healthy.Store(true)
	}
	metrics := NewMetrics()
	proxy := NewProxy(cfg, pool, metrics, testLogger())
	srv := httptest.NewServer(accessLog(testLogger(), proxy))
	t.Cleanup(srv.Close)
	return srv, pool, metrics
}

func post(t *testing.T, url, body string) *http.Response {
	t.Helper()
	resp, err := http.Post(url+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("POST %s: %v", url, err)
	}
	return resp
}

func readAll(t *testing.T, resp *http.Response) string {
	t.Helper()
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	return string(b)
}

// TestRetrySendsNonEmptyBody is the single most important test in this file:
// r.Body is one-shot, so a naive implementation retries with an empty body and
// still logs "retry succeeded".
func TestRetrySendsNonEmptyBody(t *testing.T) {
	var (
		mu       sync.Mutex
		failBody []string
		okBody   []string
	)

	failSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		failBody = append(failBody, string(b))
		mu.Unlock()
		http.Error(w, "upstream exploded", http.StatusInternalServerError)
	}))
	defer failSrv.Close()

	okSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		okBody = append(okBody, string(b))
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"ok"}}]}`))
	}))
	defer okSrv.Close()

	cfg := testConfig(failSrv.URL, okSrv.URL)
	cfg.Retry.Max = 1
	gw, pool, _ := newGateway(t, cfg)

	// Make the failing backend the least-loaded one so the first attempt is
	// deterministic (least_conn picks it).
	pool.Backends()[1].Running.Store(1)

	resp := post(t, gw.URL, chatBody)
	body := readAll(t, resp)

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200 (body %q)", resp.StatusCode, body)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(failBody) != 1 || len(okBody) != 1 {
		t.Fatalf("expected 1 attempt per backend, got fail=%d ok=%d", len(failBody), len(okBody))
	}
	if okBody[0] == "" {
		t.Fatal("retried request carried an EMPTY body - r.Body was consumed and not buffered")
	}
	if okBody[0] != chatBody {
		t.Fatalf("retried body differs:\n got %q\nwant %q", okBody[0], chatBody)
	}
}

func TestDoesNotRetry4xx(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"bad model"}}`))
	}))
	defer srv.Close()

	cfg := testConfig(srv.URL, srv.URL)
	cfg.Retry.Max = 2
	gw, _, _ := newGateway(t, cfg)

	resp := post(t, gw.URL, chatBody)
	body := readAll(t, resp)

	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", resp.StatusCode)
	}
	if !strings.Contains(body, "bad model") {
		t.Fatalf("upstream error body was not preserved: %q", body)
	}
	if got := hits.Load(); got != 1 {
		t.Fatalf("client errors were retried: %d hits, want 1", got)
	}
}

func TestFinal5xxIsForwardedNotMasked(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "model not loaded", http.StatusServiceUnavailable)
	}))
	defer srv.Close()

	cfg := testConfig(srv.URL)
	cfg.Retry.Max = 1
	gw, _, _ := newGateway(t, cfg)

	resp := post(t, gw.URL, chatBody)
	body := readAll(t, resp)
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503 forwarded from upstream", resp.StatusCode)
	}
	if !strings.Contains(body, "model not loaded") {
		t.Fatalf("upstream body lost: %q", body)
	}
}

// TestRetriesWhenStreamDiesBeforeFirstByte exercises the "read the first chunk
// before committing headers" design: a backend that accepts the request and dies
// during prefill must be retried transparently.
func TestRetriesWhenStreamDiesBeforeFirstByte(t *testing.T) {
	var (
		deadHits  atomic.Int32
		aliveHits atomic.Int32
	)

	dead := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		deadHits.Add(1)
		hj, ok := w.(http.Hijacker)
		if !ok {
			t.Error("hijacker unavailable")
			return
		}
		conn, buf, err := hj.Hijack()
		if err != nil {
			t.Error(err)
			return
		}
		_, _ = buf.WriteString("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n")
		_ = buf.Flush()
		_ = conn.Close() // die before emitting a single token
	}))
	defer dead.Close()

	alive := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		aliveHits.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		f := w.(http.Flusher)
		_, _ = io.WriteString(w, "data: {\"token\":\"hello\"}\n\n")
		f.Flush()
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
		f.Flush()
	}))
	defer alive.Close()

	cfg := testConfig(dead.URL, alive.URL)
	cfg.Retry.Max = 1
	gw, pool, _ := newGateway(t, cfg)
	pool.Backends()[1].Running.Store(1) // pick the dead backend first

	resp := post(t, gw.URL, `{"model":"qwen","stream":true}`)
	body := readAll(t, resp)

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want 200", resp.StatusCode)
	}
	if !strings.Contains(body, "hello") {
		t.Fatalf("client did not receive the retried stream: %q", body)
	}
	if deadHits.Load() != 1 || aliveHits.Load() != 1 {
		t.Fatalf("unexpected hits dead=%d alive=%d", deadHits.Load(), aliveHits.Load())
	}
}

// TestDoesNotRetryAfterFirstByte is the correctness boundary: once a token has
// reached the client, replaying the generation elsewhere would duplicate text.
func TestDoesNotRetryAfterFirstByte(t *testing.T) {
	var aliveHits atomic.Int32

	partial := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hj, ok := w.(http.Hijacker)
		if !ok {
			t.Error("hijacker unavailable")
			return
		}
		conn, buf, err := hj.Hijack()
		if err != nil {
			t.Error(err)
			return
		}
		_, _ = buf.WriteString("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n")
		_, _ = buf.WriteString("data: {\"token\":\"first\"}\n\n")
		_ = buf.Flush()
		time.Sleep(50 * time.Millisecond)
		_ = conn.Close() // die mid-stream
	}))
	defer partial.Close()

	alive := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		aliveHits.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: DUPLICATE\n\n")
	}))
	defer alive.Close()

	cfg := testConfig(partial.URL, alive.URL)
	cfg.Retry.Max = 2
	gw, pool, _ := newGateway(t, cfg)
	pool.Backends()[1].Running.Store(1) // pick the partial backend first

	resp := post(t, gw.URL, `{"model":"qwen","stream":true}`)
	body := readAll(t, resp)

	if !strings.Contains(body, "first") {
		t.Fatalf("client lost the partial stream: %q", body)
	}
	if aliveHits.Load() != 0 {
		t.Fatalf("gateway retried after tokens were delivered (alive hits=%d) - this duplicates output", aliveHits.Load())
	}
}

// TestStreamsIncrementally catches the classic bug of buffering the whole SSE
// response. It goes through accessLog on purpose, because a status wrapper that
// forgets to implement http.Flusher silently disables streaming.
func TestStreamsIncrementally(t *testing.T) {
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		f := w.(http.Flusher)
		_, _ = io.WriteString(w, "data: {\"token\":\"one\"}\n\n")
		f.Flush()
		time.Sleep(400 * time.Millisecond)
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
		f.Flush()
	}))
	defer upstream.Close()

	cfg := testConfig(upstream.URL)
	gw, _, _ := newGateway(t, cfg)

	start := time.Now()
	resp := post(t, gw.URL, `{"model":"qwen","stream":true}`)
	defer resp.Body.Close()
	if ct := resp.Header.Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Fatalf("content-type = %q, want text/event-stream", ct)
	}

	br := bufio.NewReader(resp.Body)
	line, err := br.ReadString('\n')
	firstChunkAt := time.Since(start)
	if err != nil {
		t.Fatalf("read first chunk: %v", err)
	}
	if !strings.Contains(line, "one") {
		t.Fatalf("unexpected first chunk %q", line)
	}
	rest, _ := io.ReadAll(br)
	totalAt := time.Since(start)

	if firstChunkAt > 250*time.Millisecond {
		t.Fatalf("first chunk took %v: the stream is being buffered, not flushed", firstChunkAt)
	}
	if totalAt < 350*time.Millisecond {
		t.Fatalf("stream finished in %v, expected to span the 400ms upstream delay", totalAt)
	}
	if !strings.Contains(string(rest), "[DONE]") {
		t.Fatalf("tail of stream missing: %q", rest)
	}
}

func TestClientDisconnectCancelsUpstream(t *testing.T) {
	started := make(chan struct{}, 1)
	canceled := make(chan struct{}, 1)

	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Real servers consume the body; doing so also lets net/http notice the
		// peer going away and cancel this request's context.
		_, _ = io.Copy(io.Discard, r.Body)
		started <- struct{}{}
		select {
		case <-r.Context().Done():
			canceled <- struct{}{}
		case <-time.After(5 * time.Second):
			t.Error("upstream request was not cancelled after the client left")
		}
	}))
	defer upstream.Close()

	cfg := testConfig(upstream.URL)
	gw, _, _ := newGateway(t, cfg)

	ctx, cancel := context.WithCancel(context.Background())
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, gw.URL+"/v1/chat/completions", strings.NewReader(chatBody))
	req.Header.Set("Content-Type", "application/json")

	go func() {
		<-started
		cancel() // client walks away mid-generation
	}()

	_, err := http.DefaultClient.Do(req)
	if err == nil {
		t.Fatal("expected the client request to be cancelled")
	}

	select {
	case <-canceled:
	case <-time.After(3 * time.Second):
		t.Fatal("upstream context was never cancelled - GPU time would be wasted")
	}
}

func TestNonStreamingTimeoutReturns504(t *testing.T) {
	slow := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(600 * time.Millisecond)
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer slow.Close()

	cfg := testConfig(slow.URL)
	cfg.Retry.Max = 0
	cfg.Timeouts.Request = Duration(80 * time.Millisecond)
	gw, _, _ := newGateway(t, cfg)

	start := time.Now()
	resp := post(t, gw.URL, chatBody)
	_ = readAll(t, resp)
	elapsed := time.Since(start)

	if resp.StatusCode != http.StatusGatewayTimeout {
		t.Fatalf("status = %d, want 504", resp.StatusCode)
	}
	if elapsed > 400*time.Millisecond {
		t.Fatalf("timeout took %v, request timeout was not applied", elapsed)
	}
}

func TestRequestBodyTooLarge(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("upstream must not be called for an oversized body")
	}))
	defer srv.Close()

	cfg := testConfig(srv.URL)
	cfg.Retry.MaxBodyBytes = 64
	gw, _, _ := newGateway(t, cfg)

	big := strings.Repeat("x", 200)
	resp := post(t, gw.URL, `{"model":"`+big+`"}`)
	_ = readAll(t, resp)
	if resp.StatusCode != http.StatusRequestEntityTooLarge {
		t.Fatalf("status = %d, want 413", resp.StatusCode)
	}
}

func TestForwardsPathQueryHeaders(t *testing.T) {
	var gotPath, gotQuery, gotCustom, gotXFF, gotBackend string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotQuery = r.URL.RawQuery
		gotCustom = r.Header.Get("X-Custom")
		gotXFF = r.Header.Get("X-Forwarded-For")
		gotBackend = r.Header.Get("X-Gateway-Backend")
		w.Header().Set("X-Upstream", "yes")
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer srv.Close()

	cfg := testConfig(srv.URL)
	gw, _, _ := newGateway(t, cfg)

	req, _ := http.NewRequest(http.MethodPost, gw.URL+"/v1/chat/completions?trace=1", strings.NewReader(chatBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Custom", "abc")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	body := readAll(t, resp)

	if resp.StatusCode != http.StatusOK || !strings.Contains(body, "ok") {
		t.Fatalf("unexpected response %d %q", resp.StatusCode, body)
	}
	if gotPath != "/v1/chat/completions" {
		t.Fatalf("path = %q", gotPath)
	}
	if gotQuery != "trace=1" {
		t.Fatalf("query = %q", gotQuery)
	}
	if gotCustom != "abc" {
		t.Fatalf("custom header lost: %q", gotCustom)
	}
	if gotXFF == "" {
		t.Fatal("X-Forwarded-For not set")
	}
	if gotBackend != "b1" {
		t.Fatalf("backend header = %q, want b1", gotBackend)
	}
	if resp.Header.Get("X-Upstream") != "yes" || resp.Header.Get("X-Gateway-Backend") != "b1" {
		t.Fatalf("response headers not propagated: %v", resp.Header)
	}
}

func TestAllBackendsDownReturns503(t *testing.T) {
	cfg := testConfig("http://127.0.0.1:1")
	pool, err := NewPool(cfg, testLogger())
	if err != nil {
		t.Fatal(err)
	}
	// No backend marked healthy.
	proxy := NewProxy(cfg, pool, NewMetrics(), testLogger())
	gw := httptest.NewServer(proxy)
	defer gw.Close()

	resp := post(t, gw.URL, chatBody)
	_ = readAll(t, resp)
	if resp.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("status = %d, want 503", resp.StatusCode)
	}
}

func TestMetricsRecordOutcome(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer srv.Close()

	cfg := testConfig(srv.URL)
	gw, _, metrics := newGateway(t, cfg)

	resp := post(t, gw.URL, chatBody)
	_ = readAll(t, resp)

	// The gateway records the outcome after the last byte is flushed, so from
	// the client's point of view metrics become visible a moment later.
	waitFor(t, time.Second, "requests_total", func() bool {
		v, ok := findMetricValue(metrics, "gateway_requests_total", map[string]string{"backend": "b1", "code": "200"})
		return ok && v == 1
	})
	waitFor(t, time.Second, "running gauge back to 0", func() bool {
		v, ok := findMetricValue(metrics, "gateway_backend_running_requests", map[string]string{"backend": "b1"})
		return ok && v == 0
	})
	waitFor(t, time.Second, "ttft observation", func() bool {
		v, ok := findHistogramCount(metrics, "gateway_ttft_seconds", map[string]string{"backend": "b1"})
		return ok && v == 1
	})
}

// findMetricValue scans the gathered metrics for a single gauge/counter sample.
func findMetricValue(m *Metrics, name string, labels map[string]string) (float64, bool) {
	families, err := m.Registry().Gather()
	if err != nil {
		return 0, false
	}
	for _, f := range families {
		if f.GetName() != name {
			continue
		}
		for _, metric := range f.GetMetric() {
			if !labelsMatch(metric, labels) {
				continue
			}
			if c := metric.GetCounter(); c != nil {
				return c.GetValue(), true
			}
			if g := metric.GetGauge(); g != nil {
				return g.GetValue(), true
			}
		}
	}
	return 0, false
}

func findHistogramCount(m *Metrics, name string, labels map[string]string) (uint64, bool) {
	families, err := m.Registry().Gather()
	if err != nil {
		return 0, false
	}
	for _, f := range families {
		if f.GetName() != name {
			continue
		}
		for _, metric := range f.GetMetric() {
			if labelsMatch(metric, labels) && metric.GetHistogram() != nil {
				return metric.GetHistogram().GetSampleCount(), true
			}
		}
	}
	return 0, false
}

func labelsMatch(metric *dto.Metric, want map[string]string) bool {
	got := make(map[string]string, len(metric.GetLabel()))
	for _, lp := range metric.GetLabel() {
		got[lp.GetName()] = lp.GetValue()
	}
	for k, v := range want {
		if got[k] != v {
			return false
		}
	}
	return true
}

func TestSingleJoiningSlash(t *testing.T) {
	cases := []struct{ a, b, want string }{
		{"", "/v1/chat", "/v1/chat"},
		{"http://h", "/v1/chat", "http://h/v1/chat"},
		{"http://h/", "/v1/chat", "http://h/v1/chat"},
		{"http://h/prefix", "v1", "http://h/prefix/v1"},
	}
	for _, c := range cases {
		if got := singleJoiningSlash(c.a, c.b); got != c.want {
			t.Errorf("singleJoiningSlash(%q,%q) = %q, want %q", c.a, c.b, got, c.want)
		}
	}
}

func TestWantsStream(t *testing.T) {
	if !wantsStream([]byte(`{"stream":true}`)) {
		t.Fatal("stream:true not detected")
	}
	if wantsStream([]byte(`{"stream":false}`)) || wantsStream(nil) || wantsStream([]byte("not json")) {
		t.Fatal("non-stream body misdetected as stream")
	}
}

func ExampleProxy() {
	// Documentation-by-example: the proxy is an http.Handler.
	var _ http.Handler = (*Proxy)(nil)
	fmt.Println("ok")
	// Output: ok
}
