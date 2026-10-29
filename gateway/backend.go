package main

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/url"
	"sync"
	"sync/atomic"
	"time"
)

// Backend is one upstream inference server (a vLLM instance).
//
// All mutable counters are atomics because they are read and written by many
// goroutines: one per in-flight request plus the health checker.
//
// NOTE: because the struct embeds atomics it must never be copied. Backends are
// therefore always handled as *Backend (see []*Backend in Pool); copying one
// would trip go vet's copylocks check and silently produce a second, unrelated
// counter set.
type Backend struct {
	Name   string
	URL    *url.URL
	Weight int

	// Healthy gates selection only. Requests already in flight are never
	// interrupted when a backend is removed ("soft drain").
	Healthy atomic.Bool

	// Running is the number of in-flight requests, used by least_conn.
	Running atomic.Int64

	// TotalRequests counts requests assigned to this backend.
	TotalRequests atomic.Int64

	// ProbeFailures counts health probes that have failed since start.
	ProbeFailures atomic.Int64

	failStreak atomic.Int32
	okStreak   atomic.Int32

	// currentWeight is the smooth weighted round-robin accumulator.
	// It is only touched while Pool.selMu is held.
	currentWeight int64
}

// Acquire reserves one in-flight slot and returns the new count.
func (b *Backend) Acquire() int64 {
	b.TotalRequests.Add(1)
	return b.Running.Add(1)
}

// Release frees the in-flight slot reserved by Acquire.
func (b *Backend) Release() int64 { return b.Running.Add(-1) }

// BackendStatus is the JSON shape exposed by /debug/backends.
type BackendStatus struct {
	Name          string `json:"name"`
	URL           string `json:"url"`
	Weight        int    `json:"weight"`
	Healthy       bool   `json:"healthy"`
	Running       int64  `json:"running"`
	TotalRequests int64  `json:"total_requests"`
	ProbeFailures int64  `json:"probe_failures"`
}

// HealthConfig is the pool's view of health checking.
type HealthConfig struct {
	Interval       time.Duration
	Path           string
	Timeout        time.Duration
	UnhealthyAfter int
	HealthyAfter   int
}

// Pool owns the backend list and performs health checking.
type Pool struct {
	mu       sync.RWMutex
	backends []*Backend

	// selMu serialises backend selection together with the in-flight counter
	// increment. Without it, two goroutines can both observe the same minimum
	// Running value, pick the same backend, and then both increment it — under
	// load that degenerates into "everything goes to one instance". The lock is
	// held for a handful of atomic loads, so it is not a throughput concern for
	// an LLM gateway, where requests live for seconds.
	selMu sync.Mutex

	health HealthConfig
	client *http.Client
	log    *slog.Logger

	// onHealthChange is invoked on every health transition (optional).
	onHealthChange func(*Backend, bool)

	done     chan struct{}
	stopOnce sync.Once
	wg       sync.WaitGroup
}

// NewPool builds a pool from the validated config.
func NewPool(cfg *Config, log *slog.Logger) (*Pool, error) {
	backends := make([]*Backend, 0, len(cfg.Backends))
	for _, bc := range cfg.Backends {
		u, err := url.Parse(bc.URL)
		if err != nil {
			return nil, fmt.Errorf("parse backend %q: %w", bc.Name, err)
		}
		weight := bc.Weight
		if weight <= 0 {
			weight = 1
		}
		backends = append(backends, &Backend{
			Name:   bc.Name,
			URL:    u,
			Weight: weight,
		})
	}

	p := &Pool{
		backends: backends,
		health: HealthConfig{
			Interval:       cfg.HealthCheck.Interval.D(),
			Path:           cfg.HealthCheck.Path,
			Timeout:        cfg.HealthCheck.Timeout.D(),
			UnhealthyAfter: cfg.HealthCheck.UnhealthyAfter,
			HealthyAfter:   cfg.HealthCheck.HealthyAfter,
		},
		client: &http.Client{
			// No global Timeout: it is enforced per probe via context.
			Transport: &http.Transport{
				// A health probe should validate the path a *new* request takes,
				// including a fresh TCP handshake. Reusing a keep-alive
				// connection can keep reporting an instance healthy after it
				// stopped accepting new connections.
				DisableKeepAlives:   true,
				MaxIdleConnsPerHost: 1,
				IdleConnTimeout:     30 * time.Second,
			},
		},
		log:  log,
		done: make(chan struct{}),
	}
	return p, nil
}

// SetHealthHook registers a callback fired on health transitions.
func (p *Pool) SetHealthHook(fn func(*Backend, bool)) { p.onHealthChange = fn }

// Start probes every backend once synchronously, then starts the background
// loop. The synchronous first pass means the gateway never serves traffic to an
// instance that has not answered a probe yet (vLLM can take minutes to load).
func (p *Pool) Start(ctx context.Context) {
	p.probeAll(ctx, true)

	p.wg.Add(1)
	go func() {
		defer p.wg.Done()
		p.healthLoop()
	}()
}

// Stop terminates the health checker and waits for it to exit.
func (p *Pool) Stop() {
	p.stopOnce.Do(func() { close(p.done) })
	p.wg.Wait()
}

func (p *Pool) healthLoop() {
	ticker := time.NewTicker(p.health.Interval)
	defer ticker.Stop() // time.Tick cannot be stopped and would leak

	for {
		select {
		case <-ticker.C:
			p.probeAll(context.Background(), false)
		case <-p.done:
			return
		}
	}
}

// probeAll probes backends concurrently. initial=true applies results strictly
// (no consecutive-success threshold) so startup reflects reality immediately.
func (p *Pool) probeAll(ctx context.Context, initial bool) {
	p.mu.RLock()
	backends := p.backends // pointer slice: copying the header is safe
	p.mu.RUnlock()

	var wg sync.WaitGroup
	for _, b := range backends {
		wg.Add(1)
		go func(b *Backend) {
			defer wg.Done()
			ok := p.probe(ctx, b)
			p.applyProbe(b, ok, initial)
		}(b)
	}
	wg.Wait()
}

func (p *Pool) probe(ctx context.Context, b *Backend) bool {
	ctx, cancel := context.WithTimeout(ctx, p.health.Timeout)
	defer cancel()

	u := *b.URL
	u.Path = singleJoiningSlash(u.Path, p.health.Path)

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return false
	}
	req.Header.Set("Accept", "application/json")

	resp, err := p.client.Do(req)
	if err != nil {
		return false
	}
	defer func() {
		// Drain a little so the connection can be reused.
		_, _ = io.CopyN(io.Discard, resp.Body, 512)
		resp.Body.Close()
	}()
	return resp.StatusCode >= 200 && resp.StatusCode < 300
}

func (p *Pool) applyProbe(b *Backend, ok, initial bool) {
	was := b.Healthy.Load()
	changed := false

	switch {
	case ok:
		b.failStreak.Store(0)
		b.okStreak.Add(1)
		if !was && (initial || int(b.okStreak.Load()) >= p.health.HealthyAfter) {
			b.Healthy.Store(true)
			changed = true
			p.log.Info("backend recovered", "backend", b.Name, "url", b.URL.String())
		}
	case !ok:
		b.okStreak.Store(0)
		b.ProbeFailures.Add(1)
		fails := b.failStreak.Add(1)
		if was && (initial || int(fails) >= p.health.UnhealthyAfter) {
			b.Healthy.Store(false)
			changed = true
			p.log.Warn("backend removed from rotation", "backend", b.Name, "url", b.URL.String(), "consecutive_failures", fails)
		}
	}

	if initial {
		state := "unhealthy"
		if b.Healthy.Load() {
			state = "healthy"
		}
		p.log.Info("initial probe", "backend", b.Name, "state", state)
	}

	if changed || initial {
		p.notify(b, b.Healthy.Load())
	}
}

func (p *Pool) notify(b *Backend, healthy bool) {
	if p.onHealthChange != nil {
		p.onHealthChange(b, healthy)
	}
}

// Snapshot returns a copy of the backend states for /debug/backends.
func (p *Pool) Snapshot() []BackendStatus {
	p.mu.RLock()
	defer p.mu.RUnlock()

	out := make([]BackendStatus, 0, len(p.backends))
	for _, b := range p.backends {
		out = append(out, BackendStatus{
			Name:          b.Name,
			URL:           b.URL.String(),
			Weight:        b.Weight,
			Healthy:       b.Healthy.Load(),
			Running:       b.Running.Load(),
			TotalRequests: b.TotalRequests.Load(),
			ProbeFailures: b.ProbeFailures.Load(),
		})
	}
	return out
}

// HealthyCount reports how many backends are currently in rotation.
func (p *Pool) HealthyCount() int {
	p.mu.RLock()
	defer p.mu.RUnlock()

	n := 0
	for _, b := range p.backends {
		if b.Healthy.Load() {
			n++
		}
	}
	return n
}

// Backends returns the pointer slice. Callers must not mutate it.
func (p *Pool) Backends() []*Backend {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.backends
}
