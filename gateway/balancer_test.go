package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func testLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, &slog.HandlerOptions{Level: slog.LevelError}))
}

// testConfig builds a minimal valid config pointing at the given URLs.
func testConfig(urls ...string) *Config {
	cfg := &Config{
		Listen:   ":0",
		Strategy: string(LeastConn),
		LogLevel: "error",
		Timeouts: TimeoutConfig{
			Connect:         Duration(time.Second),
			Request:         Duration(5 * time.Second),
			StreamFirstByte: Duration(2 * time.Second),
			Shutdown:        Duration(time.Second),
		},
		Retry: RetryConfig{
			Max:             2,
			Backoff:         Duration(time.Millisecond),
			MaxBackoff:      Duration(5 * time.Millisecond),
			MaxBodyBytes:    1 << 20,
			MaxResponseBody: 1 << 20,
		},
		HealthCheck: HealthCheckConfig{
			Interval:       Duration(50 * time.Millisecond),
			Path:           "/health",
			Timeout:        Duration(time.Second),
			UnhealthyAfter: 2,
			HealthyAfter:   2,
		},
	}
	for i, u := range urls {
		cfg.Backends = append(cfg.Backends, BackendConfig{
			Name:   fmt.Sprintf("b%d", i+1),
			URL:    u,
			Weight: 1,
		})
	}
	cfg.applyDefaults()
	return cfg
}

// newTestProxy marks every backend healthy and skips the probe loop so tests
// are deterministic and fast.
func newTestProxy(t *testing.T, cfg *Config) (*Proxy, *Pool) {
	t.Helper()
	pool, err := NewPool(cfg, testLogger())
	if err != nil {
		t.Fatalf("NewPool: %v", err)
	}
	for _, b := range pool.Backends() {
		b.Healthy.Store(true)
	}
	return NewProxy(cfg, pool, NewMetrics(), testLogger()), pool
}

func waitFor(t *testing.T, timeout time.Duration, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

func TestAcquireLeastConnPrefersIdleBackend(t *testing.T) {
	cfg := testConfig("http://127.0.0.1:1", "http://127.0.0.1:2")
	pool, err := NewPool(cfg, testLogger())
	if err != nil {
		t.Fatal(err)
	}
	backends := pool.Backends()
	for _, b := range backends {
		b.Healthy.Store(true)
	}
	// b1 looks busy, so least_conn must pick b2 even though b1 comes first.
	backends[0].Running.Store(5)

	for i := 0; i < 10; i++ {
		b, err := pool.Acquire(LeastConn, nil)
		if err != nil {
			t.Fatalf("Acquire: %v", err)
		}
		if b.Name != "b2" {
			t.Fatalf("picked %s, want b2 (b1 has 5 in flight)", b.Name)
		}
		b.Release()
	}
}

func TestLeastConnBreaksTiesRoundRobin(t *testing.T) {
	// Regression test: with strict `<`, two idle backends tie and the first one
	// wins every time, so a single-client workload only ever uses one instance.
	cfg := testConfig("http://127.0.0.1:1", "http://127.0.0.1:2")
	pool, _ := NewPool(cfg, testLogger())
	for _, b := range pool.Backends() {
		b.Healthy.Store(true)
	}

	counts := map[string]int{}
	for i := 0; i < 20; i++ {
		b, err := pool.Acquire(LeastConn, nil)
		if err != nil {
			t.Fatal(err)
		}
		counts[b.Name]++
		b.Release() // fully sequential: every request sees two idle backends
	}

	if counts["b1"] != 10 || counts["b2"] != 10 {
		t.Fatalf("sequential traffic was not spread evenly: %v", counts)
	}
}

func TestAcquireSkipsUnhealthyAndExcluded(t *testing.T) {
	cfg := testConfig("http://127.0.0.1:1", "http://127.0.0.1:2")
	pool, _ := NewPool(cfg, testLogger())
	b1, b2 := pool.Backends()[0], pool.Backends()[1]
	b1.Healthy.Store(false)
	b2.Healthy.Store(true)

	b, err := pool.Acquire(LeastConn, nil)
	if err != nil || b.Name != "b2" {
		t.Fatalf("got (%v,%v), want b2", b, err)
	}
	b.Release()

	// Excluding every healthy backend falls back to the full healthy set rather
	// than failing: a transient error on the last instance deserves a retry.
	b, err = pool.Acquire(LeastConn, map[string]struct{}{"b2": {}})
	if err != nil || b.Name != "b2" {
		t.Fatalf("got (%v,%v), want fallback to b2", b, err)
	}
	b.Release()

	b1.Healthy.Store(false)
	b2.Healthy.Store(false)
	if _, err := pool.Acquire(LeastConn, nil); !errors.Is(err, ErrNoHealthyBackend) {
		t.Fatalf("got %v, want ErrNoHealthyBackend", err)
	}
}

func TestAcquireConcurrentDistributionAndRelease(t *testing.T) {
	cfg := testConfig("http://127.0.0.1:1", "http://127.0.0.1:2")
	pool, _ := NewPool(cfg, testLogger())
	for _, b := range pool.Backends() {
		b.Healthy.Store(true)
	}

	const goroutines = 200
	var wg sync.WaitGroup
	var mu sync.Mutex
	counts := map[string]int{}

	start := make(chan struct{})
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			b, err := pool.Acquire(LeastConn, nil)
			if err != nil {
				t.Errorf("Acquire: %v", err)
				return
			}
			mu.Lock()
			counts[b.Name]++
			mu.Unlock()
			time.Sleep(2 * time.Millisecond)
			b.Release()
		}()
	}
	close(start)
	wg.Wait()

	// The whole point of taking selMu around "choose + increment": under this
	// race, a naive implementation sends almost everything to one backend.
	if counts["b1"] < goroutines/4 || counts["b2"] < goroutines/4 {
		t.Fatalf("unbalanced distribution: %v", counts)
	}
	for _, b := range pool.Backends() {
		if got := b.Running.Load(); got != 0 {
			t.Fatalf("%s leaked %d in-flight slots", b.Name, got)
		}
	}
}

func TestWeightedRoundRobinHonoursWeights(t *testing.T) {
	cfg := testConfig("http://127.0.0.1:1", "http://127.0.0.1:2", "http://127.0.0.1:3")
	// Weights 2:1:1 → b1 should get half the picks.
	cfg.Backends[0].Weight = 2
	pool, _ := NewPool(cfg, testLogger())
	for _, b := range pool.Backends() {
		b.Healthy.Store(true)
	}

	counts := map[string]int{}
	const picks = 400
	for i := 0; i < picks; i++ {
		b, err := pool.Acquire(RoundRobin, nil)
		if err != nil {
			t.Fatal(err)
		}
		counts[b.Name]++
		b.Release()
	}

	if counts["b1"] < picks/2-10 || counts["b1"] > picks/2+10 {
		t.Fatalf("b1 (weight 2) got %d/%d picks, want ~%d", counts["b1"], picks, picks/2)
	}
	for _, name := range []string{"b2", "b3"} {
		if counts[name] < picks/4-10 || counts[name] > picks/4+10 {
			t.Fatalf("%s got %d picks, want ~%d", name, counts[name], picks/4)
		}
	}
}

func TestSmoothWeightedRoundRobinAvoidsBursts(t *testing.T) {
	// nginx's smooth WRR with weights 2:1 alternates instead of emitting A,A,B.
	cfg := testConfig("http://127.0.0.1:1", "http://127.0.0.1:2")
	cfg.Backends[0].Weight = 2
	pool, _ := NewPool(cfg, testLogger())
	for _, b := range pool.Backends() {
		b.Healthy.Store(true)
	}

	var order []string
	for i := 0; i < 6; i++ {
		b, _ := pool.Acquire(RoundRobin, nil)
		order = append(order, b.Name)
		b.Release()
	}
	// First three picks must include b2; a naive [2,1] expansion would not.
	seen := map[string]bool{}
	for _, n := range order[:3] {
		seen[n] = true
	}
	if !seen["b2"] {
		t.Fatalf("bursty order %v: b2 starved in the first 3 picks", order)
	}
}

func TestHealthCheckRemovesAndRestoresBackend(t *testing.T) {
	var fail atomic.Bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if fail.Load() {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	cfg := testConfig(srv.URL)
	cfg.HealthCheck.Interval = Duration(20 * time.Millisecond)
	cfg.HealthCheck.UnhealthyAfter = 2
	cfg.HealthCheck.HealthyAfter = 2

	pool, err := NewPool(cfg, testLogger())
	if err != nil {
		t.Fatal(err)
	}
	pool.Start(context.Background())
	defer pool.Stop()

	b := pool.Backends()[0]
	if !b.Healthy.Load() {
		t.Fatal("initial probe should mark the backend healthy")
	}

	fail.Store(true)
	waitFor(t, 2*time.Second, "backend to be removed", func() bool { return !b.Healthy.Load() })

	fail.Store(false)
	waitFor(t, 2*time.Second, "backend to recover", func() bool { return b.Healthy.Load() })
}

func TestHealthCheckRequiresConsecutiveFailures(t *testing.T) {
	// A single failed probe must not remove a backend.
	var fail atomic.Bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if fail.Load() {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	cfg := testConfig(srv.URL)
	cfg.HealthCheck.Interval = Duration(20 * time.Millisecond)
	cfg.HealthCheck.UnhealthyAfter = 3

	pool, _ := NewPool(cfg, testLogger())
	pool.Start(context.Background())
	defer pool.Stop()
	b := pool.Backends()[0]

	fail.Store(true)
	time.Sleep(25 * time.Millisecond) // one tick at most
	fail.Store(false)
	time.Sleep(60 * time.Millisecond)
	if !b.Healthy.Load() {
		t.Fatal("backend was removed after a single transient failure")
	}
}
