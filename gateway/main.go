package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"golang.org/x/time/rate"
)

// version is injected at build time: -ldflags "-X main.version=$(git rev-parse --short HEAD)".
var version = "dev"

func main() {
	configPath := flag.String("config", "config.yaml", "path to config.yaml")
	listen := flag.String("listen", "", "override listen address, e.g. :9000")
	flag.Parse()

	if err := run(*configPath, *listen); err != nil {
		fmt.Fprintf(os.Stderr, "fatal: %v\n", err)
		os.Exit(1)
	}
}

func run(configPath, listenOverride string) error {
	cfg, err := LoadConfig(configPath)
	if err != nil {
		return err
	}
	if listenOverride != "" {
		cfg.Listen = listenOverride
	}

	log := newLogger(cfg.LogLevel)
	slog.SetDefault(log)

	metrics := NewMetrics()

	pool, err := NewPool(cfg, log)
	if err != nil {
		return err
	}
	pool.SetHealthHook(func(b *Backend, healthy bool) {
		metrics.SetHealthy(b.Name, healthy)
	})
	for _, b := range pool.Backends() {
		metrics.SetHealthy(b.Name, false)
		metrics.SetRunning(b.Name, 0)
	}

	// First probe is synchronous and bounded by health_check.timeout, so the
	// process never advertises ready before an upstream has answered.
	pool.Start(context.Background())
	defer pool.Stop()

	proxy := NewProxy(cfg, pool, metrics, log)

	mux := http.NewServeMux()
	mux.Handle("/metrics", metrics.Handler())
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	})
	mux.HandleFunc("/readyz", func(w http.ResponseWriter, r *http.Request) {
		healthy := pool.HealthyCount()
		code := http.StatusOK
		if healthy == 0 {
			code = http.StatusServiceUnavailable
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(code)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"status":           map[bool]string{true: "ready", false: "not_ready"}[healthy > 0],
			"healthy_backends": healthy,
			"total_backends":   len(pool.Backends()),
		})
	})
	mux.HandleFunc("/debug/backends", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(pool.Snapshot())
	})
	mux.Handle("/", proxy)

	var handler http.Handler = mux
	handler = accessLog(log, handler)
	if cfg.RateLimit.RPS > 0 {
		handler = rateLimit(rate.NewLimiter(rate.Limit(cfg.RateLimit.RPS), cfg.RateLimit.Burst), handler)
		log.Info("rate limiting enabled", "rps", cfg.RateLimit.RPS, "burst", cfg.RateLimit.Burst)
	}

	srv := &http.Server{
		Addr:    cfg.Listen,
		Handler: handler,
		// ReadHeaderTimeout protects against slowloris; there is deliberately no
		// WriteTimeout, because a legitimate stream can last for minutes.
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       120 * time.Second,
		WriteTimeout:      0,
	}

	log.Info("gateway starting",
		"version", version,
		"listen", cfg.Listen,
		"strategy", cfg.Strategy,
		"backends", len(cfg.Backends),
		"retry_max", cfg.Retry.Max,
	)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	serverErr := make(chan error, 1)
	go func() {
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			serverErr <- err
			return
		}
		serverErr <- nil
	}()

	select {
	case err := <-serverErr:
		if err != nil {
			return fmt.Errorf("listen on %s: %w", cfg.Listen, err)
		}
		return nil
	case <-ctx.Done():
		log.Info("shutdown signal received, draining connections", "grace", cfg.Timeouts.Shutdown.D().String())
	}

	shutdownCtx, cancel := context.WithTimeout(context.Background(), cfg.Timeouts.Shutdown.D())
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Error("graceful shutdown timed out", "error", err)
	}
	pool.Stop()
	log.Info("gateway stopped")
	return nil
}

func newLogger(level string) *slog.Logger {
	var l slog.Level
	switch level {
	case "debug":
		l = slog.LevelDebug
	case "warn":
		l = slog.LevelWarn
	case "error":
		l = slog.LevelError
	default:
		l = slog.LevelInfo
	}
	return slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: l}))
}

// rateLimit rejects requests over the global token bucket with a 429.
func rateLimit(limiter *rate.Limiter, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !limiter.Allow() {
			w.Header().Set("Retry-After", "1")
			writeAPIError(w, http.StatusTooManyRequests, "gateway rate limit exceeded")
			return
		}
		next.ServeHTTP(w, r)
	})
}

// statusRecorder captures the status code for access logs. It must forward
// Flush, otherwise the proxy's http.Flusher assertion would fail and SSE would
// silently be buffered.
type statusRecorder struct {
	http.ResponseWriter
	status int
	bytes  int64
}

func (r *statusRecorder) WriteHeader(code int) {
	if r.status == 0 {
		r.status = code
		r.ResponseWriter.WriteHeader(code)
	}
}

func (r *statusRecorder) Write(b []byte) (int, error) {
	if r.status == 0 {
		r.status = http.StatusOK
	}
	n, err := r.ResponseWriter.Write(b)
	r.bytes += int64(n)
	return n, err
}

func (r *statusRecorder) Flush() {
	if f, ok := r.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func accessLog(log *slog.Logger, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rec := &statusRecorder{ResponseWriter: w}
		next.ServeHTTP(rec, r)
		if rec.status == 0 {
			rec.status = http.StatusOK
		}
		log.Info("request",
			"method", r.Method,
			"path", r.URL.Path,
			"status", rec.status,
			"duration_ms", time.Since(start).Milliseconds(),
			"bytes", rec.bytes,
			"remote", r.RemoteAddr,
		)
	})
}
