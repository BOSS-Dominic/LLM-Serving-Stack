package main

import (
	"net/http"
	"strconv"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// Metrics holds the gateway's Prometheus collectors. A dedicated registry is
// used instead of prometheus.DefaultRegisterer so tests can build independent
// instances and so the /metrics endpoint exposes exactly what we intend.
type Metrics struct {
	registry *prometheus.Registry

	requests *prometheus.CounterVec
	duration *prometheus.HistogramVec
	ttft     *prometheus.HistogramVec
	retries  *prometheus.CounterVec

	running  *prometheus.GaugeVec
	healthy  *prometheus.GaugeVec
	inflight prometheus.Gauge
}

// NewMetrics builds and registers all collectors.
func NewMetrics() *Metrics {
	registry := prometheus.NewRegistry()

	m := &Metrics{
		registry: registry,
		requests: prometheus.NewCounterVec(prometheus.CounterOpts{
			Namespace: "gateway",
			Name:      "requests_total",
			Help:      "Total requests proxied, by backend, HTTP status code and whether they streamed.",
		}, []string{"backend", "code", "stream"}),
		duration: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Namespace: "gateway",
			Name:      "request_duration_seconds",
			Help:      "End-to-end request duration as observed by the gateway.",
			Buckets:   []float64{.05, .1, .25, .5, 1, 2.5, 5, 10, 20, 30, 60, 120, 300},
		}, []string{"backend", "stream"}),
		ttft: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Namespace: "gateway",
			Name:      "ttft_seconds",
			Help:      "Time to first response byte, as observed by the gateway.",
			Buckets:   []float64{.01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10, 30, 60},
		}, []string{"backend"}),
		retries: prometheus.NewCounterVec(prometheus.CounterOpts{
			Namespace: "gateway",
			Name:      "retries_total",
			Help:      "Retries triggered after a backend failure, by failed backend.",
		}, []string{"backend"}),
		running: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Namespace: "gateway",
			Name:      "backend_running_requests",
			Help:      "In-flight requests per backend.",
		}, []string{"backend"}),
		healthy: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Namespace: "gateway",
			Name:      "backend_healthy",
			Help:      "1 if the backend is in rotation, 0 otherwise.",
		}, []string{"backend"}),
		inflight: prometheus.NewGauge(prometheus.GaugeOpts{
			Namespace: "gateway",
			Name:      "inflight_requests",
			Help:      "Requests currently being proxied by the gateway.",
		}),
	}

	registry.MustRegister(
		m.requests, m.duration, m.ttft, m.retries,
		m.running, m.healthy, m.inflight,
	)
	return m
}

// Handler serves the Prometheus exposition format.
func (m *Metrics) Handler() http.Handler {
	return promhttp.HandlerFor(m.registry, promhttp.HandlerOpts{})
}

// Registry exposes the underlying registry (used by tests).
func (m *Metrics) Registry() *prometheus.Registry { return m.registry }

// noTTFT marks "this request never produced a first byte". It is distinct from
// a measured duration of zero, which is a legitimate value on platforms whose
// clock granularity is coarser than a localhost round trip.
const noTTFT = time.Duration(-1)

// ObserveRequest records the outcome of one proxied request. Pass noTTFT when
// no first byte was ever received.
func (m *Metrics) ObserveRequest(backend string, code int, stream bool, dur, ttft time.Duration) {
	if backend == "" {
		backend = "none"
	}
	streamLabel := "false"
	if stream {
		streamLabel = "true"
	}
	m.requests.WithLabelValues(backend, statusLabel(code), streamLabel).Inc()
	m.duration.WithLabelValues(backend, streamLabel).Observe(dur.Seconds())
	if ttft >= 0 {
		m.ttft.WithLabelValues(backend).Observe(ttft.Seconds())
	}
}

// IncRetry counts a retry against the backend that failed.
func (m *Metrics) IncRetry(backend string) {
	m.retries.WithLabelValues(backend).Inc()
}

// SetRunning mirrors a backend's in-flight count into its gauge.
func (m *Metrics) SetRunning(backend string, n int64) {
	m.running.WithLabelValues(backend).Set(float64(n))
}

// SetHealthy records whether a backend is in rotation.
func (m *Metrics) SetHealthy(backend string, healthy bool) {
	v := 0.0
	if healthy {
		v = 1
	}
	m.healthy.WithLabelValues(backend).Set(v)
}

// IncInflight / DecInflight track the gateway-wide concurrency.
func (m *Metrics) IncInflight() { m.inflight.Inc() }
func (m *Metrics) DecInflight() { m.inflight.Dec() }

func statusLabel(code int) string {
	if code <= 0 {
		return "error"
	}
	return strconv.Itoa(code)
}
