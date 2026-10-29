package main

import (
	"bytes"
	"fmt"
	"net/url"
	"os"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// Duration is a time.Duration that decodes from YAML strings such as "10s".
type Duration time.Duration

// D returns the underlying time.Duration.
func (d Duration) D() time.Duration { return time.Duration(d) }

// UnmarshalYAML implements yaml.Unmarshaler so the config file can use
// human-readable values ("250ms", "30s") instead of raw nanoseconds.
func (d *Duration) UnmarshalYAML(value *yaml.Node) error {
	var s string
	if err := value.Decode(&s); err != nil {
		return fmt.Errorf("duration must be a string like \"30s\": %w", err)
	}
	parsed, err := time.ParseDuration(s)
	if err != nil {
		return fmt.Errorf("invalid duration %q: %w", s, err)
	}
	*d = Duration(parsed)
	return nil
}

// Config is the gateway configuration, loaded from config.yaml.
type Config struct {
	Listen      string            `yaml:"listen"`
	Strategy    string            `yaml:"strategy"`
	LogLevel    string            `yaml:"log_level"`
	Backends    []BackendConfig   `yaml:"backends"`
	HealthCheck HealthCheckConfig `yaml:"health_check"`
	Timeouts    TimeoutConfig     `yaml:"timeouts"`
	Retry       RetryConfig       `yaml:"retry"`
	RateLimit   RateLimitConfig   `yaml:"rate_limit"`
}

// BackendConfig describes one upstream vLLM instance.
type BackendConfig struct {
	Name   string `yaml:"name"`
	URL    string `yaml:"url"`
	Weight int    `yaml:"weight"`
}

// HealthCheckConfig controls active probing of upstreams.
type HealthCheckConfig struct {
	Interval       Duration `yaml:"interval"`
	Path           string   `yaml:"path"`
	Timeout        Duration `yaml:"timeout"`
	UnhealthyAfter int      `yaml:"unhealthy_after"`
	HealthyAfter   int      `yaml:"healthy_after"`
}

// TimeoutConfig separates the three timeouts that matter for an LLM gateway.
//
// There is deliberately no single global request timeout: a streaming
// generation can legitimately take minutes, while the first token must arrive
// quickly or the request should be retried elsewhere.
type TimeoutConfig struct {
	Connect         Duration `yaml:"connect"`
	Request         Duration `yaml:"request"`
	StreamFirstByte Duration `yaml:"stream_first_byte"`
	Shutdown        Duration `yaml:"shutdown"`
}

// RetryConfig controls retry-after-failure behaviour.
type RetryConfig struct {
	Max             int      `yaml:"max"`
	Backoff         Duration `yaml:"backoff"`
	MaxBackoff      Duration `yaml:"max_backoff"`
	MaxBodyBytes    int64    `yaml:"max_body_bytes"`
	MaxResponseBody int64    `yaml:"max_response_body_bytes"`
}

// RateLimitConfig is an optional global token bucket in front of the pool.
type RateLimitConfig struct {
	RPS   float64 `yaml:"rps"`
	Burst int     `yaml:"burst"`
}

// LoadConfig reads and validates a YAML config file.
func LoadConfig(path string) (*Config, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read config %q: %w", path, err)
	}

	cfg := &Config{}
	dec := yaml.NewDecoder(bytes.NewReader(raw))
	dec.KnownFields(true) // reject typos instead of silently ignoring them
	if err := dec.Decode(cfg); err != nil {
		return nil, fmt.Errorf("parse config %q: %w", path, err)
	}

	cfg.applyEnvOverrides()
	cfg.applyDefaults()
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return cfg, nil
}

// applyEnvOverrides lets container deployments override the config without
// rebuilding the image. Backends may be given as "name=url,name=url" or as a
// bare comma-separated list of URLs.
func (c *Config) applyEnvOverrides() {
	if v := strings.TrimSpace(os.Getenv("GATEWAY_LISTEN")); v != "" {
		c.Listen = v
	}
	raw := strings.TrimSpace(os.Getenv("GATEWAY_BACKENDS"))
	if raw == "" {
		return
	}
	backends := make([]BackendConfig, 0, 4)
	for i, entry := range strings.Split(raw, ",") {
		entry = strings.TrimSpace(entry)
		if entry == "" {
			continue
		}
		name, target := fmt.Sprintf("env-%d", i+1), entry
		if k, v, ok := strings.Cut(entry, "="); ok {
			name, target = strings.TrimSpace(k), strings.TrimSpace(v)
		}
		backends = append(backends, BackendConfig{Name: name, URL: target, Weight: 1})
	}
	if len(backends) > 0 {
		c.Backends = backends
	}
}

func (c *Config) applyDefaults() {
	if c.Listen == "" {
		c.Listen = ":9000"
	}
	if c.Strategy == "" {
		c.Strategy = string(LeastConn)
	}
	if c.LogLevel == "" {
		c.LogLevel = "info"
	}

	if c.HealthCheck.Interval == 0 {
		c.HealthCheck.Interval = Duration(10 * time.Second)
	}
	if c.HealthCheck.Path == "" {
		c.HealthCheck.Path = "/health"
	}
	if c.HealthCheck.Timeout == 0 {
		c.HealthCheck.Timeout = Duration(2 * time.Second)
	}
	if c.HealthCheck.UnhealthyAfter <= 0 {
		c.HealthCheck.UnhealthyAfter = 2
	}
	if c.HealthCheck.HealthyAfter <= 0 {
		c.HealthCheck.HealthyAfter = 2
	}

	if c.Timeouts.Connect == 0 {
		c.Timeouts.Connect = Duration(5 * time.Second)
	}
	if c.Timeouts.Request == 0 {
		c.Timeouts.Request = Duration(60 * time.Second)
	}
	if c.Timeouts.StreamFirstByte == 0 {
		c.Timeouts.StreamFirstByte = Duration(30 * time.Second)
	}
	if c.Timeouts.Shutdown == 0 {
		c.Timeouts.Shutdown = Duration(15 * time.Second)
	}

	if c.Retry.Max < 0 {
		c.Retry.Max = 0
	}
	if c.Retry.Backoff == 0 {
		c.Retry.Backoff = Duration(100 * time.Millisecond)
	}
	if c.Retry.MaxBackoff == 0 {
		c.Retry.MaxBackoff = Duration(1 * time.Second)
	}
	if c.Retry.MaxBodyBytes == 0 {
		c.Retry.MaxBodyBytes = 8 << 20 // 8 MiB
	}
	if c.Retry.MaxResponseBody == 0 {
		c.Retry.MaxResponseBody = 16 << 20 // 16 MiB
	}
	if c.RateLimit.Burst == 0 && c.RateLimit.RPS > 0 {
		c.RateLimit.Burst = int(c.RateLimit.RPS * 2)
	}
}

// Validate checks the invariants that would otherwise fail at request time.
func (c *Config) Validate() error {
	if len(c.Backends) == 0 {
		return fmt.Errorf("config: at least one backend is required")
	}

	seen := make(map[string]struct{}, len(c.Backends))
	for i := range c.Backends {
		b := &c.Backends[i]
		if b.Name == "" {
			return fmt.Errorf("config: backend #%d has no name", i+1)
		}
		if _, dup := seen[b.Name]; dup {
			return fmt.Errorf("config: duplicate backend name %q", b.Name)
		}
		seen[b.Name] = struct{}{}

		u, err := url.Parse(b.URL)
		if err != nil {
			return fmt.Errorf("config: backend %q has invalid url: %w", b.Name, err)
		}
		if u.Scheme != "http" && u.Scheme != "https" {
			return fmt.Errorf("config: backend %q url must start with http:// or https://", b.Name)
		}
		if u.Host == "" {
			return fmt.Errorf("config: backend %q url has no host", b.Name)
		}
		if b.Weight == 0 {
			b.Weight = 1
		}
		if b.Weight < 0 {
			return fmt.Errorf("config: backend %q weight must be positive", b.Name)
		}
	}

	switch Strategy(c.Strategy) {
	case RoundRobin, LeastConn:
	default:
		return fmt.Errorf("config: unknown strategy %q (want %q or %q)", c.Strategy, RoundRobin, LeastConn)
	}

	if c.Timeouts.Request <= 0 || c.Timeouts.StreamFirstByte <= 0 {
		return fmt.Errorf("config: timeouts must be positive")
	}
	if c.HealthCheck.Interval <= 0 {
		return fmt.Errorf("config: health_check.interval must be positive")
	}
	return nil
}
