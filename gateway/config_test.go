package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func writeConfig(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

const minimalConfig = `
listen: ":9100"
strategy: round_robin
backends:
  - name: a
    url: http://127.0.0.1:8000
    weight: 2
  - name: b
    url: http://127.0.0.1:8001
`

func TestLoadConfigAppliesDefaults(t *testing.T) {
	cfg, err := LoadConfig(writeConfig(t, minimalConfig))
	if err != nil {
		t.Fatal(err)
	}

	if cfg.Listen != ":9100" || cfg.Strategy != "round_robin" {
		t.Fatalf("unexpected basics: %+v", cfg)
	}
	if cfg.Timeouts.StreamFirstByte.D() != 30*time.Second {
		t.Fatalf("stream_first_byte default = %v, want 30s", cfg.Timeouts.StreamFirstByte.D())
	}
	if cfg.Timeouts.Request.D() != 60*time.Second {
		t.Fatalf("request default = %v, want 60s", cfg.Timeouts.Request.D())
	}
	if cfg.Retry.Max != 0 {
		t.Fatalf("retry.max default = %d, want 0 (retries are opt-in)", cfg.Retry.Max)
	}
	if cfg.HealthCheck.UnhealthyAfter != 2 || cfg.HealthCheck.HealthyAfter != 2 {
		t.Fatalf("health thresholds = %d/%d, want 2/2", cfg.HealthCheck.UnhealthyAfter, cfg.HealthCheck.HealthyAfter)
	}
	if cfg.Backends[1].Weight != 1 {
		t.Fatalf("missing weight should default to 1, got %d", cfg.Backends[1].Weight)
	}
}

func TestLoadConfigRejectsUnknownFields(t *testing.T) {
	// A typo like "backends_" must be a hard error, not silently ignored.
	bad := strings.Replace(minimalConfig, "backends:", "backends_:", 1)
	if _, err := LoadConfig(writeConfig(t, bad)); err == nil {
		t.Fatal("expected an error for an unknown field")
	}
}

func TestLoadConfigRejectsDuplicateNames(t *testing.T) {
	dup := strings.Replace(minimalConfig, "name: b", "name: a", 1)
	if _, err := LoadConfig(writeConfig(t, dup)); err == nil {
		t.Fatal("expected an error for duplicate backend names")
	}
}

func TestLoadConfigRejectsBadURL(t *testing.T) {
	bad := strings.Replace(minimalConfig, "http://127.0.0.1:8000", "127.0.0.1:8000", 1)
	if _, err := LoadConfig(writeConfig(t, bad)); err == nil {
		t.Fatal("expected an error for a URL without a scheme")
	}
}

func TestLoadConfigRejectsUnknownStrategy(t *testing.T) {
	bad := strings.Replace(minimalConfig, "strategy: round_robin", "strategy: random", 1)
	if _, err := LoadConfig(writeConfig(t, bad)); err == nil {
		t.Fatal("expected an error for an unknown strategy")
	}
}

func TestLoadConfigDurationParsing(t *testing.T) {
	bad := strings.Replace(minimalConfig, "strategy: round_robin", "strategy: round_robin\ntimeouts:\n  request: 60", 1)
	if _, err := LoadConfig(writeConfig(t, bad)); err == nil {
		t.Fatal("expected an error for a bare number duration")
	}
}

func TestEnvOverridesBackends(t *testing.T) {
	t.Setenv("GATEWAY_BACKENDS", "gpu-a=http://10.0.0.1:8000,gpu-b=http://10.0.0.2:8000")

	cfg, err := LoadConfig(writeConfig(t, minimalConfig))
	if err != nil {
		t.Fatal(err)
	}
	if len(cfg.Backends) != 2 {
		t.Fatalf("got %d backends, want 2", len(cfg.Backends))
	}
	if cfg.Backends[0].Name != "gpu-a" || cfg.Backends[1].URL != "http://10.0.0.2:8000" {
		t.Fatalf("unexpected backends: %+v", cfg.Backends)
	}
}

func TestEnvOverridesBareURLList(t *testing.T) {
	t.Setenv("GATEWAY_BACKENDS", "http://10.0.0.1:8000,http://10.0.0.2:8000")

	cfg, err := LoadConfig(writeConfig(t, minimalConfig))
	if err != nil {
		t.Fatal(err)
	}
	if len(cfg.Backends) != 2 || cfg.Backends[0].Name != "env-1" {
		t.Fatalf("unexpected backends: %+v", cfg.Backends)
	}
}

func TestConfigRequiresBackends(t *testing.T) {
	body := "listen: \":9000\"\n"
	if _, err := LoadConfig(writeConfig(t, body)); err == nil {
		t.Fatal("expected an error when no backends are configured")
	}
}
