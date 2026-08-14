package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// writeTemp writes content to a temp file and returns its path.
func writeTemp(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatalf("write temp config: %v", err)
	}
	return path
}

const validYAML = `
providers:
  - name: openai
    type: openai
    base_url: https://api.openai.com/v1
    api_key: ${OPENAI_API_KEY}
  - name: anthropic
    type: anthropic
    base_url: https://api.anthropic.com/v1
    api_key: ${ANTHROPIC_API_KEY}
    anthropic_version: "2023-06-01"
models:
  - name: lite
    strategy: failover
    candidates:
      - provider: openai
        model: gpt-4o-mini
      - provider: anthropic
        model: claude-haiku-4-5
`

func TestLoad(t *testing.T) {
	path := writeTemp(t, validYAML)
	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(cfg.Providers) != 2 {
		t.Fatalf("expected 2 providers, got %d", len(cfg.Providers))
	}
	if len(cfg.Models) != 1 {
		t.Fatalf("expected 1 model, got %d", len(cfg.Models))
	}
	if cfg.Providers[1].AnthropicVersion != "2023-06-01" {
		t.Errorf("anthropic_version = %q, want 2023-06-01", cfg.Providers[1].AnthropicVersion)
	}
	if cfg.Models[0].Candidates[0].Model != "gpt-4o-mini" {
		t.Errorf("candidate model = %q, want gpt-4o-mini", cfg.Models[0].Candidates[0].Model)
	}
}

func TestLoadEnvSubstitution(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "sk-test-123")
	t.Setenv("ANTHROPIC_API_KEY", "ant-test-456")

	path := writeTemp(t, validYAML)
	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got := cfg.Providers[0].APIKey; got != "sk-test-123" {
		t.Errorf("openai api_key = %q, want sk-test-123", got)
	}
	if got := cfg.Providers[1].APIKey; got != "ant-test-456" {
		t.Errorf("anthropic api_key = %q, want ant-test-456", got)
	}
}

func TestLoadEnvMissingExpandsToEmpty(t *testing.T) {
	// Ensure the env vars are unset so expansion yields empty strings.
	os.Unsetenv("OPENAI_API_KEY")
	os.Unsetenv("ANTHROPIC_API_KEY")

	path := writeTemp(t, validYAML)
	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("Load should not fail on missing env vars: %v", err)
	}
	if cfg.Providers[0].APIKey != "" {
		t.Errorf("missing env should expand to empty, got %q", cfg.Providers[0].APIKey)
	}
}

func TestLoadEnvSetButEmpty(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("ANTHROPIC_API_KEY", "")

	path := writeTemp(t, validYAML)
	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("Load should not fail on set-but-empty env vars: %v", err)
	}
	if cfg.Providers[0].APIKey != "" {
		t.Errorf("set-but-empty env should expand to empty, got %q", cfg.Providers[0].APIKey)
	}
}

func TestLoadMissingFile(t *testing.T) {
	if _, err := Load(filepath.Join(t.TempDir(), "nope.yaml")); err == nil {
		t.Fatal("expected error for missing file")
	}
}

func TestLoadInvalidYAML(t *testing.T) {
	path := writeTemp(t, "providers: [unclosed")
	if _, err := Load(path); err == nil {
		t.Fatal("expected error for invalid yaml")
	}
}

func TestValidateNoProviders(t *testing.T) {
	cfg := &Config{Models: []ModelConfig{{Name: "m", Strategy: "failover"}}}
	if err := cfg.Validate(); err == nil {
		t.Fatal("expected error for no providers")
	}
}

func TestValidateNoModels(t *testing.T) {
	cfg := &Config{Providers: []ProviderConfig{{Name: "p"}}}
	if err := cfg.Validate(); err == nil {
		t.Fatal("expected error for no models")
	}
}

func TestValidateUnknownProvider(t *testing.T) {
	cfg := &Config{
		Providers: []ProviderConfig{{Name: "openai"}},
		Models: []ModelConfig{{
			Name:     "m",
			Strategy: "failover",
			Candidates: []ModelCandidate{
				{Provider: "nope", Model: "x"},
			},
		}},
	}
	err := cfg.Validate()
	if err == nil {
		t.Fatal("expected error for unknown provider")
	}
	if !strings.Contains(err.Error(), "unknown provider") {
		t.Errorf("error should mention unknown provider, got: %v", err)
	}
}

func TestValidateBadStrategy(t *testing.T) {
	cfg := &Config{
		Providers: []ProviderConfig{{Name: "openai"}},
		Models: []ModelConfig{{
			Name:     "m",
			Strategy: "random",
			Candidates: []ModelCandidate{
				{Provider: "openai", Model: "x"},
			},
		}},
	}
	err := cfg.Validate()
	if err == nil {
		t.Fatal("expected error for bad strategy")
	}
	if !strings.Contains(err.Error(), "strategy") {
		t.Errorf("error should mention strategy, got: %v", err)
	}
}

func TestValidateOK(t *testing.T) {
	cfg := &Config{
		Providers: []ProviderConfig{{Name: "openai"}},
		Models: []ModelConfig{{
			Name:     "m",
			Strategy: "round_robin",
			Candidates: []ModelCandidate{
				{Provider: "openai", Model: "x"},
			},
		}},
	}
	if err := cfg.Validate(); err != nil {
		t.Fatalf("expected valid config, got: %v", err)
	}
}

func TestTimeoutDurationValid(t *testing.T) {
	p := ProviderConfig{Name: "openai", Timeout: "30s"}
	d, err := p.TimeoutDuration()
	if err != nil {
		t.Fatalf("TimeoutDuration: %v", err)
	}
	if d != 30*time.Second {
		t.Errorf("duration = %v, want 30s", d)
	}
}

func TestTimeoutDurationEmptyDefaultsTo60s(t *testing.T) {
	p := ProviderConfig{Name: "openai"}
	d, err := p.TimeoutDuration()
	if err != nil {
		t.Fatalf("TimeoutDuration: %v", err)
	}
	if d != 60*time.Second {
		t.Errorf("duration = %v, want 60s", d)
	}
}

func TestTimeoutDurationInvalid(t *testing.T) {
	p := ProviderConfig{Name: "openai", Timeout: "abc"}
	if _, err := p.TimeoutDuration(); err == nil {
		t.Fatal("expected error for invalid timeout")
	}
}

func TestModelCooldownDuration(t *testing.T) {
	// Empty cooldown defaults to 30s.
	m := ModelConfig{Name: "m"}
	d, err := m.CooldownDuration()
	if err != nil {
		t.Fatalf("CooldownDuration: %v", err)
	}
	if d != 30*time.Second {
		t.Errorf("duration = %v, want 30s", d)
	}

	// Explicit cooldown parses.
	m = ModelConfig{Name: "m", Cooldown: "2s"}
	d, err = m.CooldownDuration()
	if err != nil {
		t.Fatalf("CooldownDuration: %v", err)
	}
	if d != 2*time.Second {
		t.Errorf("duration = %v, want 2s", d)
	}

	// Invalid cooldown returns a descriptive error.
	m = ModelConfig{Name: "m", Cooldown: "abc"}
	_, err = m.CooldownDuration()
	if err == nil {
		t.Fatal("expected error for invalid cooldown")
	}
	if !strings.Contains(err.Error(), "invalid cooldown") {
		t.Errorf("error should mention invalid cooldown, got: %v", err)
	}
}

func TestValidateInvalidTimeout(t *testing.T) {
	cfg := &Config{
		Providers: []ProviderConfig{{Name: "openai", Timeout: "abc"}},
		Models: []ModelConfig{{
			Name:     "m",
			Strategy: "failover",
			Candidates: []ModelCandidate{
				{Provider: "openai", Model: "x"},
			},
		}},
	}
	err := cfg.Validate()
	if err == nil {
		t.Fatal("expected error for invalid provider timeout")
	}
	if !strings.Contains(err.Error(), "timeout") {
		t.Errorf("error should mention timeout, got: %v", err)
	}
}

func TestValidServerAddr(t *testing.T) {
	tests := []struct {
		name string
		host string
		port int
		want string
	}{
		{name: "empty host and zero port default", host: "", port: 0, want: "0.0.0.0:8080"},
		{name: "explicit host and port", host: "127.0.0.1", port: 9090, want: "127.0.0.1:9090"},
		{name: "empty host with explicit port", host: "", port: 9090, want: "0.0.0.0:9090"},
		{name: "explicit host with zero port", host: "127.0.0.1", port: 0, want: "127.0.0.1:8080"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := &Config{Server: ServerConfig{Host: tt.host, Port: tt.port}}
			if got := cfg.Addr(); got != tt.want {
				t.Errorf("Addr() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestServerPortInvalid(t *testing.T) {
	cfg := &Config{
		Server:    ServerConfig{Host: "0.0.0.0", Port: 65536},
		Providers: []ProviderConfig{{Name: "openai"}},
		Models: []ModelConfig{{
			Name:     "m",
			Strategy: "failover",
			Candidates: []ModelCandidate{
				{Provider: "openai", Model: "x"},
			},
		}},
	}
	err := cfg.Validate()
	if err == nil {
		t.Fatal("expected error for invalid server port")
	}
	if !strings.Contains(err.Error(), "server.port must be 1-65535") {
		t.Errorf("error should mention server.port range, got: %v", err)
	}
}

func TestMCPEnvExpansion(t *testing.T) {
	t.Setenv("MCP_SERVER_TOKEN", "mcp-secret-789")
	t.Setenv("MCP_SERVER_URL", "http://127.0.0.1:18091/mcp")

	path := writeTemp(t, `
providers:
  - name: openai
    type: openai
    base_url: https://api.openai.com/v1
    api_key: ${OPENAI_API_KEY}
models:
  - name: m
    strategy: failover
    candidates:
      - provider: openai
        model: gpt-4o-mini
mcp:
  servers:
    - name: tools
      transport: streamable
      url: ${MCP_SERVER_URL}
      bearer_token: ${MCP_SERVER_TOKEN}
`)
	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(cfg.MCP.Servers) != 1 {
		t.Fatalf("expected 1 mcp server, got %d", len(cfg.MCP.Servers))
	}
	if got := cfg.MCP.Servers[0].BearerToken; got != "mcp-secret-789" {
		t.Errorf("bearer_token = %q, want mcp-secret-789", got)
	}
	if got := cfg.MCP.Servers[0].URL; got != "http://127.0.0.1:18091/mcp" {
		t.Errorf("url = %q, want expanded url", got)
	}
}

func TestMCSTransportDefault(t *testing.T) {
	m := MCPServer{Name: "tools"}
	if got := m.TransportOrDefault(); got != "streamable" {
		t.Errorf("TransportOrDefault() = %q, want streamable", got)
	}
	m = MCPServer{Name: "tools", Transport: "  SSE  "}
	if got := m.TransportOrDefault(); got != "sse" {
		t.Errorf("TransportOrDefault() = %q, want sse", got)
	}
}

func TestMCPTimeoutDefault(t *testing.T) {
	m := MCPServer{Name: "tools"}
	d, err := m.TimeoutDuration()
	if err != nil {
		t.Fatalf("TimeoutDuration: %v", err)
	}
	if d != 30*time.Second {
		t.Errorf("duration = %v, want 30s", d)
	}
}

func TestMCPInvalidTransport(t *testing.T) {
	cfg := &Config{
		Providers: []ProviderConfig{{Name: "openai"}},
		Models: []ModelConfig{{
			Name:     "m",
			Strategy: "failover",
			Candidates: []ModelCandidate{
				{Provider: "openai", Model: "x"},
			},
		}},
		MCP: MCPSection{Servers: []MCPServer{{Name: "tools", Transport: "http", URL: "http://127.0.0.1:18091/mcp"}}},
	}
	err := cfg.Validate()
	if err == nil {
		t.Fatal("expected error for invalid mcp transport")
	}
	if !strings.Contains(err.Error(), "transport") {
		t.Errorf("error should mention transport, got: %v", err)
	}
}

func TestMCPInvalidTimeout(t *testing.T) {
	cfg := &Config{
		Providers: []ProviderConfig{{Name: "openai"}},
		Models: []ModelConfig{{
			Name:     "m",
			Strategy: "failover",
			Candidates: []ModelCandidate{
				{Provider: "openai", Model: "x"},
			},
		}},
		MCP: MCPSection{Servers: []MCPServer{
			{Name: "tools", Transport: "streamable", URL: "http://127.0.0.1:18091/mcp", Timeout: "abc"},
		}},
	}
	err := cfg.Validate()
	if err == nil {
		t.Fatal("expected error for invalid mcp timeout")
	}
	if !strings.Contains(err.Error(), "timeout") {
		t.Errorf("error should mention timeout, got: %v", err)
	}
}
