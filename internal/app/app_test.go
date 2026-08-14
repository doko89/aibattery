package app

import (
	"testing"

	"github.com/aibattery/router/internal/adapter"
	"github.com/aibattery/router/internal/chat"
	"github.com/aibattery/router/internal/config"
)

// fakeBaseURL points at nothing; providers are only constructed, never called.
const fakeBaseURL = "http://127.0.0.1:1"

// minimalConfig returns a config with one provider of each supported type and
// one model referencing them.
func minimalConfig() *config.Config {
	return &config.Config{
		Providers: []config.ProviderConfig{
			{Name: "openai", Type: "openai", BaseURL: fakeBaseURL, APIKey: "k", Timeout: "1ms"},
			{Name: "anthropic", Type: "anthropic", BaseURL: fakeBaseURL, APIKey: "k", AnthropicVersion: "2023-06-01", Timeout: "1ms"},
			{Name: "gemini", Type: "gemini", BaseURL: fakeBaseURL, APIKey: "k", Timeout: "1ms"},
		},
		Models: []config.ModelConfig{
			{
				Name:     "lite",
				Strategy: "failover",
				Candidates: []config.ModelCandidate{
					{Provider: "openai", Model: "gpt-4o-mini"},
					{Provider: "anthropic", Model: "claude-haiku-4-5"},
					{Provider: "gemini", Model: "glm-4.7"},
				},
			},
		},
	}
}

func TestNewBuildsAllProviderTypes(t *testing.T) {
	a, err := New(minimalConfig())
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	if a.Handler == nil {
		t.Fatal("expected non-nil Handler")
	}
	if a.Addr == "" {
		t.Fatal("expected non-empty Addr")
	}
	if a.Config() == nil {
		t.Fatal("Config() returned nil")
	}

	// The providers map is internal; verify via the handler's behavior is not
	// possible without a server, so we re-run buildProviders to inspect types.
	providers, err := buildProviders(minimalConfig().Providers)
	if err != nil {
		t.Fatalf("buildProviders returned error: %v", err)
	}
	if len(providers) != 3 {
		t.Fatalf("expected 3 providers, got %d", len(providers))
	}

	wantKeys := []string{"openai", "anthropic", "gemini"}
	for _, k := range wantKeys {
		if _, ok := providers[k]; !ok {
			t.Errorf("missing provider key %q", k)
		}
	}

	// Verify each provider is the right concrete type via its Name() method
	// (the concrete adapter types are unexported, so we cannot type-assert).
	wantNames := map[string]string{
		"openai":    "openai",
		"anthropic": "anthropic",
		"gemini":    "gemini",
	}
	for k, want := range wantNames {
		if got := providers[k].Name(); got != want {
			t.Errorf("provider %q has Name()=%q, want %q", k, got, want)
		}
	}
}

func TestNewUnsupportedProviderType(t *testing.T) {
	cfg := minimalConfig()
	cfg.Providers[0].Type = "ollama"
	if _, err := New(cfg); err == nil {
		t.Fatal("expected error for unsupported provider type, got nil")
	}
}

func TestNewDuplicateProviderNames(t *testing.T) {
	cfg := minimalConfig()
	cfg.Providers = append(cfg.Providers, config.ProviderConfig{
		Name: "openai", Type: "openai", BaseURL: fakeBaseURL, APIKey: "k", Timeout: "1ms",
	})
	if _, err := New(cfg); err == nil {
		t.Fatal("expected error for duplicate provider name, got nil")
	}
}

func TestNewAddrFromConfig(t *testing.T) {
	cfg := minimalConfig()
	cfg.Server = config.ServerConfig{Host: "127.0.0.1", Port: 9090}
	a, err := New(cfg)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	if a.Addr != "127.0.0.1:9090" {
		t.Errorf("Addr = %q, want 127.0.0.1:9090", a.Addr)
	}
}

func TestNewAddrEnvOverridesConfig(t *testing.T) {
	t.Setenv("ROUTER_ADDR", ":9999")
	cfg := minimalConfig()
	cfg.Server = config.ServerConfig{Host: "127.0.0.1", Port: 9090}
	a, err := New(cfg)
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	if a.Addr != ":9999" {
		t.Errorf("Addr = %q, want :9999 (ROUTER_ADDR override)", a.Addr)
	}
}

func TestNewSkipsUnreachableMCPServer(t *testing.T) {
	cfg := minimalConfig()
	cfg.MCP = config.MCPSection{
		Servers: []config.MCPServer{
			{Name: "unreachable", Transport: "streamable", URL: "http://127.0.0.1:1"},
		},
	}
	a, err := New(cfg)
	if err != nil {
		t.Fatalf("New returned error for unreachable MCP server: %v", err)
	}
	if a.Handler == nil {
		t.Fatal("expected non-nil Handler despite unreachable MCP server")
	}
}

// compile-time check that the adapter constructors return chat.Provider.
var (
	_ chat.Provider = adapter.NewOpenAIProvider(fakeBaseURL, "k", 0)
	_ chat.Provider = adapter.NewAnthropicProvider(fakeBaseURL, "k", "2023-06-01", 0)
	_ chat.Provider = adapter.NewGeminiProvider(fakeBaseURL, "k", 0)
)