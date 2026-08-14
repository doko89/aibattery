// Package config loads and validates the router's YAML configuration. It
// supports ${ENV_VAR} substitution for provider API keys so secrets can be
// kept out of the config file.
package config

import (
	"fmt"
	"os"
	"regexp"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// Config is the root configuration document.
type Config struct {
	Server    ServerConfig    `yaml:"server"`
	MCP       MCPSection      `yaml:"mcp"`
	Providers []ProviderConfig `yaml:"providers"`
	Models    []ModelConfig    `yaml:"models"`
}

// ServerConfig describes the HTTP listen address. An empty Host defaults to
// "0.0.0.0" and a Port of 0 defaults to 8080.
type ServerConfig struct {
	Host string `yaml:"host"`
	Port int    `yaml:"port"`
	// ClientKey is an optional static client key. When non-empty, the API
	// requires Authorization: Bearer <client_key> on /v1/* routes; empty
	// disables auth.
	ClientKey string `yaml:"client_key"`
}

// MCPSection lists the MCP servers (tool providers) the MCP client may attach to.
type MCPSection struct {
	Servers []MCPServer `yaml:"servers"`
}

// MCPServer describes a remote MCP server.
type MCPServer struct {
	Name        string `yaml:"name"`
	Transport   string `yaml:"transport"`   // "streamable" | "sse" (default "streamable")
	URL         string `yaml:"url"`
	BearerToken string `yaml:"bearer_token"` // optional; attached as Authorization: Bearer
	Timeout     string `yaml:"timeout"`      // optional duration string
}

// TransportOrDefault returns the effective transport: the lowercase trimmed
// value of m.Transport, or "streamable" when empty.
func (m MCPServer) TransportOrDefault() string {
	t := strings.ToLower(strings.TrimSpace(m.Transport))
	if t == "" {
		return "streamable"
	}
	return t
}

// TimeoutDuration parses m.Timeout with time.ParseDuration. An empty string
// returns the default of 30 seconds. An invalid value returns a descriptive
// error.
func (m MCPServer) TimeoutDuration() (time.Duration, error) {
	if m.Timeout == "" {
		return 30 * time.Second, nil
	}
	d, err := time.ParseDuration(m.Timeout)
	if err != nil {
		return 0, fmt.Errorf("config: mcp server %q has invalid timeout %q: %w", m.Name, m.Timeout, err)
	}
	return d, nil
}

// ProviderConfig describes a single upstream provider that adapters connect to.
type ProviderConfig struct {
	Name             string `yaml:"name"`
	Type             string `yaml:"type"`
	BaseURL          string `yaml:"base_url"`
	APIKey           string `yaml:"api_key"`
	AnthropicVersion string `yaml:"anthropic_version"`
	// Timeout is an optional per-provider HTTP timeout as a duration string
	// (e.g. "30s" or "2m"). Empty means the default of 60s.
	Timeout string `yaml:"timeout"`
}

// TimeoutDuration parses p.Timeout with time.ParseDuration. An empty string
// returns the default of 60 seconds. An invalid value returns a descriptive
// error.
func (p ProviderConfig) TimeoutDuration() (time.Duration, error) {
	if p.Timeout == "" {
		return 60 * time.Second, nil
	}
	d, err := time.ParseDuration(p.Timeout)
	if err != nil {
		return 0, fmt.Errorf("config: provider %q has invalid timeout %q: %w", p.Name, p.Timeout, err)
	}
	return d, nil
}

// ModelConfig describes an aggregated model exposed by the router. Strategy is
// one of "failover" or "round_robin". Cooldown is an optional per-model failure
// cooldown window as a duration string (e.g. "30s" or "2m"); empty means the
// default of 30s.
type ModelConfig struct {
	Name       string           `yaml:"name"`
	Strategy   string           `yaml:"strategy"`
	Cooldown   string           `yaml:"cooldown"`
	Candidates []ModelCandidate `yaml:"candidates"`
}

// CooldownDuration parses m.Cooldown with time.ParseDuration. An empty string
// returns the default of 30 seconds. An invalid value returns a descriptive
// error.
func (m ModelConfig) CooldownDuration() (time.Duration, error) {
	if m.Cooldown == "" {
		return 30 * time.Second, nil
	}
	d, err := time.ParseDuration(m.Cooldown)
	if err != nil {
		return 0, fmt.Errorf("config: model %q has invalid cooldown %q: %w", m.Name, m.Cooldown, err)
	}
	return d, nil
}

// ModelCandidate references a concrete provider+model pair that a ModelConfig
// may route to.
type ModelCandidate struct {
	Provider string `yaml:"provider"`
	Model    string `yaml:"model"`
}

// envRef matches a ${ENV_VAR} reference inside a string.
var envRef = regexp.MustCompile(`\$\{([A-Za-z_][A-Za-z0-9_]*)\}`)

// Load reads the YAML file at path, unmarshals it into a Config, expands
// ${ENV_VAR} references in every ProviderConfig.APIKey, and validates the
// result.
func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("config: read %s: %w", path, err)
	}

	var cfg Config
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("config: parse %s: %w", path, err)
	}

	cfg.expandEnv()

	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	return &cfg, nil
}

// expandEnv replaces every ${ENV_VAR} reference in each provider's APIKey and
// each MCP server's BearerToken/URL with the value of the corresponding
// environment variable. A referenced variable that is set-but-empty expands to
// the empty string (so the router still loads for reads-most); a missing
// variable also expands to an empty string.
func (c *Config) expandEnv() {
	for i := range c.Providers {
		c.Providers[i].APIKey = expandEnv(c.Providers[i].APIKey)
	}
	for i := range c.MCP.Servers {
		c.MCP.Servers[i].BearerToken = expandEnv(c.MCP.Servers[i].BearerToken)
		c.MCP.Servers[i].URL = expandEnv(c.MCP.Servers[i].URL)
	}
}

// expandEnv substitutes ${ENV_VAR} references in s using os.Getenv.
func expandEnv(s string) string {
	return envRef.ReplaceAllStringFunc(s, func(m string) string {
		name := m[2 : len(m)-1] // strip "${" and "}"
		return os.Getenv(name)
	})
}

// Addr returns the effective listen address as "host:port". An empty Host
// defaults to "0.0.0.0" and a Port of 0 defaults to 8080.
func (c *Config) Addr() string {
	host := c.Server.Host
	if host == "" {
		host = "0.0.0.0"
	}
	port := c.Server.Port
	if port == 0 {
		port = 8080
	}
	return fmt.Sprintf("%s:%d", host, port)
}

// Validate checks the configuration for structural correctness. It returns an
// error if there are no providers, no models, a model candidate references an
// unknown provider, a model strategy is not one of "failover" or
// "round_robin", a provider has a non-empty Timeout that fails to parse, or
// server.port is set to a value outside 1-65535.
func (c *Config) Validate() error {
	if c.Server.Port != 0 && (c.Server.Port < 1 || c.Server.Port > 65535) {
		return fmt.Errorf("config: server.port must be 1-65535, got %d", c.Server.Port)
	}
	if len(c.Providers) == 0 {
		return fmt.Errorf("config: at least one provider is required")
	}
	if len(c.Models) == 0 {
		return fmt.Errorf("config: at least one model is required")
	}

	known := make(map[string]bool, len(c.Providers))
	for _, p := range c.Providers {
		if p.Name == "" {
			return fmt.Errorf("config: provider with empty name")
		}
		if _, err := p.TimeoutDuration(); err != nil {
			return err
		}
		known[p.Name] = true
	}

	for _, s := range c.MCP.Servers {
		if s.Name == "" {
			return fmt.Errorf("config: mcp server with empty name")
		}
		if s.URL == "" {
			return fmt.Errorf("config: mcp server %q has empty url", s.Name)
		}
		switch s.TransportOrDefault() {
		case "streamable", "sse":
		default:
			return fmt.Errorf("config: mcp server %q has unsupported transport %q (want \"streamable\" or \"sse\")", s.Name, s.Transport)
		}
		if _, err := s.TimeoutDuration(); err != nil {
			return err
		}
	}

	for _, m := range c.Models {
		if m.Name == "" {
			return fmt.Errorf("config: model with empty name")
		}
		switch m.Strategy {
		case "failover", "round_robin":
		default:
			return fmt.Errorf("config: model %q has unsupported strategy %q (want \"failover\" or \"round_robin\")", m.Name, m.Strategy)
		}
		if _, err := m.CooldownDuration(); err != nil {
			return err
		}
		for _, cand := range m.Candidates {
			if !known[cand.Provider] {
				return fmt.Errorf("config: model %q references unknown provider %q", m.Name, cand.Provider)
			}
		}
	}
	return nil
}
