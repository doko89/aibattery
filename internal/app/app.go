// Package app is the COMPOSITION ROOT of the AI router. It wires the
// configuration, provider adapters, routing registry, and HTTP API together
// into a single runnable App. It is the only place that knows how every
// concrete piece fits together; the underlying packages (config, chat,
// adapter, routing, api) remain decoupled from one another.
package app

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"strings"

	"github.com/aibattery/router/internal/adapter"
	"github.com/aibattery/router/internal/api"
	"github.com/aibattery/router/internal/chat"
	"github.com/aibattery/router/internal/config"
	"github.com/aibattery/router/internal/mcp"
	"github.com/aibattery/router/internal/routing"
	"github.com/aibattery/router/internal/tools"
)

// App is the fully-wired application. It holds the resolved configuration,
// the assembled http.Handler, and the listen address. It does not listen until
// Run is called.
type App struct {
	cfg       *config.Config
	Handler   http.Handler // fully wired http.Handler (api.NewServer result)
	Addr      string
	logger    *slog.Logger
	mcClients []*mcp.Client
}

// New wires everything together from cfg and returns a ready-to-run App. It
// does not start listening. The listen address comes from cfg.Addr(), unless
// the ROUTER_ADDR environment variable is set and non-empty, in which case it
// overrides the config value.
func New(cfg *config.Config) (*App, error) {
	if cfg == nil {
		return nil, fmt.Errorf("app: nil config")
	}

	providers, err := buildProviders(cfg.Providers)
	if err != nil {
		return nil, err
	}

	registry, err := routing.NewRegistry(cfg.Models)
	if err != nil {
		return nil, fmt.Errorf("app: build registry: %w", err)
	}

	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: logLevel()}))

	tr := tools.New()
	if err := tools.RegisterBuiltin(tr); err != nil {
		return nil, fmt.Errorf("app: register builtin tools: %w", err)
	}

	mcClients, err := wireMCPServers(cfg.MCP.Servers, tr, logger)
	if err != nil {
		return nil, err
	}

	handler := api.NewServer(api.Deps{
		Providers:          providers,
		Registry:           registry,
		Logger:             logger,
		Tools:              tr,
		ClientKey:          cfg.Server.ClientKey,
		ReasoningCachePath: os.Getenv("REASONING_CACHE_FILE"),
	})

	addr := cfg.Addr()
	if env := os.Getenv("ROUTER_ADDR"); env != "" {
		addr = env
	}

	return &App{
		cfg:       cfg,
		Handler:   handler,
		Addr:      addr,
		logger:    logger,
		mcClients: mcClients,
	}, nil
}

// logLevel resolves the LOG_LEVEL env var (debug/info/warn/error) to a slog
// level; unset or unknown values default to info.
func logLevel() slog.Level {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("LOG_LEVEL"))) {
	case "debug":
		return slog.LevelDebug
	case "warn":
		return slog.LevelWarn
	case "error":
		return slog.LevelError
	default:
		return slog.LevelInfo
	}
}

// remoteCall returns a tools.RemoteFunc that proxies a tool invocation to an
// upstream MCP server. The raw JSON argument object is decoded into a map and
// forwarded; the returned text is the concatenated text content of the result.
func remoteCall(mc *mcp.Client) tools.RemoteFunc {
	return func(ctx context.Context, name string, args json.RawMessage) (string, error) {
		var am map[string]any
		if len(args) > 0 {
			_ = json.Unmarshal(args, &am)
		}
		text, _, err := mc.CallTool(ctx, name, am)
		return text, err
	}
}

// wireMCPServers connects to each configured MCP server, registers its remote
// tools into tr, and returns the connected clients for shutdown. A server that
// fails to connect is skipped (logged) rather than failing the whole app.
func wireMCPServers(servers []config.MCPServer, tr *tools.Registry, logger *slog.Logger) ([]*mcp.Client, error) {
	var clients []*mcp.Client
	for _, server := range servers {
		timeout, err := server.TimeoutDuration()
		if err != nil {
			return nil, err
		}
		httpc := &http.Client{Timeout: timeout}
		mc, err := mcp.New(mcp.Config{
			Name:        server.Name,
			URL:         server.URL,
			Transport:   server.TransportOrDefault(),
			BearerToken: server.BearerToken,
			HTTPClient:  httpc,
		})
		if err != nil {
			return nil, fmt.Errorf("app: mcp server %q: %w", server.Name, err)
		}

		ctx := context.Background()
		if err := mc.Connect(ctx); err != nil {
			logger.Warn("mcp server connect failed; skipping", "server", server.Name, "error", err)
			continue
		}

		list, err := mc.ListTools(ctx)
		if err != nil {
			logger.Warn("mcp server list tools failed; skipping", "server", server.Name, "error", err)
			continue
		}
		for _, t := range list {
			if err := tr.RegisterRemote(t, remoteCall(mc)); err != nil {
				logger.Warn("mcp tool registration skipped", "server", server.Name, "tool", t.Name, "error", err)
			}
		}
		clients = append(clients, mc)
	}
	return clients, nil
}

// buildProviders constructs a chat.Provider for each ProviderConfig, keyed by
// provider name, preserving config order. It enforces unique provider names
// and rejects unsupported provider types.
func buildProviders(providers []config.ProviderConfig) (map[string]chat.Provider, error) {
	out := make(map[string]chat.Provider, len(providers))
	for _, p := range providers {
		if _, exists := out[p.Name]; exists {
			return nil, fmt.Errorf("app: duplicate provider name %q", p.Name)
		}

		timeout, err := p.TimeoutDuration()
		if err != nil {
			return nil, err
		}

		var prov chat.Provider
		switch strings.ToLower(strings.TrimSpace(p.Type)) {
		case "openai":
			prov = adapter.NewOpenAIProvider(p.BaseURL, p.APIKey, timeout)
		case "anthropic":
			prov = adapter.NewAnthropicProvider(p.BaseURL, p.APIKey, p.AnthropicVersion, timeout)
		case "gemini":
			prov = adapter.NewGeminiProvider(p.BaseURL, p.APIKey, timeout)
		default:
			return nil, fmt.Errorf("app: unsupported provider type %q", p.Type)
		}

		out[p.Name] = prov
	}
	return out, nil
}

// Run starts the HTTP server on a.Addr and blocks until it is shut down. It
// wraps http.ErrServerClosed so a graceful shutdown is not reported as an
// error.
func (a *App) Run() error {
	srv := &http.Server{
		Addr:    a.Addr,
		Handler: a.Handler,
	}
	a.logger.Info("router listening", "addr", a.Addr)
	if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		return err
	}
	return nil
}

// Shutdown gracefully shuts down the HTTP server, waiting up to ctx's deadline
// for in-flight requests to complete, then closes any connected MCP clients.
func (a *App) Shutdown(ctx context.Context) error {
	srv := &http.Server{Handler: a.Handler}
	err := srv.Shutdown(ctx)
	for _, mc := range a.mcClients {
		if cerr := mc.Close(ctx); cerr != nil {
			a.logger.Warn("mcp client close failed", "error", cerr)
		}
	}
	return err
}

// Config returns the underlying configuration, exposed for tests and tooling.
func (a *App) Config() *config.Config { return a.cfg }