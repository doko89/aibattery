package api

import (
	"crypto/subtle"
	"encoding/json"
	"log/slog"
	"net/http"
	"strings"

	"github.com/aibattery/router/internal/chat"
	"github.com/aibattery/router/internal/routing"
	toolkit "github.com/aibattery/router/internal/tools"
)

// Deps carries the dependencies the API needs to serve requests.
type Deps struct {
	Providers map[string]chat.Provider
	Registry  *routing.Registry
	Logger    *slog.Logger
	// Tools is the aggregated tool gateway (local + remote MCP tools). It may
	// be nil, in which case the /v1/tools and /v1/tools/call routes are not
	// registered and no tools are attached to chat requests.
	Tools *toolkit.Registry // may be nil
	// ClientKey optionally requires every request (except GET /health) to
	// carry `Authorization: Bearer <ClientKey>`. Empty string disables auth.
	ClientKey string
	// ReasoningCachePath optionally persists the reasoning_content echo cache
	// to this file so it survives restarts. Empty string keeps it in-memory.
	ReasoningCachePath string
}

// Server holds the resolved dependencies for the HTTP handlers.
type Server struct {
	deps Deps
}

// NewServer wires the routes and middleware chain and returns the root
// http.Handler. Middleware order is RequestID → Logging → Recovery.
func NewServer(deps Deps) http.Handler {
	if deps.Logger == nil {
		deps.Logger = slog.Default()
	}
	s := &Server{deps: deps}
	if deps.ReasoningCachePath != "" {
		reasoningByCallID = newReasoningCache(deps.ReasoningCachePath)
	}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/chat/completions", s.handleChatCompletions)
	mux.HandleFunc("POST /anthropic/v1/messages", s.handleAnthropicMessages)
	mux.HandleFunc("GET /v1/models", s.handleModels)
	mux.HandleFunc("GET /health", s.handleHealth)
	if deps.Tools != nil {
		mux.HandleFunc("GET /v1/tools", s.handleTools)
		mux.HandleFunc("POST /v1/tools/call", s.handleToolCall)
	}

	// auth skips GET /health so liveness probes stay open.
	var inner http.Handler = mux
	if deps.ClientKey != "" {
		inner = auth(deps.ClientKey)(inner)
	}
	return requestID(logging(deps.Logger)(recovery(deps.Logger)(inner)))
}

// auth enforces `Authorization: Bearer <clientKey>` on every route except
// GET /health, which stays open for liveness probes. The scheme is matched
// case-insensitively per HTTP semantics and the key compared in constant time.
func auth(clientKey string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == http.MethodGet && r.URL.Path == "/health" {
				next.ServeHTTP(w, r)
				return
			}
			parts := strings.Fields(r.Header.Get("Authorization"))
			if len(parts) != 2 || !strings.EqualFold(parts[0], "Bearer") ||
				subtle.ConstantTimeCompare([]byte(parts[1]), []byte(clientKey)) != 1 {
				writeError(w, http.StatusUnauthorized, "missing or invalid client key", "authentication_error", "invalid_api_key")
				return
			}
			next.ServeHTTP(w, r)
		})
	}
}

// handleHealth serves a simple liveness probe.
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

// writeJSON writes v as a JSON response with the given status code.
func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

// writeError writes an OpenAI-compatible error envelope.
func writeError(w http.ResponseWriter, status int, message, typ, code string) {
	writeJSON(w, status, errorResponse{Error: apiError{Message: message, Type: typ, Code: code}})
}