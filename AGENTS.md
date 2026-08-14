# AGENTS.md — AI Router (aibattery)

Go 1.26 AI router/proxy: two client-facing wires in front of OpenAI, Anthropic,
and Gemini upstreams — OpenAI `POST /v1/chat/completions` and Anthropic
`POST /anthropic/v1/messages`. Module: `github.com/aibattery/router`.
Stdlib-only `net/http` (no Fiber/Gin) — the sole dependency is `gopkg.in/yaml.v3`.

## Commands

- `go run ./cmd/server` — run (needs `config.yaml` + provider keys; `ROUTER_CONFIG` overrides path, `ROUTER_ADDR` overrides listen addr)
- `go build ./...` / `go test ./...` — build and test (96 tests, 7 packages, all green). No Makefile.
- `go vet ./...` — clean.

## Architecture (DDD — dependencies point inward)

```
config → app (composition root) → adapter ── implements ──> chat (domain port)
                                       ↑                        ↑
api (HTTP) → routing (model registry) ──┘                        └── mcp, tools
```

- `internal/chat/` — DOMAIN PORT. `Provider` interface + canonical `ChatRequest`/`ChatResponse`/`StreamDelta`/`Tool`/`ToolCall` types. Pure Go types, zero deps; every other package depends on these. No tests (types only).
- `internal/adapter/` — `openai.go`, `anthropic.go`, `gemini.go` translate the canonical model ↔ each provider's native wire (auth headers, request bodies, SSE frames, finish-reason mapping). Each has its own `_test.go` with httptest fake upstreams.
- `internal/routing/` — virtual model → ordered candidates; strategies `failover` / `round_robin`; failure cooldown. Deliberately decoupled from `chat` (imports only `config` + stdlib).
- `internal/api/` — handlers + middleware (`RequestID → Logging → Recovery`, plus client key auth when `server.client_key` is set). Two client-facing wires: OpenAI `/v1/chat/completions` (`chat_completions.go`, `response.go`, `sse.go`) and Anthropic `/anthropic/v1/messages` (`anthropic_messages.go`, `anthropic_stream.go`), both always registered. Auth gates every route except `GET /health`: `Authorization: Bearer <client_key>`, constant-time compare (`crypto/subtle`), 401 with envelope code `invalid_api_key`. OpenAI wire errors via `writeError`. Routes registered with Go 1.22 method patterns on `http.NewServeMux`.
- `internal/app/` — composition root; the only place that wires config → providers → registry → tools → MCP → handler.
- `internal/mcp/` — MCP client (transports: `streamable`, `sse`), JSON-RPC 2.0, `initialize` handshake.
- `internal/tools/` — concurrency-safe tool registry: local Go executors + tools proxied to remote MCP servers.

## Critical contracts (breaking these breaks tests / behavior)

- **`chat.Provider.Stream`**: return an error ONLY if the stream failed before any content was delivered. Errors after the first delta must be emitted as `StreamDelta{FinishReason: "error"}` and the method must return nil. This is asserted by tests (`TestAnthropicStream_ErrorAfterText`, `TestGeminiStream_ErrorAfterFirstDelta`).
- **Final stream chunk** carries `FinishReason` + cumulative `Usage`; tool calls appear only on the final chunk with `FinishReason == "tool_calls"`. Gemini reports `STOP` even when emitting function calls — adapters must remap to canonical `"tool_calls"`.
- **`reasoning_effort` validation** lives in `internal/api/chat_completions.go` (OpenAI wire only): valid lowercase values `none`/`minimal`/`low`/`medium`/`high`/`xhigh`/`max` (case-sensitive), invalid → 400 with code `invalid_reasoning_effort` BEFORE any upstream call. Per-provider mapping lives in adapters: OpenAI passthrough + drop temperature, Anthropic thinking (enabled+budget legacy ≤4.5 / adaptive 4.6+, budget < max_tokens), Gemini thinkingConfig (thinkingLevel 3.x + temperature forced 1.0, thinkingBudget 2.x).
- **Tool call serialization per wire**: OpenAI wire carries `arguments` as a JSON **string** on the final chunk (`finish_reason: "tool_calls"`, see `api/response.go` + `api/sse.go`); Anthropic wire emits `tool_use` content blocks with an `input` object and `stop_reason: "tool_use"` (see `api/anthropic_messages.go` + `api/anthropic_stream.go`).
- **Routing cooldown is a hardcoded `30s`** via package-level `var now = time.Now` in `internal/routing/routing.go`. Tests override `routing.now` to simulate time. Keep that seam.

## Gotchas

- **Config `${ENV_VAR}` expansion** happens ONLY in provider `api_key` and MCP `bearer_token`/`url` fields. A missing/empty var expands to `""` — the router still loads (reads-most deployments). Don't expect expansion elsewhere.
- **MCP connect failure is non-fatal**: a server that fails to connect or list tools is skipped with a warning log. But *invalid* config (bad timeout string, unknown transport, empty URL) fails startup. `wireMCPServers` (internal/app/app.go) has no covering tests — be careful when editing.
- **Routes `GET /v1/tools` + `POST /v1/tools/call` are registered only when `Deps.Tools != nil`** (see `api.NewServer`). Tool call errors: unknown tool → 404 with code `tool_not_found`; execution failure → 502.
- **Defaults**: provider timeout 60s, MCP server timeout 30s (empty string → default, via `TimeoutDuration()` on each config type).
- **Built-in tools are always registered**: `echo_text` (defaults arg to `"abc"`) and `get_utc_time`. Tool names must be unique across local + remote — duplicates are skipped with a warning (MCP) or error (local).
- **Tests**: httptest fake upstreams for adapters; fake `chat.Provider` implementations for API handler tests (`handler_test.go` plus `anthropic_messages_test.go` + `anthropic_stream_test.go`); naming `TestXxx_Behavior`. Run `go test -count=1 ./...` to bypass cache.
- **`.gitignore`d files**: `anthropic-api.md`, `gemini-api.md`, `openai-compatible-api.md` are wire-format reference notes — not build inputs. `.omo/` is session artifacts; ignore.
- **Graceful shutdown** on SIGINT/SIGTERM: drains in-flight requests, then closes MCP clients.
