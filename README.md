# AI Router

A cross-compatible, model-aggregating AI router written in Go. It exposes two
client-facing wire formats: an OpenAI-compatible `POST /v1/chat/completions`
and an Anthropic-compatible `POST /anthropic/v1/messages`. Requests are
transparently forwarded to OpenAI, Anthropic, and Google Gemini upstreams —
speaking each provider's **native** auth and wire format while presenting a
unified surface to your clients.

## Features

- **Cross-compatible router** — OpenAI-compatible (`/v1/chat/completions`) and
  Anthropic-compatible (`/anthropic/v1/messages`) APIs in front of OpenAI,
  Anthropic, and Gemini. Native auth per provider (`Authorization: Bearer`,
  `x-api-key` + `anthropic-version`, `x-goog-api-key`) and request/response
  translation into each provider's canonical wire format.
- **Anthropic-compatible endpoint** — `POST /anthropic/v1/messages` speaks the
  Anthropic Messages API wire, including SSE events (`message_start`,
  `content_block_start`, `content_block_delta`, `content_block_stop`,
  `message_delta`, `message_stop`) and `tool_use` content blocks with
  `input_json_delta`. Anthropic-native clients (Claude Code, SDKs) connect by
  setting `ANTHROPIC_BASE_URL=https://host/anthropic` — the SDK appends
  `/v1/messages`.
- **Client key auth** — optional `server.client_key` in config. When set, every
  request except `GET /health` requires `Authorization: Bearer <client_key>`;
  a missing or wrong key returns 401 with code `invalid_api_key` (constant-time
  compare). Empty `client_key` disables auth.
- **`reasoning_effort`** — `POST /v1/chat/completions` accepts an optional
  `reasoning_effort` field with lowercase values
  `none`/`minimal`/`low`/`medium`/`high`/`xhigh`/`max`. Invalid values get a
  400 (`invalid_reasoning_effort`) before any upstream call. Mapped per
  provider: passthrough for OpenAI (temperature dropped), thinking for
  Anthropic, thinkingConfig for Gemini.
- **Aggregated virtual models** — a single virtual model name maps to an
  ordered list of concrete `{provider, model}` candidates.
- **Failover strategy** — candidates are attempted in config order; a failed
  candidate is skipped for a **30s cooldown** before it is retried.
- **Round-robin strategy** — candidates rotate per request for load spreading,
  with the same failure-aware cooldown.
- **Streaming** — SSE streaming supported end-to-end for all three providers.
- **Graceful shutdown** — drains in-flight requests on SIGINT/SIGTERM.

## Architecture (Domain-Driven Design)

```
internal/
├── chat/            # DOMAIN PORT — the chat.Provider interface + canonical
│                    #   ChatRequest / ChatResponse / StreamDelta model.
├── adapter/         # INFRASTRUCTURE — concrete providers translating the
│                    #   canonical model to each provider's native wire:
│                    #   openai.go, anthropic.go, gemini.go.
├── routing/         # MODEL-AGGREGATION bounded context — virtual model →
│                    #   ordered candidate list with failover / round_robin
│                    #   strategies and 30s failure cooldown.
├── api/             # PRESENTATION — HTTP handlers + middleware (RequestID,
│                    #   Logging, Recovery, client key auth). OpenAI and
│                    #   Anthropic client-facing wires.
└── app/             # COMPOSITION ROOT — wires config → providers → registry
                    #   → handler into a runnable App.
```

Dependencies point inward: `api` → `routing`/`chat`, `adapter` implements
`chat.Provider`, and `app` is the only place that knows how everything fits
together.

## Configuration

Configuration lives in a YAML file (default `config.yaml`, override with the
`ROUTER_CONFIG` env var). API keys are referenced via `${ENV_VAR}` and expanded
at load time.

```yaml
server:
  host: "0.0.0.0"
  port: 8080
  # Optional static client key; when set, all requests except GET /health
  # require "Authorization: Bearer <client_key>". Empty disables auth.
  client_key: "sk-your-client-key"

providers:
  - name: openai
    type: openai
    base_url: https://api.openai.com/v1
    api_key: ${OPENAI_API_KEY}
    timeout: "60s"

  - name: anthropic
    type: anthropic
    base_url: https://api.anthropic.com/v1
    api_key: ${ANTHROPIC_API_KEY}
    anthropic_version: "2023-06-01"
    timeout: "120s"

  - name: gemini
    type: gemini
    base_url: https://generativelanguage.googleapis.com/v1beta
    api_key: ${GEMINI_API_KEY}
    timeout: "120s"

models:
  - name: lite
    strategy: failover
    candidates:
      - provider: openai
        model: gpt-4o-mini
      - provider: anthropic
        model: claude-haiku-4-5
      - provider: gemini
        model: glm-4.7

  - name: fast
    strategy: round_robin
    candidates:
      - provider: openai
        model: deepseek-v4-flash
      - provider: anthropic
        model: claude-haiku-4-5
      - provider: gemini
        model: glm-4.7
```

## Environment Variables

| Variable        | Purpose                                            | Default     |
|-----------------|----------------------------------------------------|-------------|
| `OPENAI_API_KEY`   | OpenAI provider key (referenced in config)      | —           |
| `ANTHROPIC_API_KEY`| Anthropic provider key (referenced in config)   | —           |
| `GEMINI_API_KEY`   | Gemini provider key (referenced in config)      | —           |
| `ROUTER_ADDR`      | Optional override for the listen address        | config `server.host`/`server.port` |
| `ROUTER_CONFIG`    | Path to the YAML config file                    | `config.yaml` |

`server.client_key` is a config-file key (not an env var): an optional static
client key; empty disables auth.

The listen address is set via the `server.host` and `server.port` keys in the
config file (empty host defaults to `0.0.0.0`, port `0` defaults to `8080`).
Setting the `ROUTER_ADDR` env var overrides the config value.

## Usage

```sh
go run ./cmd/server
```

### Non-streaming request

```sh
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <client_key>" \
  -d '{
    "model": "lite",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

The `Authorization: Bearer <client_key>` header is required when
`server.client_key` is set; without it, auth is disabled. `GET /health` stays
unauthenticated either way.

### Streaming request

```sh
curl -N http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <client_key>" \
  -d '{
    "model": "fast",
    "stream": true,
    "messages": [{"role": "user", "content": "Tell me a story"}]
  }'
```

### Anthropic-compatible usage

The router also speaks the Anthropic Messages API at
`POST /anthropic/v1/messages`:

```sh
curl -N http://localhost:8080/anthropic/v1/messages \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <client_key>" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "fast",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

The auth header is the same one the middleware checks everywhere:
`Authorization: Bearer <client_key>` (only when `server.client_key` is set).

Anthropic-native clients (Claude Code, Anthropic SDKs) connect by pointing
their base URL at the router: `ANTHROPIC_BASE_URL=https://host/anthropic` (the
SDK appends `/v1/messages`).

### Health check

```sh
curl http://localhost:8080/health
```

## Notes

- The router exposes **two client-facing wire shapes**: the OpenAI
  chat-completions shape at `/v1/chat/completions` and the Anthropic Messages
  shape at `/anthropic/v1/messages`. Pick whichever matches your client;
  internally both are translated to the canonical model and then to each
  upstream's native wire.
- `reasoning_effort` (OpenAI wire only): optional field on
  `/v1/chat/completions`, valid lowercase values
  `none`/`minimal`/`low`/`medium`/`high`/`xhigh`/`max` (case-sensitive). An
  invalid value returns 400 with code `invalid_reasoning_effort` before any
  upstream call.
- A provider whose `timeout` is empty defaults to 60s.

## Tools / MCP

The router aggregates tools from local Go executors and remote MCP servers into
a single gateway, exposes them over HTTP, and passes the whole set to upstream
providers for function calling.

### Configuration

Remote MCP servers are declared under `mcp.servers`. Each entry supports either
the `streamable` (default) or `sse` transport:

```yaml
mcp:
  servers:
    - name: my-tools
      transport: streamable
      url: https://tools.example.com/mcp
      bearer_token: ${MCP_BEARER_TOKEN}
      timeout: "30s"

    - name: legacy-tools
      transport: sse
      url: https://legacy.example.com/sse
      timeout: "60s"
```

A server that fails to connect at startup is skipped (logged) rather than
failing the whole app.

### Built-in local tools

Two local tools are always registered:

- `echo_text` — echoes back the provided `text` argument.
- `get_utc_time` — returns the current UTC time as ISO 8601.

### Endpoints

- `GET /v1/tools` — returns the aggregated tool set in the OpenAI
  function-calling shape (`{"object":"list","data":[...]}`).
- `POST /v1/tools/call` — executes a tool by name. Body:
  `{"name":"echo_text","arguments":{"text":"hi"}}`. Returns an MCP-style
  `CallResult` (`{"content":[{"type":"text","text":"..."}],"isError":false}`).

### Function calling

The aggregated tool set is attached to every chat request, so upstream models
see the full tool list and can request tool invocations via function calling.