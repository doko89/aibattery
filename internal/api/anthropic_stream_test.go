package api

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/aibattery/router/internal/chat"
	"github.com/aibattery/router/internal/config"
	"github.com/aibattery/router/internal/routing"
)

// newAnthropicStreamServer builds a Server (not the route handler — the
// /anthropic/v1/messages route is registered elsewhere) plus the Selector for
// the virtual model, so anthropicStream can be exercised directly.
func newAnthropicStreamServer(t *testing.T, providers map[string]chat.Provider) (*Server, routing.Selector) {
	t.Helper()
	reg, err := routing.NewRegistry([]config.ModelConfig{
		{Name: "virtual-a", Strategy: "failover", Candidates: []config.ModelCandidate{{Provider: "p1", Model: "m1"}}},
	})
	if err != nil {
		t.Fatalf("NewRegistry() error = %v", err)
	}
	sel, err := reg.Select("virtual-a")
	if err != nil {
		t.Fatalf("Select() error = %v", err)
	}
	s := &Server{deps: Deps{
		Providers: providers,
		Registry:  reg,
		Logger:    slog.New(slog.NewTextHandler(io.Discard, nil)),
	}}
	return s, sel
}

func TestAnthropicStream_TextDeltas(t *testing.T) {
	p1 := &fakeProvider{
		name: "p1",
		stream: func(_ context.Context, req chat.ChatRequest, emit chat.StreamFunc) error {
			if req.Model != "m1" {
				t.Errorf("stream got model %q, want m1", req.Model)
			}
			if err := emit(chat.StreamDelta{Delta: "Hel"}); err != nil {
				return err
			}
			if err := emit(chat.StreamDelta{Delta: "lo"}); err != nil {
				return err
			}
			usage := chat.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15}
			return emit(chat.StreamDelta{FinishReason: "stop", Usage: &usage})
		},
	}
	s, sel := newAnthropicStreamServer(t, map[string]chat.Provider{"p1": p1})

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/anthropic/v1/messages", nil)
	s.anthropicStream(rec, req, sel, chat.ChatRequest{Model: "virtual-a", Stream: true})

	if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Errorf("Content-Type = %q, want text/event-stream", ct)
	}
	body := rec.Body.String()
	for _, want := range []string{
		"event: message_start",
		`"type":"message"`,
		`"id":"msg_`,
		"event: content_block_start",
		`"type":"text"`,
		`"text":"Hel"`,
		`"text":"lo"`,
		"event: content_block_stop",
		"event: message_delta",
		`"stop_reason":"end_turn"`,
		`"output_tokens":5`,
		"event: message_stop",
	} {
		if !strings.Contains(body, want) {
			t.Errorf("body missing %q:\n%s", want, body)
		}
	}
	if strings.Contains(body, "tool_use") {
		t.Errorf("body unexpectedly contains tool_use:\n%s", body)
	}
}

func TestAnthropicStream_ToolCalls(t *testing.T) {
	p1 := &fakeProvider{
		name: "p1",
		stream: func(_ context.Context, req chat.ChatRequest, emit chat.StreamFunc) error {
			if req.Model != "m1" {
				t.Errorf("stream got model %q, want m1", req.Model)
			}
			if err := emit(chat.StreamDelta{Delta: "calling"}); err != nil {
				return err
			}
			return emit(chat.StreamDelta{
				FinishReason: "tool_calls",
				ToolCalls: []chat.ToolCall{{
					ID:        "call_1",
					Name:      "search_web",
					Arguments: json.RawMessage(`{"query":"x"}`),
				}},
			})
		},
	}
	s, sel := newAnthropicStreamServer(t, map[string]chat.Provider{"p1": p1})

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/anthropic/v1/messages", nil)
	s.anthropicStream(rec, req, sel, chat.ChatRequest{Model: "virtual-a", Stream: true})

	body := rec.Body.String()
	for _, want := range []string{
		"event: message_start",
		"event: content_block_start",
		`"type":"tool_use"`,
		`"id":"call_1"`,
		`"name":"search_web"`,
		`"type":"input_json_delta"`,
		`"partial_json":"{\"query\":\"x\"}"`,
		"event: content_block_stop",
		"event: message_delta",
		`"stop_reason":"tool_use"`,
		"event: message_stop",
	} {
		if !strings.Contains(body, want) {
			t.Errorf("body missing %q:\n%s", want, body)
		}
	}
}

func TestAnthropicStream_AuthMissingKeyReturns401(t *testing.T) {
	h := newKeyedTestServer(t, map[string]chat.Provider{"p1": &fakeProvider{name: "p1"}}, "secret")

	rec := doJSON(t, h, http.MethodPost, "/anthropic/v1/messages", map[string]any{
		"model":    "virtual-a",
		"stream":   true,
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	})

	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want 401; body=%s", rec.Code, rec.Body.String())
	}
	var errResp errorResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &errResp); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if errResp.Error.Code != "invalid_api_key" || errResp.Error.Type != "authentication_error" {
		t.Errorf("error = %+v, want code invalid_api_key type authentication_error", errResp.Error)
	}
}
