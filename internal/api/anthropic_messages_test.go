package api

import (
	"context"
	"encoding/json"
	"net/http"
	"strings"
	"testing"

	"github.com/aibattery/router/internal/chat"
	"github.com/aibattery/router/internal/config"
	toolkit "github.com/aibattery/router/internal/tools"
)

// anthropicTestMessage mirrors the Anthropic non-streaming response shape for
// assertions.
type anthropicTestMessage struct {
	ID         string               `json:"id"`
	Type       string               `json:"type"`
	Role       string               `json:"role"`
	Model      string               `json:"model"`
	Content    []anthropicTestBlock `json:"content"`
	StopReason string               `json:"stop_reason"`
	Usage      struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

type anthropicTestBlock struct {
	Type  string         `json:"type"`
	Text  string         `json:"text"`
	ID    string         `json:"id"`
	Name  string         `json:"name"`
	Input map[string]any `json:"input"`
}

// anthropicTestError mirrors the Anthropic error envelope for assertions.
type anthropicTestError struct {
	Type  string `json:"type"`
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

func anthropicTestServer(t *testing.T, providers map[string]chat.Provider) http.Handler {
	t.Helper()
	return newTestServer(t, providers, []config.ModelConfig{
		{Name: "virtual-a", Strategy: "failover", Candidates: []config.ModelCandidate{{Provider: "p1", Model: "m1"}}},
	})
}

func TestAnthropicMessages_NonStreamHappyPath(t *testing.T) {
	p1 := &fakeProvider{
		name: "p1",
		complete: func(_ context.Context, req chat.ChatRequest) (chat.ChatResponse, error) {
			if req.Model != "m1" {
				t.Errorf("Complete got model %q, want m1", req.Model)
			}
			return chat.ChatResponse{
				ID:           "resp-1",
				Model:        "m1",
				Content:      "hi",
				FinishReason: "stop",
				Usage:        chat.Usage{PromptTokens: 5, CompletionTokens: 3, TotalTokens: 8},
			}, nil
		},
	}
	h := anthropicTestServer(t, map[string]chat.Provider{"p1": p1})

	rec := doJSON(t, h, http.MethodPost, "/anthropic/v1/messages", map[string]any{
		"model":      "virtual-a",
		"max_tokens": 1024,
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
	})

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	var out anthropicTestMessage
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if out.Type != "message" {
		t.Errorf("type = %q, want message", out.Type)
	}
	if out.Role != "assistant" {
		t.Errorf("role = %q, want assistant", out.Role)
	}
	if out.Model != "m1" {
		t.Errorf("model = %q, want m1", out.Model)
	}
	if len(out.Content) != 1 || out.Content[0].Type != "text" || out.Content[0].Text != "hi" {
		t.Errorf("content = %+v, want [{text hi}]", out.Content)
	}
	if out.StopReason != "end_turn" {
		t.Errorf("stop_reason = %q, want end_turn", out.StopReason)
	}
	if out.Usage.InputTokens != 5 || out.Usage.OutputTokens != 3 {
		t.Errorf("usage = %+v, want input_tokens 5 output_tokens 3", out.Usage)
	}
}

func TestAnthropicMessages_ToolCalls(t *testing.T) {
	p1 := &fakeProvider{
		name: "p1",
		complete: func(context.Context, chat.ChatRequest) (chat.ChatResponse, error) {
			return chat.ChatResponse{
				FinishReason: "tool_calls",
				ToolCalls: []chat.ToolCall{{
					ID:        "call_1",
					Name:      "search_web",
					Arguments: json.RawMessage(`{"query":"news"}`),
				}},
			}, nil
		},
	}
	h := anthropicTestServer(t, map[string]chat.Provider{"p1": p1})

	rec := doJSON(t, h, http.MethodPost, "/anthropic/v1/messages", map[string]any{
		"model":      "virtual-a",
		"max_tokens": 1024,
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
	})

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	var out anthropicTestMessage
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if out.StopReason != "tool_use" {
		t.Errorf("stop_reason = %q, want tool_use", out.StopReason)
	}
	if len(out.Content) != 1 {
		t.Fatalf("content len = %d, want 1 tool_use block", len(out.Content))
	}
	blk := out.Content[0]
	if blk.Type != "tool_use" {
		t.Errorf("content[0].type = %q, want tool_use", blk.Type)
	}
	if blk.ID != "call_1" {
		t.Errorf("content[0].id = %q, want call_1", blk.ID)
	}
	if blk.Name != "search_web" {
		t.Errorf("content[0].name = %q, want search_web", blk.Name)
	}
	if blk.Input["query"] != "news" {
		t.Errorf("content[0].input = %v, want query news", blk.Input)
	}
}

func TestAnthropic_ServerToolExecuted(t *testing.T) {
	var gotArgs json.RawMessage
	tr := toolkit.New()
	if err := tr.RegisterLocal(chat.Tool{Name: "search_web", Description: "search", InputSchema: map[string]any{"type": "object"}}, func(_ context.Context, args json.RawMessage) (string, error) {
		gotArgs = append(json.RawMessage(nil), args...)
		return "cerah 32C", nil
	}); err != nil {
		t.Fatalf("RegisterLocal() error = %v", err)
	}
	p1 := &fakeProvider{
		name: "p1",
		scripted: []chat.ChatResponse{
			{
				FinishReason: "tool_calls",
				ToolCalls: []chat.ToolCall{{
					ID:        "call_1",
					Name:      "search_web",
					Arguments: json.RawMessage(`{"query":"cuaca"}`),
				}},
			},
			{Content: "suhu cerah 32C", FinishReason: "stop"},
		},
	}
	h := newToolTestServer(t, map[string]chat.Provider{"p1": p1}, []config.ModelConfig{
		{Name: "virtual-a", Strategy: "failover", Candidates: []config.ModelCandidate{{Provider: "p1", Model: "m1"}}},
	}, tr)

	rec := doJSON(t, h, http.MethodPost, "/anthropic/v1/messages", map[string]any{
		"model":      "virtual-a",
		"max_tokens": 1024,
		"messages":   []map[string]any{{"role": "user", "content": "cuaca hari ini?"}},
	})

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	var out anthropicTestMessage
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if out.StopReason != "end_turn" {
		t.Errorf("stop_reason = %q, want end_turn", out.StopReason)
	}
	if len(out.Content) != 1 || out.Content[0].Type != "text" || !strings.Contains(out.Content[0].Text, "32C") {
		t.Errorf("content = %+v, want text block containing 32C", out.Content)
	}
	if string(gotArgs) != `{"query":"cuaca"}` {
		t.Errorf("tool args = %s, want {\"query\":\"cuaca\"}", gotArgs)
	}
	if len(p1.requests) != 2 {
		t.Fatalf("provider called %d times, want 2", len(p1.requests))
	}
	msgs := p1.requests[1].Messages
	if len(msgs) < 2 {
		t.Fatalf("second request messages len = %d, want >= 2", len(msgs))
	}
	asst := msgs[len(msgs)-2]
	if asst.Role != chat.RoleAssistant || len(asst.ToolCalls) != 1 || asst.ToolCalls[0].ID != "call_1" {
		t.Errorf("assistant message = %+v, want role assistant with tool call call_1", asst)
	}
	toolMsg := msgs[len(msgs)-1]
	if toolMsg.Role != chat.RoleTool || toolMsg.ToolCallID != "call_1" || toolMsg.Content != "cerah 32C" {
		t.Errorf("tool message = %+v, want role tool, call id call_1, content cerah 32C", toolMsg)
	}
}

func TestAnthropicMessages_MaxTokensDefault(t *testing.T) {
	var got chat.ChatRequest
	p1 := &fakeProvider{
		name: "p1",
		complete: func(_ context.Context, req chat.ChatRequest) (chat.ChatResponse, error) {
			got = req
			return chat.ChatResponse{Content: "ok", FinishReason: "stop"}, nil
		},
	}
	h := anthropicTestServer(t, map[string]chat.Provider{"p1": p1})

	rec := doJSON(t, h, http.MethodPost, "/anthropic/v1/messages", map[string]any{
		"model":    "virtual-a",
		"messages": []map[string]any{{"role": "user", "content": "hi"}},
	})

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	if got.MaxTokens == nil || *got.MaxTokens != 1024 {
		t.Errorf("MaxTokens = %v, want 1024", got.MaxTokens)
	}
}

func TestAnthropicMessages_ContentAsArray(t *testing.T) {
	var got chat.ChatRequest
	p1 := &fakeProvider{
		name: "p1",
		complete: func(_ context.Context, req chat.ChatRequest) (chat.ChatResponse, error) {
			got = req
			return chat.ChatResponse{Content: "ok", FinishReason: "stop"}, nil
		},
	}
	h := anthropicTestServer(t, map[string]chat.Provider{"p1": p1})

	rec := doJSON(t, h, http.MethodPost, "/anthropic/v1/messages", map[string]any{
		"model":      "virtual-a",
		"max_tokens": 1024,
		"messages":   []map[string]any{{"role": "user", "content": []map[string]any{{"type": "text", "text": "hello"}}}},
	})

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	if len(got.Messages) != 1 || got.Messages[0].Content != "hello" {
		t.Errorf("messages = %+v, want content hello", got.Messages)
	}
}

func TestAnthropicMessages_ModelNotFoundReturns404(t *testing.T) {
	h := anthropicTestServer(t, map[string]chat.Provider{"p1": &fakeProvider{name: "p1"}})

	rec := doJSON(t, h, http.MethodPost, "/anthropic/v1/messages", map[string]any{
		"model":      "nope",
		"max_tokens": 1024,
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
	})

	if rec.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404; body=%s", rec.Code, rec.Body.String())
	}
	var errResp anthropicTestError
	if err := json.Unmarshal(rec.Body.Bytes(), &errResp); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if errResp.Type != "error" {
		t.Errorf("type = %q, want error", errResp.Type)
	}
	if errResp.Error.Type != "invalid_request_error" {
		t.Errorf("error.type = %q, want invalid_request_error", errResp.Error.Type)
	}
}

func TestAnthropicMessages_AuthRequired(t *testing.T) {
	h := newKeyedTestServer(t, map[string]chat.Provider{"p1": &fakeProvider{name: "p1"}}, "secret")

	rec := doJSON(t, h, http.MethodPost, "/anthropic/v1/messages", map[string]any{
		"model":      "virtual-a",
		"max_tokens": 1024,
		"messages":   []map[string]any{{"role": "user", "content": "hi"}},
	})

	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want 401", rec.Code)
	}
}
