package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/aibattery/router/internal/chat"
	"github.com/aibattery/router/internal/config"
	"github.com/aibattery/router/internal/routing"
	toolkit "github.com/aibattery/router/internal/tools"
)

// fakeProvider is a canned chat.Provider for tests.
type fakeProvider struct {
	name     string
	complete func(ctx context.Context, req chat.ChatRequest) (chat.ChatResponse, error)
	stream   func(ctx context.Context, req chat.ChatRequest, emit chat.StreamFunc) error
}

func (f *fakeProvider) Name() string { return f.name }

func (f *fakeProvider) Complete(ctx context.Context, req chat.ChatRequest) (chat.ChatResponse, error) {
	if f.complete != nil {
		return f.complete(ctx, req)
	}
	return chat.ChatResponse{}, errors.New("fakeProvider: Complete not implemented")
}

func (f *fakeProvider) Stream(ctx context.Context, req chat.ChatRequest, emit chat.StreamFunc) error {
	if f.stream != nil {
		return f.stream(ctx, req, emit)
	}
	return errors.New("fakeProvider: Stream not implemented")
}

// newTestServer builds a Server wired with the given providers and models.
func newTestServer(t *testing.T, providers map[string]chat.Provider, models []config.ModelConfig) http.Handler {
	t.Helper()
	reg, err := routing.NewRegistry(models)
	if err != nil {
		t.Fatalf("NewRegistry() error = %v", err)
	}
	return NewServer(Deps{
		Providers: providers,
		Registry:  reg,
		Logger:    slog.New(slog.NewTextHandler(io.Discard, nil)),
	})
}

func doJSON(t *testing.T, h http.Handler, method, path string, body any) *httptest.ResponseRecorder {
	t.Helper()
	var rdr io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			t.Fatalf("marshal body: %v", err)
		}
		rdr = bytes.NewReader(b)
	}
	req := httptest.NewRequest(method, path, rdr)
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func TestNonStreamSuccess(t *testing.T) {
	p1 := &fakeProvider{
		name: "p1",
		complete: func(_ context.Context, req chat.ChatRequest) (chat.ChatResponse, error) {
			if req.Model != "m1" {
				t.Errorf("Complete got model %q, want m1", req.Model)
			}
			return chat.ChatResponse{
				ID:           "resp-1",
				Model:        "m1",
				Content:      "hello from p1",
				FinishReason: "stop",
				Usage:        chat.Usage{PromptTokens: 5, CompletionTokens: 3, TotalTokens: 8},
			}, nil
		},
	}
	h := newTestServer(t, map[string]chat.Provider{"p1": p1}, []config.ModelConfig{
		{Name: "virtual-a", Strategy: "failover", Candidates: []config.ModelCandidate{{Provider: "p1", Model: "m1"}}},
	})

	rec := doJSON(t, h, http.MethodPost, "/v1/chat/completions", map[string]any{
		"model":    "virtual-a",
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	})

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	var out chatCompletionResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if out.Object != "chat.completion" {
		t.Errorf("object = %q, want chat.completion", out.Object)
	}
	if out.Model != "m1" {
		t.Errorf("model = %q, want m1", out.Model)
	}
	if len(out.Choices) != 1 {
		t.Fatalf("choices len = %d, want 1", len(out.Choices))
	}
	if out.Choices[0].Message.Content != "hello from p1" {
		t.Errorf("content = %q, want hello from p1", out.Choices[0].Message.Content)
	}
	if out.Choices[0].Message.Role != "assistant" {
		t.Errorf("role = %q, want assistant", out.Choices[0].Message.Role)
	}
	if out.Choices[0].FinishReason != "stop" {
		t.Errorf("finish_reason = %q, want stop", out.Choices[0].FinishReason)
	}
	if out.Usage == nil || out.Usage.TotalTokens != 8 {
		t.Errorf("usage = %+v, want total 8", out.Usage)
	}
}

func TestFailoverToSecondCandidate(t *testing.T) {
	p1 := &fakeProvider{
		name: "p1",
		complete: func(context.Context, chat.ChatRequest) (chat.ChatResponse, error) {
			return chat.ChatResponse{}, errors.New("p1 down")
		},
	}
	p2 := &fakeProvider{
		name: "p2",
		complete: func(_ context.Context, req chat.ChatRequest) (chat.ChatResponse, error) {
			if req.Model != "m2" {
				t.Errorf("got model %q, want m2", req.Model)
			}
			return chat.ChatResponse{Content: "from p2", FinishReason: "stop"}, nil
		},
	}
	h := newTestServer(t, map[string]chat.Provider{"p1": p1, "p2": p2}, []config.ModelConfig{
		{Name: "virtual-a", Strategy: "failover", Candidates: []config.ModelCandidate{
			{Provider: "p1", Model: "m1"},
			{Provider: "p2", Model: "m2"},
		}},
	})

	rec := doJSON(t, h, http.MethodPost, "/v1/chat/completions", map[string]any{
		"model":    "virtual-a",
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	})

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	var out chatCompletionResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if out.Model != "m2" {
		t.Errorf("model = %q, want m2 (second candidate)", out.Model)
	}
	if out.Choices[0].Message.Content != "from p2" {
		t.Errorf("content = %q, want from p2", out.Choices[0].Message.Content)
	}
}

func TestAllCandidatesFailReturns502(t *testing.T) {
	p1 := &fakeProvider{
		name: "p1",
		complete: func(context.Context, chat.ChatRequest) (chat.ChatResponse, error) {
			return chat.ChatResponse{}, errors.New("p1 down")
		},
	}
	h := newTestServer(t, map[string]chat.Provider{"p1": p1}, []config.ModelConfig{
		{Name: "virtual-a", Strategy: "failover", Candidates: []config.ModelCandidate{{Provider: "p1", Model: "m1"}}},
	})

	rec := doJSON(t, h, http.MethodPost, "/v1/chat/completions", map[string]any{
		"model":    "virtual-a",
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	})

	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502; body=%s", rec.Code, rec.Body.String())
	}
	var errResp errorResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &errResp); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if errResp.Error.Type != "upstream_error" {
		t.Errorf("error type = %q, want upstream_error", errResp.Error.Type)
	}
}

func TestUnknownModelReturns404(t *testing.T) {
	p1 := &fakeProvider{name: "p1"}
	h := newTestServer(t, map[string]chat.Provider{"p1": p1}, []config.ModelConfig{
		{Name: "virtual-a", Strategy: "failover", Candidates: []config.ModelCandidate{{Provider: "p1", Model: "m1"}}},
	})

	rec := doJSON(t, h, http.MethodPost, "/v1/chat/completions", map[string]any{
		"model":    "nope",
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	})

	if rec.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404; body=%s", rec.Code, rec.Body.String())
	}
	var errResp errorResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &errResp); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if errResp.Error.Code != "model_not_found" {
		t.Errorf("error code = %q, want model_not_found", errResp.Error.Code)
	}
}

func TestMissingModelReturns400(t *testing.T) {
	p1 := &fakeProvider{name: "p1"}
	h := newTestServer(t, map[string]chat.Provider{"p1": p1}, []config.ModelConfig{
		{Name: "virtual-a", Strategy: "failover", Candidates: []config.ModelCandidate{{Provider: "p1", Model: "m1"}}},
	})

	rec := doJSON(t, h, http.MethodPost, "/v1/chat/completions", map[string]any{
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	})

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400; body=%s", rec.Code, rec.Body.String())
	}
}

func TestStreaming(t *testing.T) {
	p1 := &fakeProvider{
		name: "p1",
		stream: func(_ context.Context, req chat.ChatRequest, emit chat.StreamFunc) error {
			if req.Model != "m1" {
				t.Errorf("stream got model %q, want m1", req.Model)
			}
			if err := emit(chat.StreamDelta{Delta: "Hello"}); err != nil {
				return err
			}
			usage := chat.Usage{PromptTokens: 5, CompletionTokens: 2, TotalTokens: 7}
			return emit(chat.StreamDelta{Delta: " world", FinishReason: "stop", Usage: &usage})
		},
	}
	h := newTestServer(t, map[string]chat.Provider{"p1": p1}, []config.ModelConfig{
		{Name: "virtual-a", Strategy: "failover", Candidates: []config.ModelCandidate{{Provider: "p1", Model: "m1"}}},
	})

	rec := doJSON(t, h, http.MethodPost, "/v1/chat/completions", map[string]any{
		"model":    "virtual-a",
		"stream":   true,
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	})

	if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Errorf("Content-Type = %q, want text/event-stream", ct)
	}
	body := rec.Body.String()
	if !strings.Contains(body, "data: [DONE]") {
		t.Errorf("body missing [DONE]: %q", body)
	}
	if !strings.Contains(body, `"object":"chat.completion.chunk"`) {
		t.Errorf("body missing chunk object: %q", body)
	}
	if !strings.Contains(body, `"content":"Hello"`) {
		t.Errorf("body missing first delta content: %q", body)
	}
	if !strings.Contains(body, `"role":"assistant"`) {
		t.Errorf("body missing assistant role on first chunk: %q", body)
	}
	if !strings.Contains(body, `"finish_reason":"stop"`) {
		t.Errorf("body missing finish_reason stop: %q", body)
	}
	if !strings.Contains(body, `"total_tokens":7`) {
		t.Errorf("body missing usage total_tokens: %q", body)
	}
}

func TestNonStreamToolCalls(t *testing.T) {
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
	h := newTestServer(t, map[string]chat.Provider{"p1": p1}, []config.ModelConfig{
		{Name: "virtual-a", Strategy: "failover", Candidates: []config.ModelCandidate{{Provider: "p1", Model: "m1"}}},
	})

	rec := doJSON(t, h, http.MethodPost, "/v1/chat/completions", map[string]any{
		"model":    "virtual-a",
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	})

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	var out chatCompletionResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	tc := out.Choices[0].Message.ToolCalls
	if len(tc) != 1 {
		t.Fatalf("tool_calls len = %d, want 1", len(tc))
	}
	if tc[0].ID != "call_1" {
		t.Errorf("tool_call id = %q, want call_1", tc[0].ID)
	}
	if tc[0].Type != "function" {
		t.Errorf("tool_call type = %q, want function", tc[0].Type)
	}
	if tc[0].Function.Name != "search_web" {
		t.Errorf("function name = %q, want search_web", tc[0].Function.Name)
	}
	// arguments must be a JSON string on the wire, not an object
	if tc[0].Function.Arguments != `{"query":"news"}` {
		t.Errorf("arguments = %q, want %q", tc[0].Function.Arguments, `{"query":"news"}`)
	}
}

func TestStreamingToolCalls(t *testing.T) {
	p1 := &fakeProvider{
		name: "p1",
		stream: func(_ context.Context, req chat.ChatRequest, emit chat.StreamFunc) error {
			if err := emit(chat.StreamDelta{Delta: "Hello"}); err != nil {
				return err
			}
			return emit(chat.StreamDelta{
				FinishReason: "tool_calls",
				ToolCalls: []chat.ToolCall{{
					ID:        "call_1",
					Name:      "search_web",
					Arguments: json.RawMessage(`{"query":"news"}`),
				}},
			})
		},
	}
	h := newTestServer(t, map[string]chat.Provider{"p1": p1}, []config.ModelConfig{
		{Name: "virtual-a", Strategy: "failover", Candidates: []config.ModelCandidate{{Provider: "p1", Model: "m1"}}},
	})

	rec := doJSON(t, h, http.MethodPost, "/v1/chat/completions", map[string]any{
		"model":    "virtual-a",
		"stream":   true,
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	})

	body := rec.Body.String()
	if !strings.Contains(body, `"tool_calls":[{"function":{"arguments":"{\"query\":\"news\"}","name":"search_web"},"id":"call_1","type":"function"}]`) {
		t.Errorf("body missing tool_calls delta: %q", body)
	}
	if !strings.Contains(body, `"finish_reason":"tool_calls"`) {
		t.Errorf("body missing finish_reason tool_calls: %q", body)
	}
	// content-only chunks must not carry tool_calls
	for _, line := range strings.Split(body, "\n") {
		if !strings.Contains(line, `"content":"Hello"`) {
			continue
		}
		if strings.Contains(line, "tool_calls") {
			t.Errorf("content chunk unexpectedly carries tool_calls: %q", line)
		}
	}
}

func TestStreamingFailoverBeforeContent(t *testing.T) {
	p1 := &fakeProvider{
		name: "p1",
		stream: func(context.Context, chat.ChatRequest, chat.StreamFunc) error {
			return errors.New("p1 stream down")
		},
	}
	p2 := &fakeProvider{
		name: "p2",
		stream: func(_ context.Context, req chat.ChatRequest, emit chat.StreamFunc) error {
			if req.Model != "m2" {
				t.Errorf("stream got model %q, want m2", req.Model)
			}
			if err := emit(chat.StreamDelta{Delta: "recovered"}); err != nil {
				return err
			}
			return emit(chat.StreamDelta{FinishReason: "stop"})
		},
	}
	h := newTestServer(t, map[string]chat.Provider{"p1": p1, "p2": p2}, []config.ModelConfig{
		{Name: "virtual-a", Strategy: "failover", Candidates: []config.ModelCandidate{
			{Provider: "p1", Model: "m1"},
			{Provider: "p2", Model: "m2"},
		}},
	})

	rec := doJSON(t, h, http.MethodPost, "/v1/chat/completions", map[string]any{
		"model":    "virtual-a",
		"stream":   true,
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	})

	body := rec.Body.String()
	if !strings.Contains(body, "data: [DONE]") {
		t.Fatalf("body missing [DONE]: %q", body)
	}
	if !strings.Contains(body, `"content":"recovered"`) {
		t.Errorf("body missing recovered content from second candidate: %q", body)
	}
}

func TestModelsListsNames(t *testing.T) {
	p1 := &fakeProvider{name: "p1"}
	h := newTestServer(t, map[string]chat.Provider{"p1": p1}, []config.ModelConfig{
		{Name: "zebra", Strategy: "failover", Candidates: []config.ModelCandidate{{Provider: "p1", Model: "m1"}}},
		{Name: "alpha", Strategy: "failover", Candidates: []config.ModelCandidate{{Provider: "p1", Model: "m1"}}},
	})

	rec := doJSON(t, h, http.MethodGet, "/v1/models", nil)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	var out modelListResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode models: %v", err)
	}
	if out.Object != "list" {
		t.Errorf("object = %q, want list", out.Object)
	}
	if len(out.Data) != 2 {
		t.Fatalf("data len = %d, want 2", len(out.Data))
	}
	ids := []string{out.Data[0].ID, out.Data[1].ID}
	if ids[0] != "alpha" || ids[1] != "zebra" {
		t.Errorf("model ids = %v, want [alpha zebra]", ids)
	}
}

func TestHealth(t *testing.T) {
	p1 := &fakeProvider{name: "p1"}
	h := newTestServer(t, map[string]chat.Provider{"p1": p1}, []config.ModelConfig{
		{Name: "virtual-a", Strategy: "failover", Candidates: []config.ModelCandidate{{Provider: "p1", Model: "m1"}}},
	})

	rec := doJSON(t, h, http.MethodGet, "/health", nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
}

// newToolTestServer builds a Server wired with the given providers, models and
// a tool registry.
func newToolTestServer(t *testing.T, providers map[string]chat.Provider, models []config.ModelConfig, tr *toolkit.Registry) http.Handler {
	t.Helper()
	reg, err := routing.NewRegistry(models)
	if err != nil {
		t.Fatalf("NewRegistry() error = %v", err)
	}
	return NewServer(Deps{
		Providers: providers,
		Registry:  reg,
		Logger:    slog.New(slog.NewTextHandler(io.Discard, nil)),
		Tools:     tr,
	})
}

func TestToolsListsRegisteredTools(t *testing.T) {
	tr := toolkit.New()
	if err := toolkit.RegisterBuiltin(tr); err != nil {
		t.Fatalf("RegisterBuiltin() error = %v", err)
	}
	h := newToolTestServer(t, map[string]chat.Provider{"p1": &fakeProvider{name: "p1"}}, []config.ModelConfig{
		{Name: "virtual-a", Strategy: "failover", Candidates: []config.ModelCandidate{{Provider: "p1", Model: "m1"}}},
	}, tr)

	rec := doJSON(t, h, http.MethodGet, "/v1/tools", nil)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	var out toolListResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode tools: %v", err)
	}
	if out.Object != "list" {
		t.Errorf("object = %q, want list", out.Object)
	}
	if len(out.Data) != 2 {
		t.Fatalf("data len = %d, want 2", len(out.Data))
	}
	names := []string{out.Data[0].Function.Name, out.Data[1].Function.Name}
	if names[0] != "echo_text" || names[1] != "get_utc_time" {
		t.Errorf("tool names = %v, want [echo_text get_utc_time]", names)
	}
	for _, d := range out.Data {
		if d.Type != "function" {
			t.Errorf("type = %q, want function", d.Type)
		}
		if d.Function.Parameters == nil {
			t.Errorf("tool %q missing parameters", d.Function.Name)
		}
	}
}

func TestToolCallSuccess(t *testing.T) {
	tr := toolkit.New()
	if err := toolkit.RegisterBuiltin(tr); err != nil {
		t.Fatalf("RegisterBuiltin() error = %v", err)
	}
	h := newToolTestServer(t, map[string]chat.Provider{"p1": &fakeProvider{name: "p1"}}, []config.ModelConfig{
		{Name: "virtual-a", Strategy: "failover", Candidates: []config.ModelCandidate{{Provider: "p1", Model: "m1"}}},
	}, tr)

	rec := doJSON(t, h, http.MethodPost, "/v1/tools/call", map[string]any{
		"name":      "echo_text",
		"arguments": map[string]any{"text": "hello"},
	})

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	var out toolCallResult
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode result: %v", err)
	}
	if out.IsError {
		t.Errorf("isError = true, want false")
	}
	if len(out.Content) != 1 || out.Content[0].Type != "text" || out.Content[0].Text != "hello" {
		t.Errorf("content = %+v, want [{text hello}]", out.Content)
	}
}

func TestToolCallUnknownToolReturns404(t *testing.T) {
	tr := toolkit.New()
	if err := toolkit.RegisterBuiltin(tr); err != nil {
		t.Fatalf("RegisterBuiltin() error = %v", err)
	}
	h := newToolTestServer(t, map[string]chat.Provider{"p1": &fakeProvider{name: "p1"}}, []config.ModelConfig{
		{Name: "virtual-a", Strategy: "failover", Candidates: []config.ModelCandidate{{Provider: "p1", Model: "m1"}}},
	}, tr)

	rec := doJSON(t, h, http.MethodPost, "/v1/tools/call", map[string]any{
		"name":      "nope",
		"arguments": map[string]any{},
	})

	if rec.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404; body=%s", rec.Code, rec.Body.String())
	}
	var errResp errorResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &errResp); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if errResp.Error.Code != "tool_not_found" {
		t.Errorf("error code = %q, want tool_not_found", errResp.Error.Code)
	}
	if errResp.Error.Message != "tool not found: nope" {
		t.Errorf("error message = %q, want 'tool not found: nope'", errResp.Error.Message)
	}
}

func TestReasoningEffortPassthrough(t *testing.T) {
	var got chat.ChatRequest
	p1 := &fakeProvider{
		name: "p1",
		complete: func(_ context.Context, req chat.ChatRequest) (chat.ChatResponse, error) {
			got = req
			return chat.ChatResponse{ID: "resp-1", Model: "m1", Content: "ok", FinishReason: "stop"}, nil
		},
	}
	h := newTestServer(t, map[string]chat.Provider{"p1": p1}, []config.ModelConfig{
		{Name: "virtual-a", Strategy: "failover", Candidates: []config.ModelCandidate{{Provider: "p1", Model: "m1"}}},
	})

	rec := doJSON(t, h, http.MethodPost, "/v1/chat/completions", map[string]any{
		"model":            "virtual-a",
		"messages":         []map[string]string{{"role": "user", "content": "hi"}},
		"reasoning_effort": "high",
	})
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	if got.ReasoningEffort == nil || *got.ReasoningEffort != "high" {
		t.Errorf("ReasoningEffort = %v, want non-nil \"high\"", got.ReasoningEffort)
	}

	rec = doJSON(t, h, http.MethodPost, "/v1/chat/completions", map[string]any{
		"model":    "virtual-a",
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	})
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	if got.ReasoningEffort != nil {
		t.Errorf("ReasoningEffort = %v, want nil when omitted", got.ReasoningEffort)
	}
}

func TestReasoningEffort_InvalidValueReturns400(t *testing.T) {
	p1 := &fakeProvider{name: "p1"}
	h := newTestServer(t, map[string]chat.Provider{"p1": p1}, []config.ModelConfig{
		{Name: "virtual-a", Strategy: "failover", Candidates: []config.ModelCandidate{{Provider: "p1", Model: "m1"}}},
	})

	rec := doJSON(t, h, http.MethodPost, "/v1/chat/completions", map[string]any{
		"model":            "virtual-a",
		"messages":         []map[string]string{{"role": "user", "content": "hi"}},
		"reasoning_effort": "ultra",
	})

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400; body=%s", rec.Code, rec.Body.String())
	}
	var errResp errorResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &errResp); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if errResp.Error.Code != "invalid_reasoning_effort" {
		t.Errorf("error code = %q, want invalid_reasoning_effort", errResp.Error.Code)
	}
	if errResp.Error.Type != "invalid_request_error" {
		t.Errorf("error type = %q, want invalid_request_error", errResp.Error.Type)
	}
}

func TestReasoningEffort_ValidValuesAccepted(t *testing.T) {
	p1 := &fakeProvider{
		name: "p1",
		complete: func(context.Context, chat.ChatRequest) (chat.ChatResponse, error) {
			return chat.ChatResponse{ID: "resp-1", Model: "m1", Content: "ok", FinishReason: "stop"}, nil
		},
	}
	h := newTestServer(t, map[string]chat.Provider{"p1": p1}, []config.ModelConfig{
		{Name: "virtual-a", Strategy: "failover", Candidates: []config.ModelCandidate{{Provider: "p1", Model: "m1"}}},
	})

	for _, v := range []string{"none", "minimal", "low", "medium", "high", "xhigh", "max"} {
		rec := doJSON(t, h, http.MethodPost, "/v1/chat/completions", map[string]any{
			"model":            "virtual-a",
			"messages":         []map[string]string{{"role": "user", "content": "hi"}},
			"reasoning_effort": v,
		})
		if rec.Code != http.StatusOK {
			t.Errorf("reasoning_effort %q: status = %d, want 200; body=%s", v, rec.Code, rec.Body.String())
		}
	}
}

func TestReasoningEffort_CaseSensitive(t *testing.T) {
	p1 := &fakeProvider{name: "p1"}
	h := newTestServer(t, map[string]chat.Provider{"p1": p1}, []config.ModelConfig{
		{Name: "virtual-a", Strategy: "failover", Candidates: []config.ModelCandidate{{Provider: "p1", Model: "m1"}}},
	})

	rec := doJSON(t, h, http.MethodPost, "/v1/chat/completions", map[string]any{
		"model":            "virtual-a",
		"messages":         []map[string]string{{"role": "user", "content": "hi"}},
		"reasoning_effort": "HIGH",
	})

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400; body=%s", rec.Code, rec.Body.String())
	}
	var errResp errorResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &errResp); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if errResp.Error.Code != "invalid_reasoning_effort" {
		t.Errorf("error code = %q, want invalid_reasoning_effort", errResp.Error.Code)
	}
}

// newKeyedTestServer builds a Server wired with the given providers and an
// optional client key ("" disables auth).
func newKeyedTestServer(t *testing.T, providers map[string]chat.Provider, clientKey string) http.Handler {
	t.Helper()
	reg, err := routing.NewRegistry([]config.ModelConfig{
		{Name: "virtual-a", Strategy: "failover", Candidates: []config.ModelCandidate{{Provider: "p1", Model: "m1"}}},
	})
	if err != nil {
		t.Fatalf("NewRegistry() error = %v", err)
	}
	return NewServer(Deps{
		Providers: providers,
		Registry:  reg,
		Logger:    slog.New(slog.NewTextHandler(io.Discard, nil)),
		ClientKey: clientKey,
	})
}

func TestAuth_MissingHeaderReturns401(t *testing.T) {
	h := newKeyedTestServer(t, map[string]chat.Provider{"p1": &fakeProvider{name: "p1"}}, "secret")

	rec := doJSON(t, h, http.MethodPost, "/v1/chat/completions", map[string]any{
		"model":    "virtual-a",
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	})

	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want 401", rec.Code)
	}
	var errResp errorResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &errResp); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	if errResp.Error.Code != "invalid_api_key" || errResp.Error.Type != "authentication_error" {
		t.Errorf("error = %+v, want code invalid_api_key type authentication_error", errResp.Error)
	}
}

func TestAuth_WrongKeyReturns401(t *testing.T) {
	h := newKeyedTestServer(t, map[string]chat.Provider{"p1": &fakeProvider{name: "p1"}}, "secret")

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"virtual-a","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer wrong")
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want 401", rec.Code)
	}
}

func TestAuth_CorrectKeySucceeds(t *testing.T) {
	p1 := &fakeProvider{
		name: "p1",
		complete: func(context.Context, chat.ChatRequest) (chat.ChatResponse, error) {
			return chat.ChatResponse{ID: "resp-1", Model: "m1", Content: "ok", FinishReason: "stop"}, nil
		},
	}
	h := newKeyedTestServer(t, map[string]chat.Provider{"p1": p1}, "secret")

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"virtual-a","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "bearer secret") // scheme matched case-insensitively
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
}

func TestAuth_DisabledAllowsNoHeader(t *testing.T) {
	p1 := &fakeProvider{
		name: "p1",
		complete: func(context.Context, chat.ChatRequest) (chat.ChatResponse, error) {
			return chat.ChatResponse{ID: "resp-1", Model: "m1", Content: "ok", FinishReason: "stop"}, nil
		},
	}
	h := newKeyedTestServer(t, map[string]chat.Provider{"p1": p1}, "")

	rec := doJSON(t, h, http.MethodPost, "/v1/chat/completions", map[string]any{
		"model":    "virtual-a",
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
	})

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
}

func TestAuth_HealthExemptFromKey(t *testing.T) {
	h := newKeyedTestServer(t, map[string]chat.Provider{"p1": &fakeProvider{name: "p1"}}, "secret")

	rec := doJSON(t, h, http.MethodGet, "/health", nil)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}
}
