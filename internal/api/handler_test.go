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
	"path/filepath"
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
	// scripted, when non-empty, is popped one response per Complete call and
	// takes precedence over complete.
	scripted []chat.ChatResponse
	// requests records every Complete and Stream call's request for assertions.
	requests []chat.ChatRequest
}

func (f *fakeProvider) Name() string { return f.name }

func (f *fakeProvider) Complete(ctx context.Context, req chat.ChatRequest) (chat.ChatResponse, error) {
	f.requests = append(f.requests, req)
	if len(f.scripted) > 0 {
		resp := f.scripted[0]
		f.scripted = f.scripted[1:]
		return resp, nil
	}
	if f.complete != nil {
		return f.complete(ctx, req)
	}
	return chat.ChatResponse{}, errors.New("fakeProvider: Complete not implemented")
}

func (f *fakeProvider) Stream(ctx context.Context, req chat.ChatRequest, emit chat.StreamFunc) error {
	f.requests = append(f.requests, req)
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

func TestChatCompletions_ContentArray(t *testing.T) {
	p1 := &fakeProvider{
		name: "p1",
		complete: func(_ context.Context, req chat.ChatRequest) (chat.ChatResponse, error) {
			if len(req.Messages) != 1 {
				t.Errorf("Messages len = %d, want 1", len(req.Messages))
			} else if got := req.Messages[0].Content; got != "hello" {
				t.Errorf("Content = %q, want hello", got)
			}
			return chat.ChatResponse{Content: "world", FinishReason: "stop"}, nil
		},
	}
	h := newTestServer(t, map[string]chat.Provider{"p1": p1}, []config.ModelConfig{
		{Name: "virtual-a", Strategy: "failover", Candidates: []config.ModelCandidate{{Provider: "p1", Model: "m1"}}},
	})

	rec := doJSON(t, h, http.MethodPost, "/v1/chat/completions", map[string]any{
		"model": "virtual-a",
		"messages": []map[string]any{
			{"role": "user", "content": []map[string]string{{"type": "text", "text": "hello"}}},
		},
	})

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
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

// TestRateLimit429_FailoverWithCooldown verifies that a 429 puts the candidate
// into cooldown (skipped on the next request) and fails over to the next
// candidate.
func TestRateLimit429_FailoverWithCooldown(t *testing.T) {
	var p1Calls int
	p1 := &fakeProvider{
		name: "p1",
		complete: func(context.Context, chat.ChatRequest) (chat.ChatResponse, error) {
			p1Calls++
			return chat.ChatResponse{}, &chat.RateLimitError{Provider: "p1", StatusCode: http.StatusTooManyRequests, Message: "rate limited"}
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

	body := map[string]any{"model": "virtual-a", "messages": []map[string]string{{"role": "user", "content": "hi"}}}

	rec := doJSON(t, h, http.MethodPost, "/v1/chat/completions", body)
	if rec.Code != http.StatusOK {
		t.Fatalf("first request status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	if p1Calls != 1 {
		t.Errorf("p1 calls on first request = %d, want 1", p1Calls)
	}

	// Second request: p1 is in cooldown (30s default), so p2 must be hit
	// directly without touching p1.
	p1Calls = 0
	rec2 := doJSON(t, h, http.MethodPost, "/v1/chat/completions", body)
	if rec2.Code != http.StatusOK {
		t.Fatalf("second request status = %d, want 200; body=%s", rec2.Code, rec2.Body.String())
	}
	if p1Calls != 0 {
		t.Errorf("p1 calls on second request = %d, want 0 (cooldown)", p1Calls)
	}
}

// TestNonRateLimit_RetriesSameCandidate3x verifies that a non-429 failure is
// retried on the same candidate up to maxAttempts before failing over.
func TestNonRateLimit_RetriesSameCandidate3x(t *testing.T) {
	var p1Calls int
	p1 := &fakeProvider{
		name: "p1",
		complete: func(context.Context, chat.ChatRequest) (chat.ChatResponse, error) {
			p1Calls++
			if p1Calls < 3 {
				return chat.ChatResponse{}, errors.New("p1 transient down")
			}
			return chat.ChatResponse{Content: "from p1 after retries", FinishReason: "stop"}, nil
		},
	}
	p2 := &fakeProvider{
		name: "p2",
		complete: func(context.Context, chat.ChatRequest) (chat.ChatResponse, error) {
			return chat.ChatResponse{Content: "should not be reached", FinishReason: "stop"}, nil
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
	if p1Calls != 3 {
		t.Errorf("p1 calls = %d, want 3 (initial + 2 retries)", p1Calls)
	}
	var out chatCompletionResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if out.Model != "m1" || out.Choices[0].Message.Content != "from p1 after retries" {
		t.Errorf("response = %q from %q, want p1's retried response", out.Choices[0].Message.Content, out.Model)
	}
}

// TestNonRateLimit_FailsAllRetriesThenFailoverNoCooldown verifies that a
// persistent non-429 failure fails over after maxAttempts, and the candidate
// is NOT cooldowned (tried again on the next request).
func TestNonRateLimit_FailsAllRetriesThenFailoverNoCooldown(t *testing.T) {
	var p1Calls int
	p1 := &fakeProvider{
		name: "p1",
		complete: func(context.Context, chat.ChatRequest) (chat.ChatResponse, error) {
			p1Calls++
			return chat.ChatResponse{}, errors.New("p1 down")
		},
	}
	p2 := &fakeProvider{
		name: "p2",
		complete: func(context.Context, chat.ChatRequest) (chat.ChatResponse, error) {
			return chat.ChatResponse{Content: "from p2", FinishReason: "stop"}, nil
		},
	}
	h := newTestServer(t, map[string]chat.Provider{"p1": p1, "p2": p2}, []config.ModelConfig{
		{Name: "virtual-a", Strategy: "failover", Candidates: []config.ModelCandidate{
			{Provider: "p1", Model: "m1"},
			{Provider: "p2", Model: "m2"},
		}},
	})

	body := map[string]any{"model": "virtual-a", "messages": []map[string]string{{"role": "user", "content": "hi"}}}

	rec := doJSON(t, h, http.MethodPost, "/v1/chat/completions", body)
	if rec.Code != http.StatusOK {
		t.Fatalf("first request status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	if p1Calls != maxAttempts {
		t.Errorf("p1 calls on first request = %d, want %d (maxAttempts)", p1Calls, maxAttempts)
	}

	// No cooldown for non-429: the next request tries p1 again.
	rec2 := doJSON(t, h, http.MethodPost, "/v1/chat/completions", body)
	if rec2.Code != http.StatusOK {
		t.Fatalf("second request status = %d, want 200; body=%s", rec2.Code, rec2.Body.String())
	}
	if p1Calls != maxAttempts*2 {
		t.Errorf("p1 calls after second request = %d, want %d (tried again, no cooldown)", p1Calls, maxAttempts*2)
	}
}

// TestPermanent4xx_NoRetryFailsOverImmediately verifies that a 4xx error (e.g.
// 400 invalid request) is NOT retried — it fails over to the next candidate on
// the first try, so the candidate is hit exactly once.
func TestPermanent4xx_NoRetryFailsOverImmediately(t *testing.T) {
	var p1Calls int
	p1 := &fakeProvider{
		name: "p1",
		complete: func(context.Context, chat.ChatRequest) (chat.ChatResponse, error) {
			p1Calls++
			return chat.ChatResponse{}, &chat.ProviderError{Provider: "p1", StatusCode: http.StatusBadRequest, Message: "invalid request"}
		},
	}
	p2 := &fakeProvider{
		name: "p2",
		complete: func(context.Context, chat.ChatRequest) (chat.ChatResponse, error) {
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
	if p1Calls != 1 {
		t.Errorf("p1 calls = %d, want 1 (4xx must not be retried)", p1Calls)
	}
	var out chatCompletionResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if out.Model != "m2" || out.Choices[0].Message.Content != "from p2" {
		t.Errorf("response = %q from %q, want p2", out.Choices[0].Message.Content, out.Model)
	}
}

// TestTransient5xx_Retries3x verifies a 5xx status is treated as transient and
// retried up to maxAttempts on the same candidate before failing over.
func TestTransient5xx_Retries3x(t *testing.T) {
	var p1Calls int
	p1 := &fakeProvider{
		name: "p1",
		complete: func(context.Context, chat.ChatRequest) (chat.ChatResponse, error) {
			p1Calls++
			return chat.ChatResponse{}, &chat.ProviderError{Provider: "p1", StatusCode: http.StatusInternalServerError, Message: "boom"}
		},
	}
	p2 := &fakeProvider{
		name: "p2",
		complete: func(context.Context, chat.ChatRequest) (chat.ChatResponse, error) {
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
	if p1Calls != maxAttempts {
		t.Errorf("p1 calls = %d, want %d (5xx retried 3x)", p1Calls, maxAttempts)
	}
}

// TestReasoningCache_IsolatedPerSession verifies reasoning cached under one
// session key is not visible to another session (no cross-session leak).
func TestReasoningCache_IsolatedPerSession(t *testing.T) {
	reasoningByCallID.mu.Lock()
	delete(reasoningByCallID.m, "sess-A")
	delete(reasoningByCallID.m, "sess-B")
	reasoningByCallID.mu.Unlock()
	t.Cleanup(func() {
		reasoningByCallID.mu.Lock()
		delete(reasoningByCallID.m, "sess-A")
		delete(reasoningByCallID.m, "sess-B")
		reasoningByCallID.mu.Unlock()
	})

	reasoningByCallID.remember("sess-A", []chat.ToolCall{{ID: "call_1"}}, "thinking for A")
	if got := reasoningByCallID.lookup("sess-B", "call_1"); got != "" {
		t.Errorf("lookup in sess-B = %q, want empty (session isolation)", got)
	}
	if got := reasoningByCallID.lookup("sess-A", "call_1"); got != "thinking for A" {
		t.Errorf("lookup in sess-A = %q, want thinking for A", got)
	}
}

// TestReasoningCache_PersistsToDisk verifies a disk-backed cache round-trips
// through a file: an entry stored in one instance is found by a fresh instance
// constructed from the same path (simulating a router restart).
func TestReasoningCache_PersistsToDisk(t *testing.T) {
	path := filepath.Join(t.TempDir(), "reasoning_cache.json")

	c1 := newReasoningCache(path)
	c1.remember("sess-X", []chat.ToolCall{{ID: "call_9"}}, "persisted thinking")

	c2 := newReasoningCache(path)
	if got := c2.lookup("sess-X", "call_9"); got != "persisted thinking" {
		t.Errorf("fresh instance lookup = %q, want persisted thinking", got)
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

func TestChatCompletions_ToolRoundTrip(t *testing.T) {
	var got chat.ChatRequest
	p1 := &fakeProvider{
		name: "p1",
		complete: func(_ context.Context, req chat.ChatRequest) (chat.ChatResponse, error) {
			got = req
			return chat.ChatResponse{Content: "ok", FinishReason: "stop"}, nil
		},
	}
	h := newTestServer(t, map[string]chat.Provider{"p1": p1}, []config.ModelConfig{
		{Name: "virtual-a", Strategy: "failover", Candidates: []config.ModelCandidate{{Provider: "p1", Model: "m1"}}},
	})

	rec := doJSON(t, h, http.MethodPost, "/v1/chat/completions", map[string]any{
		"model": "virtual-a",
		"messages": []map[string]any{
			{"role": "user", "content": "what's the weather"},
			{"role": "assistant", "content": "", "tool_calls": []map[string]any{
				{"id": "call_1", "type": "function", "function": map[string]string{"name": "search_web", "arguments": `{"query":"weather"}`}},
			}},
			{"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
		},
	})

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	if len(got.Messages) != 3 {
		t.Fatalf("Messages len = %d, want 3", len(got.Messages))
	}
	if got.Messages[0].Content != "what's the weather" {
		t.Errorf("Messages[0].Content = %q, want what's the weather", got.Messages[0].Content)
	}
	tcs := got.Messages[1].ToolCalls
	if len(tcs) != 1 {
		t.Fatalf("Messages[1].ToolCalls len = %d, want 1", len(tcs))
	}
	if tcs[0].ID != "call_1" {
		t.Errorf("ToolCalls[0].ID = %q, want call_1", tcs[0].ID)
	}
	if tcs[0].Name != "search_web" {
		t.Errorf("ToolCalls[0].Name = %q, want search_web", tcs[0].Name)
	}
	if !strings.Contains(string(tcs[0].Arguments), "query") {
		t.Errorf("ToolCalls[0].Arguments = %s, want it to contain query", tcs[0].Arguments)
	}
	if got.Messages[2].ToolCallID != "call_1" {
		t.Errorf("Messages[2].ToolCallID = %q, want call_1", got.Messages[2].ToolCallID)
	}
	if got.Messages[2].Content != "sunny" {
		t.Errorf("Messages[2].Content = %q, want sunny", got.Messages[2].Content)
	}
}

func TestChatCompletions_ReasoningContentRoundTrip(t *testing.T) {
	var got chat.ChatRequest
	p1 := &fakeProvider{
		name: "p1",
		complete: func(_ context.Context, req chat.ChatRequest) (chat.ChatResponse, error) {
			got = req
			return chat.ChatResponse{Content: "ok", FinishReason: "stop"}, nil
		},
	}
	h := newTestServer(t, map[string]chat.Provider{"p1": p1}, []config.ModelConfig{
		{Name: "virtual-a", Strategy: "failover", Candidates: []config.ModelCandidate{{Provider: "p1", Model: "m1"}}},
	})

	rec := doJSON(t, h, http.MethodPost, "/v1/chat/completions", map[string]any{
		"model": "virtual-a",
		"messages": []map[string]any{
			{"role": "user", "content": "what's the weather"},
			{"role": "assistant", "content": "", "reasoning_content": "thinking...", "tool_calls": []map[string]any{
				{"id": "call_1", "type": "function", "function": map[string]string{"name": "search_web", "arguments": `{"query":"weather"}`}},
			}},
			{"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
		},
	})

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	if len(got.Messages) != 3 {
		t.Fatalf("Messages len = %d, want 3", len(got.Messages))
	}
	if got.Messages[1].ReasoningContent != "thinking..." {
		t.Errorf("Messages[1].ReasoningContent = %q, want thinking...", got.Messages[1].ReasoningContent)
	}
	if got.Messages[1].Content != "" {
		t.Errorf("Messages[1].Content = %q, want empty", got.Messages[1].Content)
	}
	if len(got.Messages[1].ToolCalls) != 1 {
		t.Errorf("Messages[1].ToolCalls len = %d, want 1", len(got.Messages[1].ToolCalls))
	}
	if got.Messages[0].ReasoningContent != "" {
		t.Errorf("Messages[0].ReasoningContent = %q, want empty", got.Messages[0].ReasoningContent)
	}
}

func TestChatCompletions_ClientToolsPassthrough(t *testing.T) {
	var got chat.ChatRequest
	p1 := &fakeProvider{
		name: "p1",
		complete: func(_ context.Context, req chat.ChatRequest) (chat.ChatResponse, error) {
			got = req
			return chat.ChatResponse{Content: "ok", FinishReason: "stop"}, nil
		},
	}
	// deps.Tools is nil: no aggregated tools configured.
	h := newTestServer(t, map[string]chat.Provider{"p1": p1}, []config.ModelConfig{
		{Name: "virtual-a", Strategy: "failover", Candidates: []config.ModelCandidate{{Provider: "p1", Model: "m1"}}},
	})

	rec := doJSON(t, h, http.MethodPost, "/v1/chat/completions", map[string]any{
		"model":    "virtual-a",
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
		"tools": []map[string]any{
			{"type": "function", "function": map[string]any{"name": "my_tool", "description": "d", "parameters": map[string]any{"type": "object"}}},
		},
	})

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	if len(got.Tools) != 1 {
		t.Fatalf("Tools len = %d, want 1", len(got.Tools))
	}
	if got.Tools[0].Name != "my_tool" {
		t.Errorf("Tools[0].Name = %q, want my_tool", got.Tools[0].Name)
	}
	if got.Tools[0].Description != "d" {
		t.Errorf("Tools[0].Description = %q, want d", got.Tools[0].Description)
	}
	if got.Tools[0].InputSchema == nil {
		t.Errorf("Tools[0].InputSchema = nil, want map")
	}
}

func TestChatCompletions_ClientAndServerToolsMerged(t *testing.T) {
	var got chat.ChatRequest
	p1 := &fakeProvider{
		name: "p1",
		complete: func(_ context.Context, req chat.ChatRequest) (chat.ChatResponse, error) {
			got = req
			return chat.ChatResponse{Content: "ok", FinishReason: "stop"}, nil
		},
	}
	tr := toolkit.New()
	if err := toolkit.RegisterBuiltin(tr); err != nil {
		t.Fatalf("RegisterBuiltin() error = %v", err)
	}
	h := newToolTestServer(t, map[string]chat.Provider{"p1": p1}, []config.ModelConfig{
		{Name: "virtual-a", Strategy: "failover", Candidates: []config.ModelCandidate{{Provider: "p1", Model: "m1"}}},
	}, tr)

	rec := doJSON(t, h, http.MethodPost, "/v1/chat/completions", map[string]any{
		"model":    "virtual-a",
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
		"tools": []map[string]any{
			{"type": "function", "function": map[string]any{"name": "my_tool", "description": "d", "parameters": map[string]any{"type": "object"}}},
		},
	})

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	// Client tools no longer replace the aggregated set: they are merged, so
	// the server's tools are always offered alongside the client's.
	if len(got.Tools) != 3 {
		t.Fatalf("Tools len = %d, want 3 (my_tool + echo_text + get_utc_time)", len(got.Tools))
	}
	names := map[string]bool{}
	for _, tl := range got.Tools {
		names[tl.Name] = true
	}
	for _, want := range []string{"my_tool", "echo_text", "get_utc_time"} {
		if !names[want] {
			t.Errorf("merged tools missing %q: %+v", want, got.Tools)
		}
	}
}

func TestChatCompletions_ServerToolExecuted(t *testing.T) {
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

	rec := doJSON(t, h, http.MethodPost, "/v1/chat/completions", map[string]any{
		"model":    "virtual-a",
		"messages": []map[string]string{{"role": "user", "content": "cuaca hari ini?"}},
	})

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	var out chatCompletionResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if out.Choices[0].FinishReason != "stop" {
		t.Errorf("finish_reason = %q, want stop", out.Choices[0].FinishReason)
	}
	if !strings.Contains(out.Choices[0].Message.Content, "32C") {
		t.Errorf("content = %q, want it to contain 32C", out.Choices[0].Message.Content)
	}
	if len(out.Choices[0].Message.ToolCalls) != 0 {
		t.Errorf("client saw tool_calls = %+v, want none (router executed them)", out.Choices[0].Message.ToolCalls)
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

func TestChatCompletions_ClientToolPassthrough(t *testing.T) {
	tr := toolkit.New()
	if err := tr.RegisterLocal(chat.Tool{Name: "search_web", Description: "search", InputSchema: map[string]any{"type": "object"}}, func(context.Context, json.RawMessage) (string, error) {
		return "unused", nil
	}); err != nil {
		t.Fatalf("RegisterLocal() error = %v", err)
	}
	p1 := &fakeProvider{
		name: "p1",
		complete: func(context.Context, chat.ChatRequest) (chat.ChatResponse, error) {
			return chat.ChatResponse{
				FinishReason: "tool_calls",
				ToolCalls: []chat.ToolCall{{
					ID:        "call_1",
					Name:      "my_tool",
					Arguments: json.RawMessage(`{"x":1}`),
				}},
			}, nil
		},
	}
	h := newToolTestServer(t, map[string]chat.Provider{"p1": p1}, []config.ModelConfig{
		{Name: "virtual-a", Strategy: "failover", Candidates: []config.ModelCandidate{{Provider: "p1", Model: "m1"}}},
	}, tr)

	rec := doJSON(t, h, http.MethodPost, "/v1/chat/completions", map[string]any{
		"model":    "virtual-a",
		"messages": []map[string]string{{"role": "user", "content": "hi"}},
		"tools": []map[string]any{
			{"type": "function", "function": map[string]any{"name": "my_tool", "description": "d", "parameters": map[string]any{"type": "object"}}},
		},
	})

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	var out chatCompletionResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	tc := out.Choices[0].Message.ToolCalls
	if len(tc) != 1 || tc[0].ID != "call_1" || tc[0].Function.Name != "my_tool" {
		t.Fatalf("tool_calls = %+v, want client tool call_1/my_tool passed through", tc)
	}
	if len(p1.requests) != 1 {
		t.Errorf("provider called %d times, want 1 (no internal loop)", len(p1.requests))
	}
	if len(p1.requests) == 1 && len(p1.requests[0].Tools) != 2 {
		t.Errorf("merged tool count = %d, want 2 (client my_tool + server search_web)", len(p1.requests[0].Tools))
	}
}

func TestChatCompletions_StreamServerTool(t *testing.T) {
	tr := toolkit.New()
	if err := tr.RegisterLocal(chat.Tool{Name: "search_web", Description: "search", InputSchema: map[string]any{"type": "object"}}, func(context.Context, json.RawMessage) (string, error) {
		return "cerah 32C", nil
	}); err != nil {
		t.Fatalf("RegisterLocal() error = %v", err)
	}
	streamCalls := 0
	p1 := &fakeProvider{
		name: "p1",
		stream: func(_ context.Context, _ chat.ChatRequest, emit chat.StreamFunc) error {
			streamCalls++
			return emit(chat.StreamDelta{
				FinishReason: "tool_calls",
				ToolCalls: []chat.ToolCall{{
					ID:        "call_1",
					Name:      "search_web",
					Arguments: json.RawMessage(`{"query":"cuaca"}`),
				}},
			})
		},
		complete: func(context.Context, chat.ChatRequest) (chat.ChatResponse, error) {
			return chat.ChatResponse{Content: "suhu cerah 32C", FinishReason: "stop"}, nil
		},
	}
	h := newToolTestServer(t, map[string]chat.Provider{"p1": p1}, []config.ModelConfig{
		{Name: "virtual-a", Strategy: "failover", Candidates: []config.ModelCandidate{{Provider: "p1", Model: "m1"}}},
	}, tr)

	rec := doJSON(t, h, http.MethodPost, "/v1/chat/completions", map[string]any{
		"model":    "virtual-a",
		"stream":   true,
		"messages": []map[string]string{{"role": "user", "content": "cuaca?"}},
	})

	body := rec.Body.String()
	if !strings.Contains(body, "data: [DONE]") {
		t.Errorf("body missing [DONE]: %q", body)
	}
	if !strings.Contains(body, "cerah 32C") {
		t.Errorf("body missing final content: %q", body)
	}
	if strings.Contains(body, "search_web") || strings.Contains(body, "tool_calls") {
		t.Errorf("body leaks server tool call: %q", body)
	}
	if streamCalls != 1 {
		t.Errorf("Stream called %d times, want 1", streamCalls)
	}
	if len(p1.requests) != 2 {
		t.Fatalf("provider recorded %d requests, want 2 (stream turn + tool-loop continuation)", len(p1.requests))
	}
	if !p1.requests[0].Stream {
		t.Errorf("first request Stream = %v, want true (client streamed)", p1.requests[0].Stream)
	}
	if p1.requests[1].Stream {
		t.Errorf("tool-loop continuation request Stream = %v, want false (must complete, not stream)", p1.requests[1].Stream)
	}
}

// seedReasoning stores a cached reasoning value so a test can simulate a prior
// upstream turn that produced reasoning_content for a tool call. Tests use the
// global bucket (no X-Session-ID header) via doJSON.
func seedReasoning(id, text string) {
	reasoningByCallID.remember("", []chat.ToolCall{{ID: id}}, text)
}

func TestChatCompletions_ReasoningContentEchoed(t *testing.T) {
	// Simulate a prior upstream turn: search_web call_1 produced reasoning.
	seedReasoning("call_1", "prior thinking")
	t.Cleanup(func() {
		reasoningByCallID.mu.Lock()
		delete(reasoningByCallID.m[""], "call_1")
		reasoningByCallID.mu.Unlock()
	})

	var got chat.ChatRequest
	p1 := &fakeProvider{
		name: "p1",
		complete: func(_ context.Context, req chat.ChatRequest) (chat.ChatResponse, error) {
			got = req
			return chat.ChatResponse{Content: "ok", FinishReason: "stop"}, nil
		},
	}
	h := newTestServer(t, map[string]chat.Provider{"p1": p1}, []config.ModelConfig{
		{Name: "virtual-a", Strategy: "failover", Candidates: []config.ModelCandidate{{Provider: "p1", Model: "m1"}}},
	})

	// The client echoes the tool call back WITHOUT reasoning_content, as
	// OpenCode/OpenAI SDKs always do.
	rec := doJSON(t, h, http.MethodPost, "/v1/chat/completions", map[string]any{
		"model": "virtual-a",
		"messages": []map[string]any{
			{"role": "user", "content": "what's the weather"},
			{"role": "assistant", "content": "", "tool_calls": []map[string]any{
				{"id": "call_1", "type": "function", "function": map[string]string{"name": "search_web", "arguments": `{"query":"weather"}`}},
			}},
			{"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
		},
	})

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	if len(got.Messages) != 3 {
		t.Fatalf("Messages len = %d, want 3", len(got.Messages))
	}
	if got.Messages[1].Role != chat.RoleAssistant {
		t.Fatalf("Messages[1].Role = %q, want assistant", got.Messages[1].Role)
	}
	if got.Messages[1].ReasoningContent != "prior thinking" {
		t.Errorf("Messages[1].ReasoningContent = %q, want prior thinking (echoed from cache)", got.Messages[1].ReasoningContent)
	}
	if got.Messages[0].ReasoningContent != "" {
		t.Errorf("Messages[0].ReasoningContent = %q, want empty", got.Messages[0].ReasoningContent)
	}
}

// TestChatCompletions_ReasoningEchoMissInjectsPlaceholder verifies that a
// tool-call turn with no cached reasoning (e.g. the calls were produced by a
// non-thinking provider like GLM without thinking enabled) still reaches the
// upstream with a non-empty reasoning_content, so thinking-mode candidates in
// a failover chain accept the turn instead of 400ing.
func TestChatCompletions_ReasoningEchoMissInjectsPlaceholder(t *testing.T) {
	var got chat.ChatRequest
	p1 := &fakeProvider{
		name: "p1",
		complete: func(_ context.Context, req chat.ChatRequest) (chat.ChatResponse, error) {
			got = req
			return chat.ChatResponse{Content: "ok", FinishReason: "stop"}, nil
		},
	}
	h := newTestServer(t, map[string]chat.Provider{"p1": p1}, []config.ModelConfig{
		{Name: "virtual-a", Strategy: "failover", Candidates: []config.ModelCandidate{{Provider: "p1", Model: "m1"}}},
	})

	rec := doJSON(t, h, http.MethodPost, "/v1/chat/completions", map[string]any{
		"model": "virtual-a",
		"messages": []map[string]any{
			{"role": "user", "content": "what's the weather"},
			{"role": "assistant", "content": "", "tool_calls": []map[string]any{
				{"id": "call_unknown", "type": "function", "function": map[string]string{"name": "search_web", "arguments": `{"query":"weather"}`}},
			}},
			{"role": "tool", "tool_call_id": "call_unknown", "content": "sunny"},
		},
	})

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	if got.Messages[1].ReasoningContent != reasoningPlaceholder {
		t.Errorf("Messages[1].ReasoningContent = %q, want placeholder %q", got.Messages[1].ReasoningContent, reasoningPlaceholder)
	}
	// The placeholder must never be cached as real reasoning.
	if got := reasoningByCallID.lookup("", "call_unknown"); got != "" {
		t.Errorf("placeholder leaked into cache: lookup = %q, want empty", got)
	}
}

func TestChatCompletions_ServerToolEchoesReasoning(t *testing.T) {
	tr := toolkit.New()
	if err := tr.RegisterLocal(chat.Tool{Name: "search_web", Description: "search", InputSchema: map[string]any{"type": "object"}}, func(context.Context, json.RawMessage) (string, error) {
		return "cerah 32C", nil
	}); err != nil {
		t.Fatalf("RegisterLocal() error = %v", err)
	}
	// The upstream's first completion carried reasoning_content with the tool
	// call; the internal loop must echo it on the assistant message it appends.
	p1 := &fakeProvider{
		name: "p1",
		scripted: []chat.ChatResponse{
			{
				FinishReason:     "tool_calls",
				ReasoningContent: "thinking...",
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

	rec := doJSON(t, h, http.MethodPost, "/v1/chat/completions", map[string]any{
		"model":    "virtual-a",
		"messages": []map[string]string{{"role": "user", "content": "cuaca hari ini?"}},
	})

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
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
	if asst.ReasoningContent != "thinking..." {
		t.Errorf("assistant ReasoningContent = %q, want thinking... (echoed from upstream response)", asst.ReasoningContent)
	}
}

func TestChatCompletions_StreamAllCandidatesFail(t *testing.T) {
	// Every candidate errors before delivering any content: the client must get
	// an error frame, not a misleading 200 with an empty stream.
	p1 := &fakeProvider{
		name: "p1",
		stream: func(context.Context, chat.ChatRequest, chat.StreamFunc) error {
			return errors.New("p1 down")
		},
	}
	p2 := &fakeProvider{
		name: "p2",
		stream: func(context.Context, chat.ChatRequest, chat.StreamFunc) error {
			return errors.New("p2 down")
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
		t.Errorf("body missing [DONE]: %q", body)
	}
	if !strings.Contains(body, `"error"`) {
		t.Errorf("body missing error frame (misleading empty 200): %q", body)
	}
	if strings.Contains(body, `"finish_reason":"stop"`) || strings.Contains(body, `"finish_reason":"tool_calls"`) {
		t.Errorf("body contains a success finish_reason: %q", body)
	}
}
