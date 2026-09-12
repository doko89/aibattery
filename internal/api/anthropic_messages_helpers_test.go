package api

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/aibattery/router/internal/chat"
)

func anthropicCompleteProvider(name string, response chat.ChatResponse, onRequest func(chat.ChatRequest)) *fakeProvider {
	return &fakeProvider{
		name: name,
		complete: func(_ context.Context, request chat.ChatRequest) (chat.ChatResponse, error) {
			if onRequest != nil {
				onRequest(request)
			}
			return response, nil
		},
	}
}

func anthropicMessagesBody(model string, maxTokens int, content any) map[string]any {
	body := map[string]any{
		"model":    model,
		"messages": []map[string]any{{"role": "user", "content": content}},
	}
	if maxTokens != 0 {
		body["max_tokens"] = maxTokens
	}
	return body
}

func decodeSuccessfulAnthropicMessage(t *testing.T, rec *httptest.ResponseRecorder) anthropicTestMessage {
	t.Helper()
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	var out anthropicTestMessage
	if err := json.Unmarshal(rec.Body.Bytes(), &out); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	return out
}

func decodeAnthropicError(t *testing.T, rec *httptest.ResponseRecorder) anthropicTestError {
	t.Helper()
	var errResp anthropicTestError
	if err := json.Unmarshal(rec.Body.Bytes(), &errResp); err != nil {
		t.Fatalf("decode error: %v", err)
	}
	return errResp
}
