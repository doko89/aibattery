package adapter

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/aibattery/router/internal/chat"
)

// captureRequest decodes the request body the server received.
func captureRequest(t *testing.T, r *http.Request) openAIRequest {
	t.Helper()
	var body openAIRequest
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
		t.Fatalf("decode request body: %v", err)
	}
	return body
}

func TestComplete_TranslationAndParsing(t *testing.T) {
	var got openAIRequest
	var gotAuth, gotContentType string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if r.Method != http.MethodPost {
			t.Errorf("unexpected method: %s", r.Method)
		}
		gotAuth = r.Header.Get("Authorization")
		gotContentType = r.Header.Get("Content-Type")
		got = captureRequest(t, r)

		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{
			"id": "chatcmpl-123",
			"model": "gpt-4o-mini",
			"choices": [{
				"index": 0,
				"message": {"role": "assistant", "content": "Hello there!"},
				"finish_reason": "stop"
			}],
			"usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21}
		}`)
	}))
	defer srv.Close()

	p := NewOpenAIProvider(srv.URL, "sk-test", 5*time.Second)
	if p.Name() != "openai" {
		t.Fatalf("Name() = %q, want openai", p.Name())
	}

	temp := 0.7
	maxTok := 100
	req := chat.ChatRequest{
		Model: "gpt-4o-mini",
		Messages: []chat.Message{
			{Role: chat.RoleSystem, Content: "You are helpful."},
			{Role: chat.RoleUser, Content: "Hi"},
		},
		Temperature: &temp,
		MaxTokens:   &maxTok,
	}

	resp, err := p.Complete(context.Background(), req)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}

	if gotAuth != "Bearer sk-test" {
		t.Errorf("Authorization = %q, want Bearer sk-test", gotAuth)
	}
	if gotContentType != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", gotContentType)
	}
	if got.Model != "gpt-4o-mini" {
		t.Errorf("model = %q", got.Model)
	}
	if len(got.Messages) != 2 {
		t.Fatalf("messages len = %d, want 2", len(got.Messages))
	}
	if got.Messages[0].Role != "system" || got.Messages[0].Content != "You are helpful." {
		t.Errorf("messages[0] = %+v", got.Messages[0])
	}
	if got.Messages[1].Role != "user" || got.Messages[1].Content != "Hi" {
		t.Errorf("messages[1] = %+v", got.Messages[1])
	}
	if got.Temperature == nil || *got.Temperature != 0.7 {
		t.Errorf("temperature = %v, want 0.7", got.Temperature)
	}
	if got.MaxTokens == nil || *got.MaxTokens != 100 {
		t.Errorf("max_tokens = %v, want 100", got.MaxTokens)
	}
	if got.Stream {
		t.Errorf("stream should be false for non-streaming")
	}
	if got.ReasoningEffort != nil {
		t.Errorf("reasoning_effort should be omitted, got %v", *got.ReasoningEffort)
	}

	// Second request: reasoning effort set — temperature must be dropped.
	effort := "high"
	req2 := req
	req2.ReasoningEffort = &effort
	if _, err := p.Complete(context.Background(), req2); err != nil {
		t.Fatalf("Complete (reasoning): %v", err)
	}
	if got.ReasoningEffort == nil || *got.ReasoningEffort != "high" {
		t.Errorf("reasoning_effort = %v, want high", got.ReasoningEffort)
	}
	if got.Temperature != nil {
		t.Errorf("temperature should be dropped when reasoning_effort is set, got %v", *got.Temperature)
	}

	if resp.ID != "chatcmpl-123" {
		t.Errorf("ID = %q", resp.ID)
	}
	if resp.Model != "gpt-4o-mini" {
		t.Errorf("Model = %q", resp.Model)
	}
	if resp.Content != "Hello there!" {
		t.Errorf("Content = %q", resp.Content)
	}
	if resp.FinishReason != "stop" {
		t.Errorf("FinishReason = %q, want stop", resp.FinishReason)
	}
	if resp.Usage.PromptTokens != 9 || resp.Usage.CompletionTokens != 12 || resp.Usage.TotalTokens != 21 {
		t.Errorf("Usage = %+v", resp.Usage)
	}
}

func TestComplete_OmitEmptyFields(t *testing.T) {
	var got openAIRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = captureRequest(t, r)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"id":"x","model":"m","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer srv.Close()

	p := NewOpenAIProvider(srv.URL, "k", time.Second)
	_, err := p.Complete(context.Background(), chat.ChatRequest{Model: "m", Messages: []chat.Message{{Role: chat.RoleUser, Content: "hi"}}})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if got.Temperature != nil {
		t.Errorf("temperature should be omitted, got %v", got.Temperature)
	}
	if got.MaxTokens != nil {
		t.Errorf("max_tokens should be omitted, got %v", got.MaxTokens)
	}
	if got.ReasoningEffort != nil {
		t.Errorf("reasoning_effort should be omitted, got %v", *got.ReasoningEffort)
	}
}

func TestComplete_TransportError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		fmt.Fprint(w, `{"error":{"message":"rate limited","type":"rate_limit_error"}}`)
	}))
	defer srv.Close()

	p := NewOpenAIProvider(srv.URL, "k", time.Second)
	_, err := p.Complete(context.Background(), chat.ChatRequest{Model: "m", Messages: []chat.Message{{Role: chat.RoleUser, Content: "hi"}}})
	if err == nil {
		t.Fatal("expected error for 429")
	}
	if !strings.Contains(err.Error(), "status 429") {
		t.Errorf("error = %q, want to contain status 429", err)
	}
	if !chat.IsRateLimit(err) {
		t.Fatalf("error = %v, want IsRateLimit to be true", err)
	}
}

func TestComplete_SerializesTools(t *testing.T) {
	var got openAIRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = captureRequest(t, r)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"id":"x","model":"m","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer srv.Close()

	p := NewOpenAIProvider(srv.URL, "k", time.Second)
	_, err := p.Complete(context.Background(), chat.ChatRequest{
		Model:    "m",
		Messages: []chat.Message{{Role: chat.RoleUser, Content: "hi"}},
		Tools: []chat.Tool{
			{
				Name:        "get_weather",
				Description: "Get weather for a location",
				InputSchema: map[string]any{
					"type":       "object",
					"properties": map[string]any{"location": map[string]any{"type": "string"}},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if len(got.Tools) != 1 {
		t.Fatalf("tools len = %d, want 1", len(got.Tools))
	}
	tool := got.Tools[0]
	if tool.Type != "function" {
		t.Errorf("tool.Type = %q, want function", tool.Type)
	}
	if tool.Function.Name != "get_weather" {
		t.Errorf("tool.Function.Name = %q, want get_weather", tool.Function.Name)
	}
	if tool.Function.Description != "Get weather for a location" {
		t.Errorf("tool.Function.Description = %q", tool.Function.Description)
	}
	if tool.Function.Parameters == nil {
		t.Fatal("expected parameters to be serialized")
	}
	if tool.Function.Parameters["type"] != "object" {
		t.Errorf("parameters.type = %v, want object", tool.Function.Parameters["type"])
	}
}

func TestComplete_ParsesToolCalls(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{
			"id": "chatcmpl-tc",
			"model": "gpt-4o-mini",
			"choices": [{
				"index": 0,
				"message": {
					"role": "assistant",
					"content": null,
					"tool_calls": [{
						"id": "call_abc123",
						"type": "function",
						"function": {"name": "get_weather", "arguments": "{\"location\":\"Jakarta\"}"}
					}]
				},
				"finish_reason": "tool_calls"
			}],
			"usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21}
		}`)
	}))
	defer srv.Close()

	p := NewOpenAIProvider(srv.URL, "k", time.Second)
	resp, err := p.Complete(context.Background(), chat.ChatRequest{Model: "m", Messages: []chat.Message{{Role: chat.RoleUser, Content: "hi"}}})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.FinishReason != "tool_calls" {
		t.Errorf("FinishReason = %q, want tool_calls", resp.FinishReason)
	}
	if resp.Content != "" {
		t.Errorf("Content = %q, want empty", resp.Content)
	}
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("ToolCalls len = %d, want 1", len(resp.ToolCalls))
	}
	tc := resp.ToolCalls[0]
	if tc.ID != "call_abc123" {
		t.Errorf("ToolCall.ID = %q, want call_abc123", tc.ID)
	}
	if tc.Name != "get_weather" {
		t.Errorf("ToolCall.Name = %q, want get_weather", tc.Name)
	}
	if string(tc.Arguments) != `{"location":"Jakarta"}` {
		t.Errorf("ToolCall.Arguments = %s", tc.Arguments)
	}
}

func TestStream_ToolCallAccumulation(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, ""+
			"data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]}}]}\n\n"+
			"data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"loc\"}}]}}]}\n\n"+
			"data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"ation\\\":\\\"Jakarta\\\"}\"}}]}}]}\n\n"+
			"data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n"+
			"data: [DONE]\n\n",
		)
	}))
	defer srv.Close()

	p := NewOpenAIProvider(srv.URL, "k", time.Second)

	var deltas []chat.StreamDelta
	err := p.Stream(context.Background(), chat.ChatRequest{Model: "m", Messages: []chat.Message{{Role: chat.RoleUser, Content: "hi"}}}, func(d chat.StreamDelta) error {
		deltas = append(deltas, d)
		return nil
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}

	if len(deltas) != 1 {
		t.Fatalf("got %d deltas, want 1: %+v", len(deltas), deltas)
	}
	final := deltas[0]
	if final.FinishReason != "tool_calls" {
		t.Errorf("FinishReason = %q, want tool_calls", final.FinishReason)
	}
	if len(final.ToolCalls) != 1 {
		t.Fatalf("ToolCalls len = %d, want 1", len(final.ToolCalls))
	}
	tc := final.ToolCalls[0]
	if tc.ID != "call_1" {
		t.Errorf("ToolCall.ID = %q, want call_1", tc.ID)
	}
	if tc.Name != "get_weather" {
		t.Errorf("ToolCall.Name = %q, want get_weather", tc.Name)
	}
	if string(tc.Arguments) != `{"location":"Jakarta"}` {
		t.Errorf("ToolCall.Arguments = %s, want {\"location\":\"Jakarta\"}", tc.Arguments)
	}
}

func TestStream_SSEParsing(t *testing.T) {
	var got openAIRequest
	var gotAccept string

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		got = captureRequest(t, r)
		gotAccept = r.Header.Get("Accept")
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, ""+
			"data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}\n\n"+
			"data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"}}]}\n\n"+
			"data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"}}]}\n\n"+
			"data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":7,\"total_tokens\":12}}\n\n"+
			"data: [DONE]\n\n",
		)
	}))
	defer srv.Close()

	p := NewOpenAIProvider(srv.URL, "k", time.Second)

	var deltas []chat.StreamDelta
	err := p.Stream(context.Background(), chat.ChatRequest{Model: "m", Messages: []chat.Message{{Role: chat.RoleUser, Content: "hi"}}}, func(d chat.StreamDelta) error {
		deltas = append(deltas, d)
		return nil
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}

	if !got.Stream {
		t.Errorf("stream should be true in request body")
	}
	if gotAccept != "text/event-stream" {
		t.Errorf("Accept = %q, want text/event-stream", gotAccept)
	}

	if len(deltas) != 3 {
		t.Fatalf("got %d deltas, want 3: %+v", len(deltas), deltas)
	}
	if deltas[0].Delta != "Hello" || deltas[0].FinishReason != "" {
		t.Errorf("deltas[0] = %+v", deltas[0])
	}
	if deltas[1].Delta != " world" {
		t.Errorf("deltas[1] = %+v", deltas[1])
	}
	if deltas[2].FinishReason != "stop" {
		t.Errorf("deltas[2].FinishReason = %q, want stop", deltas[2].FinishReason)
	}
	if deltas[2].Usage == nil {
		t.Fatal("expected usage on final chunk")
	}
	if deltas[2].Usage.PromptTokens != 5 || deltas[2].Usage.CompletionTokens != 7 || deltas[2].Usage.TotalTokens != 12 {
		t.Errorf("final usage = %+v", deltas[2].Usage)
	}
}

func TestStream_ErrorBeforeFirstDelta(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		fmt.Fprint(w, `{"error":{"message":"slow down"}}`)
	}))
	defer srv.Close()

	p := NewOpenAIProvider(srv.URL, "k", time.Second)
	err := p.Stream(context.Background(), chat.ChatRequest{Model: "m", Messages: []chat.Message{{Role: chat.RoleUser, Content: "hi"}}}, func(chat.StreamDelta) error {
		return nil
	})
	if err == nil {
		t.Fatal("expected error for 429 stream")
	}
	if !strings.Contains(err.Error(), "status 429") {
		t.Errorf("error = %q, want status 429", err)
	}
	if !chat.IsRateLimit(err) {
		t.Fatalf("error = %v, want IsRateLimit to be true", err)
	}
}

func TestStream_ErrorAfterFirstDelta(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		// Emit one content delta, then a malformed frame.
		fmt.Fprint(w, ""+
			"data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n"+
			"data: not-json\n\n",
		)
	}))
	defer srv.Close()

	p := NewOpenAIProvider(srv.URL, "k", time.Second)

	var deltas []chat.StreamDelta
	err := p.Stream(context.Background(), chat.ChatRequest{Model: "m", Messages: []chat.Message{{Role: chat.RoleUser, Content: "hi"}}}, func(d chat.StreamDelta) error {
		deltas = append(deltas, d)
		return nil
	})
	if err != nil {
		t.Fatalf("Stream should return nil after first delta, got %v", err)
	}
	if len(deltas) != 2 {
		t.Fatalf("got %d deltas, want 2: %+v", len(deltas), deltas)
	}
	if deltas[0].Delta != "partial" {
		t.Errorf("deltas[0] = %+v", deltas[0])
	}
	if deltas[1].FinishReason != "error" {
		t.Errorf("deltas[1].FinishReason = %q, want error", deltas[1].FinishReason)
	}
}
