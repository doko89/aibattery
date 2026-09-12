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
	if err := decodeTestBody(r, &body); err != nil {
		t.Fatalf("decode request body: %v", err)
	}
	return body
}

// openAICapture records what a fake OpenAI upstream received.
type openAICapture struct {
	body        openAIRequest
	auth        string
	contentType string
}

// newOpenAICompleteServer starts a fake OpenAI chat-completions upstream that
// checks path/method, records the request into capture, and replies with respBody.
func newOpenAICompleteServer(t *testing.T, capture *openAICapture, respBody string) *httptest.Server {
	t.Helper()
	srv := newMockServer(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/chat/completions" {
			t.Errorf("unexpected path: %s", r.URL.Path)
		}
		if r.Method != http.MethodPost {
			t.Errorf("unexpected method: %s", r.Method)
		}
		capture.auth = r.Header.Get("Authorization")
		capture.contentType = r.Header.Get("Content-Type")
		capture.body = captureRequest(t, r)

		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, respBody)
	})
	return srv
}

const openAICompleteStub = `{
	"id": "chatcmpl-123",
	"model": "gpt-4o-mini",
	"choices": [{
		"index": 0,
		"message": {"role": "assistant", "content": "Hello there!"},
		"finish_reason": "stop"
	}],
	"usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21}
}`

// openAIMinimalOKStub is the minimal successful chat-completion payload reused
// by the tests that only assert on the outgoing request body.
const openAIMinimalOKStub = `{"id":"x","model":"m","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`

func assertOpenAIHeaders(t *testing.T, capture *openAICapture) {
	t.Helper()
	if capture.auth != "Bearer sk-test" {
		t.Errorf("Authorization = %q, want Bearer sk-test", capture.auth)
	}
	if capture.contentType != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", capture.contentType)
	}
}

func assertOpenAIMessages(t *testing.T, got openAIRequest) {
	t.Helper()
	if got.Model != "gpt-4o-mini" {
		t.Errorf("model = %q", got.Model)
	}
	if len(got.Messages) != 2 {
		t.Fatalf("messages len = %d, want 2", len(got.Messages))
	}
	checkOpenAIMessagePair(t, got)
}

func checkOpenAIMessagePair(t *testing.T, got openAIRequest) {
	t.Helper()
	if got.Messages[0].Role != "system" || got.Messages[0].Content != "You are helpful." {
		t.Errorf("messages[0] = %+v", got.Messages[0])
	}
	if got.Messages[1].Role != "user" || got.Messages[1].Content != "Hi" {
		t.Errorf("messages[1] = %+v", got.Messages[1])
	}
}

func assertOpenAIOptionals(t *testing.T, got openAIRequest) {
	t.Helper()
	if got.Temperature == nil || *got.Temperature != 0.7 {
		t.Errorf("temperature = %v, want 0.7", got.Temperature)
	}
	if got.MaxTokens == nil || *got.MaxTokens != 100 {
		t.Errorf("max_tokens = %v, want 100", got.MaxTokens)
	}
	checkOpenAINonStreaming(t, got)
}

func checkOpenAINonStreaming(t *testing.T, got openAIRequest) {
	t.Helper()
	if got.Stream {
		t.Errorf("stream should be false for non-streaming")
	}
	if got.ReasoningEffort != nil {
		t.Errorf("reasoning_effort should be omitted, got %v", *got.ReasoningEffort)
	}
}

func assertOpenAIChatResponse(t *testing.T, resp chat.ChatResponse) {
	t.Helper()
	if resp.ID != "chatcmpl-123" {
		t.Errorf("ID = %q", resp.ID)
	}
	if resp.Model != "gpt-4o-mini" {
		t.Errorf("Model = %q", resp.Model)
	}
	checkOpenAIResponseContent(t, resp)
}

func checkOpenAIResponseContent(t *testing.T, resp chat.ChatResponse) {
	t.Helper()
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

func TestComplete_TranslationAndParsing(t *testing.T) {
	var capture openAICapture
	srv := newOpenAICompleteServer(t, &capture, openAICompleteStub)

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

	assertOpenAIHeaders(t, &capture)
	assertOpenAIMessages(t, capture.body)
	assertOpenAIOptionals(t, capture.body)
	assertOpenAIChatResponse(t, resp)
}

// TestComplete_ReasoningEffortDropsTemperature verifies that setting
// reasoning effort passes it through and drops temperature.
func TestComplete_ReasoningEffortDropsTemperature(t *testing.T) {
	var capture openAICapture
	srv := newOpenAICompleteServer(t, &capture, openAICompleteStub)

	p := NewOpenAIProvider(srv.URL, "sk-test", 5*time.Second)

	temp := 0.7
	maxTok := 100
	effort := "high"
	req := chat.ChatRequest{
		Model: "gpt-4o-mini",
		Messages: []chat.Message{
			{Role: chat.RoleSystem, Content: "You are helpful."},
			{Role: chat.RoleUser, Content: "Hi"},
		},
		Temperature:     &temp,
		MaxTokens:       &maxTok,
		ReasoningEffort: &effort,
	}
	if _, err := p.Complete(context.Background(), req); err != nil {
		t.Fatalf("Complete (reasoning): %v", err)
	}
	assertOpenAIReasoningEffort(t, capture.body)
}

func assertOpenAIReasoningEffort(t *testing.T, got openAIRequest) {
	t.Helper()
	if got.ReasoningEffort == nil || *got.ReasoningEffort != "high" {
		t.Errorf("reasoning_effort = %v, want high", got.ReasoningEffort)
	}
	checkOpenAITemperatureDropped(t, got)
}

func checkOpenAITemperatureDropped(t *testing.T, got openAIRequest) {
	t.Helper()
	if got.Temperature != nil {
		t.Errorf("temperature should be dropped when reasoning_effort is set, got %v", *got.Temperature)
	}
}

func TestOpenAI_ReasoningContentBody(t *testing.T) {
	var got openAIRequest
	srv := newCaptureJSONServer(t, openAIMinimalOKStub, func(r *http.Request) {
		got = captureRequest(t, r)
	})

	p := NewOpenAIProvider(srv.URL, "k", time.Second)
	_, err := p.Complete(context.Background(), chat.ChatRequest{
		Model: "m",
		Messages: []chat.Message{
			{Role: chat.RoleAssistant, Content: "first", ReasoningContent: "thinking..."},
			{Role: chat.RoleAssistant, Content: "plain"},
		},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}

	if len(got.Messages) != 2 {
		t.Fatalf("messages len = %d, want 2", len(got.Messages))
	}
	if got.Messages[0].ReasoningContent != "thinking..." {
		t.Errorf("messages[0].reasoning_content = %q, want thinking...", got.Messages[0].ReasoningContent)
	}
	if got.Messages[1].ReasoningContent != "" {
		t.Errorf("messages[1].reasoning_content = %q, want empty", got.Messages[1].ReasoningContent)
	}
	// A message without reasoning content must not carry the key on the wire.
	raw, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("marshal body: %v", err)
	}
	if strings.Contains(string(raw), `"reasoning_content":""`) {
		t.Errorf("wire body contains empty reasoning_content key; want it omitted (omitempty)")
	}
}

func TestOpenAI_ToolRoundTripBody(t *testing.T) {
	var got openAIRequest
	srv := newCaptureJSONServer(t, openAIMinimalOKStub, func(r *http.Request) {
		got = captureRequest(t, r)
	})

	p := NewOpenAIProvider(srv.URL, "k", time.Second)
	_, err := p.Complete(context.Background(), chat.ChatRequest{
		Model: "m",
		Messages: []chat.Message{
			{Role: chat.RoleUser, Content: "hi"},
			{
				Role: chat.RoleAssistant,
				ToolCalls: []chat.ToolCall{
					{ID: "call_1", Name: "search_web", Arguments: json.RawMessage(`{"query":"x"}`)},
				},
			},
			{Role: chat.Role("tool"), ToolCallID: "call_1", Content: "result"},
		},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}

	if len(got.Messages) != 3 {
		t.Fatalf("messages len = %d, want 3", len(got.Messages))
	}
	assistant := got.Messages[1]
	if assistant.Role != "assistant" {
		t.Errorf("messages[1].Role = %q, want assistant", assistant.Role)
	}
	if len(assistant.ToolCalls) != 1 {
		t.Fatalf("messages[1].ToolCalls len = %d, want 1", len(assistant.ToolCalls))
	}
	tc := assistant.ToolCalls[0]
	if tc.ID != "call_1" {
		t.Errorf("tool_calls[0].id = %q, want call_1", tc.ID)
	}
	if tc.Type != "function" {
		t.Errorf("tool_calls[0].type = %q, want function", tc.Type)
	}
	if tc.Function.Name != "search_web" {
		t.Errorf("tool_calls[0].function.name = %q, want search_web", tc.Function.Name)
	}
	if tc.Function.Arguments != `{"query":"x"}` {
		t.Errorf("tool_calls[0].function.arguments = %q, want {\"query\":\"x\"}", tc.Function.Arguments)
	}
	if assistant.ToolCallID != "" {
		t.Errorf("messages[1].tool_call_id = %q, want empty", assistant.ToolCallID)
	}
	toolMsg := got.Messages[2]
	if toolMsg.Role != "tool" {
		t.Errorf("messages[2].Role = %q, want tool", toolMsg.Role)
	}
	if toolMsg.Content != "result" {
		t.Errorf("messages[2].Content = %q, want result", toolMsg.Content)
	}
	if toolMsg.ToolCallID != "call_1" {
		t.Errorf("messages[2].tool_call_id = %q, want call_1", toolMsg.ToolCallID)
	}
	if len(toolMsg.ToolCalls) != 0 {
		t.Errorf("messages[2].ToolCalls len = %d, want 0", len(toolMsg.ToolCalls))
	}
}

func TestComplete_OmitEmptyFields(t *testing.T) {
	var got openAIRequest
	srv := newCaptureJSONServer(t, openAIMinimalOKStub, func(r *http.Request) {
		got = captureRequest(t, r)
	})

	p := NewOpenAIProvider(srv.URL, "k", time.Second)
	_, err := p.Complete(context.Background(), simpleUserRequest("m"))
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
	srv := newStaticStatusServer(t, http.StatusTooManyRequests, `{"error":{"message":"rate limited","type":"rate_limit_error"}}`)

	p := NewOpenAIProvider(srv.URL, "k", time.Second)
	_, err := p.Complete(context.Background(), simpleUserRequest("m"))
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
	srv := newCaptureJSONServer(t, openAIMinimalOKStub, func(r *http.Request) {
		got = captureRequest(t, r)
	})

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
	srv := newStaticJSONServer(t, `{
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

	p := NewOpenAIProvider(srv.URL, "k", time.Second)
	resp, err := p.Complete(context.Background(), simpleUserRequest("m"))
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
	srv := newSSEServer(t, ""+
		"data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]}}]}\n\n"+
		"data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"loc\"}}]}}]}\n\n"+
		"data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"ation\\\":\\\"Jakarta\\\"}\"}}]}}]}\n\n"+
		"data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n"+
		"data: [DONE]\n\n",
	)

	p := NewOpenAIProvider(srv.URL, "k", time.Second)

	deltas, err := collectStream(t, p, simpleUserRequest("m"))
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

	srv := newCaptureSSEServer(t, ""+
		"data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}\n\n"+
		"data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello\"}}]}\n\n"+
		"data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\" world\"}}]}\n\n"+
		"data: {\"id\":\"c1\",\"model\":\"m\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":7,\"total_tokens\":12}}\n\n"+
		"data: [DONE]\n\n",
		func(r *http.Request) {
			got = captureRequest(t, r)
			gotAccept = r.Header.Get("Accept")
		})

	p := NewOpenAIProvider(srv.URL, "k", time.Second)

	deltas, err := collectStream(t, p, simpleUserRequest("m"))
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
	srv := newStaticStatusServer(t, http.StatusTooManyRequests, `{"error":{"message":"slow down"}}`)

	p := NewOpenAIProvider(srv.URL, "k", time.Second)
	_, err := collectStream(t, p, simpleUserRequest("m"))
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
	// Emit one content delta, then a malformed frame.
	srv := newSSEServer(t, ""+
		"data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n"+
		"data: not-json\n\n",
	)

	p := NewOpenAIProvider(srv.URL, "k", time.Second)

	deltas, err := collectStream(t, p, simpleUserRequest("m"))
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

func TestOpenAI_ReasoningContentResponse(t *testing.T) {
	srv := newStaticJSONServer(t, `{
		"id": "chatcmpl-r",
		"model": "deepseek-v4-pro",
		"choices": [{
			"index": 0,
			"message": {
				"role": "assistant",
				"content": null,
				"reasoning_content": "thinking...",
				"tool_calls": [{
					"id": "call_1",
					"type": "function",
					"function": {"name": "search_web", "arguments": "{\"query\":\"x\"}"}
				}]
			},
			"finish_reason": "tool_calls"
		}],
		"usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21}
	}`)

	p := NewOpenAIProvider(srv.URL, "k", time.Second)
	resp, err := p.Complete(context.Background(), simpleUserRequest("m"))
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.ReasoningContent != "thinking..." {
		t.Errorf("ReasoningContent = %q, want thinking...", resp.ReasoningContent)
	}
	if resp.FinishReason != "tool_calls" {
		t.Errorf("FinishReason = %q, want tool_calls", resp.FinishReason)
	}
	if len(resp.ToolCalls) != 1 || resp.ToolCalls[0].ID != "call_1" {
		t.Errorf("ToolCalls = %+v, want call_1", resp.ToolCalls)
	}
}

// DeepSeek streams reasoning_content fragments BEFORE the tool-call
// deltas; the fragments must be concatenated onto the final chunk.
const openAIReasoningToolCallsFrames = "" +
	"data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"think\"}}]}\n\n" +
	"data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"ing...\"}}]}\n\n" +
	"data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"loc\\\":\\\"Jakarta\\\"}\"}}]}}]}\n\n" +
	"data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n" +
	"data: [DONE]\n\n"

const openAIReasoningPlainContentFrames = "" +
	"data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"let me think\"}}]}\n\n" +
	"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"answer\"}}]}\n\n" +
	"data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
	"data: [DONE]\n\n"

// streamOpenAIReasoning serves frames from a fake SSE upstream and collects
// the resulting deltas.
func streamOpenAIReasoning(t *testing.T, frames string) []chat.StreamDelta {
	t.Helper()
	srv := newSSEServer(t, frames)

	p := NewOpenAIProvider(srv.URL, "k", time.Second)
	deltas, err := collectStream(t, p, simpleUserRequest("m"))
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	return deltas
}

func assertReasoningToolCallsFinal(t *testing.T, deltas []chat.StreamDelta) {
	t.Helper()
	if len(deltas) != 1 {
		t.Fatalf("got %d deltas, want 1: %+v", len(deltas), deltas)
	}
	final := deltas[0]
	if final.FinishReason != "tool_calls" {
		t.Errorf("FinishReason = %q, want tool_calls", final.FinishReason)
	}
	checkReasoningConcatenated(t, final)
}

func checkReasoningConcatenated(t *testing.T, final chat.StreamDelta) {
	t.Helper()
	if final.ReasoningContent != "thinking..." {
		t.Errorf("ReasoningContent = %q, want thinking... (concatenated fragments)", final.ReasoningContent)
	}
	if len(final.ToolCalls) != 1 || final.ToolCalls[0].ID != "call_1" {
		t.Errorf("ToolCalls = %+v, want call_1", final.ToolCalls)
	}
}

func assertReasoningPlainContent(t *testing.T, deltas []chat.StreamDelta) {
	t.Helper()
	// content delta is emitted as-is; reasoning lands only on the final chunk
	if len(deltas) != 2 {
		t.Fatalf("got %d deltas, want 2: %+v", len(deltas), deltas)
	}
	if deltas[0].Delta != "answer" {
		t.Errorf("deltas[0] = %+v, want content delta", deltas[0])
	}
	checkReasoningOnFinalChunkOnly(t, deltas)
}

func checkReasoningOnFinalChunkOnly(t *testing.T, deltas []chat.StreamDelta) {
	t.Helper()
	if deltas[0].ReasoningContent != "" {
		t.Errorf("deltas[0].ReasoningContent = %q, want empty (reasoning only on final chunk)", deltas[0].ReasoningContent)
	}
	final := deltas[1]
	if final.FinishReason != "stop" {
		t.Errorf("FinishReason = %q, want stop", final.FinishReason)
	}
	if final.ReasoningContent != "let me think" {
		t.Errorf("ReasoningContent = %q, want let me think", final.ReasoningContent)
	}
}

func TestOpenAI_ReasoningContentStream(t *testing.T) {
	t.Run("tool calls", func(t *testing.T) {
		deltas := streamOpenAIReasoning(t, openAIReasoningToolCallsFrames)
		assertReasoningToolCallsFinal(t, deltas)
	})

	t.Run("plain content", func(t *testing.T) {
		deltas := streamOpenAIReasoning(t, openAIReasoningPlainContentFrames)
		assertReasoningPlainContent(t, deltas)
	})
}

func TestOpenAI_StreamTimeoutBudget(t *testing.T) {
	// Provider timeout is tiny (30ms); the upstream thinks for 100ms before the
	// first chunk. Streaming must NOT be bound by the provider timeout: it uses
	// the generous streamTimeout budget, so this must succeed.
	srv := newMockServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		time.Sleep(100 * time.Millisecond)
		fmt.Fprint(w, ""+
			"data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"}}]}\n\n"+
			"data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"+
			"data: [DONE]\n\n",
		)
	})

	p := NewOpenAIProvider(srv.URL, "k", 30*time.Millisecond)
	deltas, err := collectStream(t, p, simpleUserRequest("m"))
	if err != nil {
		t.Fatalf("Stream returned error %v; streaming must not be bound by the provider timeout", err)
	}
	if len(deltas) != 2 || deltas[0].Delta != "ok" || deltas[1].FinishReason != "stop" {
		t.Errorf("deltas = %+v, want content ok then finish stop", deltas)
	}
}

func TestOpenAI_CompleteRespectsTimeout(t *testing.T) {
	// The upstream sleeps longer than the provider timeout; Complete must fail
	// with a context deadline error (the configured timeout is a per-call
	// budget, not a total http.Client deadline).
	srv := newMockServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		time.Sleep(200 * time.Millisecond)
		fmt.Fprint(w, `{"id":"x","object":"chat.completion","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"late"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	})

	p := NewOpenAIProvider(srv.URL, "k", 30*time.Millisecond)
	_, err := p.Complete(context.Background(), simpleUserRequest("m"))
	if err == nil {
		t.Fatal("Complete succeeded; want timeout error")
	}
	if !strings.Contains(err.Error(), "deadline exceeded") {
		t.Errorf("Complete error = %v, want context deadline exceeded", err)
	}
}
