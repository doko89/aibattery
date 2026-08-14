package adapter

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/aibattery/router/internal/chat"
)

// anthropicCaptureRequest decodes the Anthropic request body received by the test
// server.
type anthropicCaptureRequest struct {
	Model       string             `json:"model"`
	MaxTokens   int                `json:"max_tokens"`
	Temperature *float64           `json:"temperature"`
	Stream      bool               `json:"stream"`
	System      string             `json:"system"`
	Messages    []anthropicMessage `json:"messages"`
	Tools       []anthropicTool    `json:"tools"`
	Thinking    *anthropicThinking `json:"thinking"`
}

func newTestProvider(t *testing.T, handler http.HandlerFunc) (*httptest.Server, chat.Provider) {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	return srv, NewAnthropicProvider(srv.URL, "test-key", "2023-06-01", 5*time.Second)
}

func TestAnthropicComplete_Translation(t *testing.T) {
	var got anthropicCaptureRequest
	var gotHeaders http.Header

	_, p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		gotHeaders = r.Header.Clone()
		if r.URL.Path != "/messages" {
			t.Errorf("unexpected path %q", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&got); err != nil {
			t.Errorf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"msg_123","type":"message","role":"assistant","model":"claude-sonnet-4-6",
			"content":[{"type":"text","text":"Hello "},{"type":"text","text":"world"}],
			"stop_reason":"end_turn","stop_sequence":null,
			"usage":{"input_tokens":12,"output_tokens":18}
		}`))
	})

	temp := 0.7
	resp, err := p.Complete(context.Background(), chat.ChatRequest{
		Model:       "claude-sonnet-4-6",
		System:      "You are helpful.",
		Temperature: &temp,
		Messages: []chat.Message{
			{Role: chat.RoleSystem, Content: "ignored system"},
			{Role: chat.RoleUser, Content: "Hi"},
			{Role: chat.RoleAssistant, Content: "Hello"},
		},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}

	// Headers.
	if gotHeaders.Get("x-api-key") != "test-key" {
		t.Errorf("x-api-key = %q", gotHeaders.Get("x-api-key"))
	}
	if gotHeaders.Get("anthropic-version") != "2023-06-01" {
		t.Errorf("anthropic-version = %q", gotHeaders.Get("anthropic-version"))
	}
	if gotHeaders.Get("Content-Type") != "application/json" {
		t.Errorf("Content-Type = %q", gotHeaders.Get("Content-Type"))
	}

	// Request body translation.
	if got.Model != "claude-sonnet-4-6" {
		t.Errorf("model = %q", got.Model)
	}
	if got.MaxTokens != 1024 {
		t.Errorf("max_tokens default = %d, want 1024", got.MaxTokens)
	}
	if got.System != "You are helpful." {
		t.Errorf("system = %q", got.System)
	}
	if got.Temperature == nil || *got.Temperature != 0.7 {
		t.Errorf("temperature = %v", got.Temperature)
	}
	if got.Stream {
		t.Error("stream should be false for Complete")
	}
	// RoleSystem dropped, only user+assistant remain.
	if len(got.Messages) != 2 {
		t.Fatalf("messages len = %d, want 2", len(got.Messages))
	}
	if got.Messages[0].Role != "user" || got.Messages[0].Content != "Hi" {
		t.Errorf("messages[0] = %+v", got.Messages[0])
	}
	if got.Messages[1].Role != "assistant" || got.Messages[1].Content != "Hello" {
		t.Errorf("messages[1] = %+v", got.Messages[1])
	}

	// Response extraction.
	if resp.ID != "msg_123" {
		t.Errorf("id = %q", resp.ID)
	}
	if resp.Content != "Hello world" {
		t.Errorf("content = %q", resp.Content)
	}
	if resp.FinishReason != "stop" {
		t.Errorf("finish_reason = %q, want stop", resp.FinishReason)
	}
	if resp.Usage.PromptTokens != 12 || resp.Usage.CompletionTokens != 18 || resp.Usage.TotalTokens != 30 {
		t.Errorf("usage = %+v", resp.Usage)
	}
}

func TestAnthropicComplete_Thinking(t *testing.T) {
	tests := []struct {
		name       string
		model      string
		effort     *string
		wantType   string
		wantBudget int
		wantMax    int
		wantAbsent bool
	}{
		{name: "legacy enabled with budget", model: "claude-sonnet-4-5", effort: strPtr("high"), wantType: "enabled", wantBudget: 16000, wantMax: 16001},
		{name: "modern adaptive", model: "claude-sonnet-4-6", effort: strPtr("high"), wantType: "adaptive", wantMax: 1024},
		{name: "disabled", model: "claude-sonnet-4-6", effort: strPtr("none"), wantType: "disabled", wantMax: 1024},
		{name: "nil effort absent", model: "claude-sonnet-4-6", wantAbsent: true, wantMax: 1024},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got anthropicCaptureRequest
			srv, p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
				_ = json.NewDecoder(r.Body).Decode(&got)
				_, _ = w.Write([]byte(`{"id":"m","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
			})
			defer srv.Close()

			_, err := p.Complete(context.Background(), chat.ChatRequest{
				Model:           tt.model,
				ReasoningEffort: tt.effort,
				Messages:        []chat.Message{{Role: chat.RoleUser, Content: "x"}},
			})
			if err != nil {
				t.Fatalf("Complete: %v", err)
			}
			if tt.wantAbsent {
				if got.Thinking != nil {
					t.Errorf("thinking = %+v, want absent", got.Thinking)
				}
			} else {
				if got.Thinking == nil {
					t.Fatal("thinking should be present")
				}
				if got.Thinking.Type != tt.wantType {
					t.Errorf("thinking.type = %q, want %q", got.Thinking.Type, tt.wantType)
				}
				if got.Thinking.BudgetTokens != tt.wantBudget {
					t.Errorf("thinking.budget_tokens = %d, want %d", got.Thinking.BudgetTokens, tt.wantBudget)
				}
			}
			if got.MaxTokens != tt.wantMax {
				t.Errorf("max_tokens = %d, want %d", got.MaxTokens, tt.wantMax)
			}
		})
	}
}

func strPtr(s string) *string { return &s }

func TestAnthropicComplete_ExplicitMaxTokens(t *testing.T) {
	mt := 4096
	var got *anthropicCaptureRequest
	srv, p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&got)
		_, _ = w.Write([]byte(`{"id":"m","content":[{"type":"text","text":"ok"}],"stop_reason":"max_tokens","usage":{"input_tokens":1,"output_tokens":2}}`))
	})
	defer srv.Close()

	resp, err := p.Complete(context.Background(), chat.ChatRequest{
		Model:     "claude-haiku-4-5",
		MaxTokens: &mt,
		Messages:  []chat.Message{{Role: chat.RoleUser, Content: "x"}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if got.MaxTokens != 4096 {
		t.Errorf("max_tokens = %d, want 4096", got.MaxTokens)
	}
	if resp.FinishReason != "length" {
		t.Errorf("finish_reason = %q, want length", resp.FinishReason)
	}
}

func TestAnthropicComplete_NoSystemNoTemp(t *testing.T) {
	var got *anthropicCaptureRequest
	srv, p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&got)
		_, _ = w.Write([]byte(`{"id":"m","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	})
	defer srv.Close()

	_, err := p.Complete(context.Background(), chat.ChatRequest{
		Model:    "claude-haiku-4-5",
		Messages: []chat.Message{{Role: chat.RoleUser, Content: "x"}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if got.System != "" {
		t.Errorf("system = %q, want empty", got.System)
	}
	if got.Temperature != nil {
		t.Errorf("temperature = %v, want nil", got.Temperature)
	}
}

func TestAnthropicStream_TextDeltaAndFinish(t *testing.T) {
	stream := strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"msg_s","type":"message","role":"assistant","model":"claude-sonnet-4-6","content":[],"stop_reason":null,"usage":{"input_tokens":25,"output_tokens":1}}}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"!"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`event: message_delta`,
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":15}}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")

	var got *anthropicCaptureRequest
	srv, p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Accept") != "text/event-stream" {
			t.Errorf("Accept = %q", r.Header.Get("Accept"))
		}
		_ = json.NewDecoder(r.Body).Decode(&got)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(stream))
	})
	defer srv.Close()

	var deltas []chat.StreamDelta
	err := p.Stream(context.Background(), chat.ChatRequest{
		Model:    "claude-sonnet-4-6",
		Messages: []chat.Message{{Role: chat.RoleUser, Content: "Hi"}},
	}, func(d chat.StreamDelta) error {
		deltas = append(deltas, d)
		return nil
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}

	if !got.Stream {
		t.Error("stream should be true in request body")
	}

	if len(deltas) != 3 {
		t.Fatalf("deltas len = %d, want 3", len(deltas))
	}
	if deltas[0].Delta != "Hello" || deltas[1].Delta != "!" {
		t.Errorf("text deltas = %q, %q", deltas[0].Delta, deltas[1].Delta)
	}
	last := deltas[2]
	if last.FinishReason != "stop" {
		t.Errorf("finish_reason = %q, want stop", last.FinishReason)
	}
	if last.Usage == nil {
		t.Fatal("usage should be set on final delta")
	}
	if last.Usage.PromptTokens != 25 || last.Usage.CompletionTokens != 15 || last.Usage.TotalTokens != 40 {
		t.Errorf("usage = %+v", last.Usage)
	}
}

func TestAnthropicStream_ErrorAfterText(t *testing.T) {
	stream := strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"usage":{"input_tokens":5,"output_tokens":1}}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}`,
		``,
		`event: error`,
		`data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`,
		``,
	}, "\n")

	srv, p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(stream))
	})
	defer srv.Close()

	var deltas []chat.StreamDelta
	err := p.Stream(context.Background(), chat.ChatRequest{
		Model:    "claude-sonnet-4-6",
		Messages: []chat.Message{{Role: chat.RoleUser, Content: "Hi"}},
	}, func(d chat.StreamDelta) error {
		deltas = append(deltas, d)
		return nil
	})
	if err != nil {
		t.Fatalf("Stream should return nil after text delivered, got %v", err)
	}
	if len(deltas) != 2 {
		t.Fatalf("deltas len = %d, want 2", len(deltas))
	}
	if deltas[0].Delta != "partial" {
		t.Errorf("delta[0] = %q", deltas[0].Delta)
	}
	if deltas[1].FinishReason != "error" {
		t.Errorf("finish_reason = %q, want error", deltas[1].FinishReason)
	}
}

func TestAnthropicStream_ErrorBeforeText(t *testing.T) {
	stream := strings.Join([]string{
		`event: error`,
		`data: {"type":"error","error":{"type":"invalid_request_error","message":"bad"}}`,
		``,
	}, "\n")

	srv, p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(stream))
	})
	defer srv.Close()

	err := p.Stream(context.Background(), chat.ChatRequest{
		Model:    "claude-sonnet-4-6",
		Messages: []chat.Message{{Role: chat.RoleUser, Content: "Hi"}},
	}, func(d chat.StreamDelta) error { return nil })
	if err == nil {
		t.Fatal("expected error before any text delivered")
	}
}

func TestAnthropicStream_TransportError(t *testing.T) {
	srv, p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"type":"error","error":{"type":"authentication_error","message":"bad key"}}`, http.StatusUnauthorized)
	})
	defer srv.Close()

	err := p.Stream(context.Background(), chat.ChatRequest{
		Model:    "claude-sonnet-4-6",
		Messages: []chat.Message{{Role: chat.RoleUser, Content: "Hi"}},
	}, func(d chat.StreamDelta) error { return nil })
	if err == nil {
		t.Fatal("expected transport error")
	}
	if !strings.Contains(err.Error(), "401") {
		t.Errorf("error should mention status, got %v", err)
	}
}

func TestAnthropicName(t *testing.T) {
	_, p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {})
	if p.Name() != "anthropic" {
		t.Errorf("Name() = %q", p.Name())
	}
}

func TestAnthropicComplete_ToolsSerialization(t *testing.T) {
	var got anthropicCaptureRequest
	srv, p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&got)
		_, _ = w.Write([]byte(`{"id":"m","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
	})
	defer srv.Close()

	_, err := p.Complete(context.Background(), chat.ChatRequest{
		Model:    "claude-sonnet-4-6",
		Messages: []chat.Message{{Role: chat.RoleUser, Content: "x"}},
		Tools: []chat.Tool{
			{
				Name:        "get_weather",
				Description: "Get current weather for a location.",
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"location": map[string]any{"type": "string"},
					},
					"required": []any{"location"},
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
	if tool.Name != "get_weather" {
		t.Errorf("tool name = %q", tool.Name)
	}
	if tool.Description != "Get current weather for a location." {
		t.Errorf("tool description = %q", tool.Description)
	}
	if tool.InputSchema == nil {
		t.Fatal("input_schema should be present")
	}
	if tool.InputSchema["type"] != "object" {
		t.Errorf("input_schema.type = %v, want object", tool.InputSchema["type"])
	}
}

func TestAnthropicComplete_ToolUseResponse(t *testing.T) {
	srv, p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"msg_tool","type":"message","role":"assistant","model":"claude-sonnet-4-6",
			"content":[
				{"type":"text","text":"I will look up the weather."},
				{"type":"tool_use","id":"toolu_01Abc","name":"get_weather","input":{"location":"Jakarta"}}
			],
			"stop_reason":"tool_use","stop_sequence":null,
			"usage":{"input_tokens":20,"output_tokens":10}
		}`))
	})
	defer srv.Close()

	resp, err := p.Complete(context.Background(), chat.ChatRequest{
		Model:    "claude-sonnet-4-6",
		Messages: []chat.Message{{Role: chat.RoleUser, Content: "weather?"}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.Content != "I will look up the weather." {
		t.Errorf("content = %q", resp.Content)
	}
	if resp.FinishReason != "tool_calls" {
		t.Errorf("finish_reason = %q, want tool_calls", resp.FinishReason)
	}
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("tool_calls len = %d, want 1", len(resp.ToolCalls))
	}
	tc := resp.ToolCalls[0]
	if tc.ID != "toolu_01Abc" {
		t.Errorf("tool call id = %q", tc.ID)
	}
	if tc.Name != "get_weather" {
		t.Errorf("tool call name = %q", tc.Name)
	}
	var args map[string]any
	if err := json.Unmarshal(tc.Arguments, &args); err != nil {
		t.Fatalf("unmarshal arguments: %v", err)
	}
	if args["location"] != "Jakarta" {
		t.Errorf("arguments = %v", args)
	}
}

func TestAnthropicStream_ToolUse(t *testing.T) {
	stream := strings.Join([]string{
		`event: message_start`,
		`data: {"type":"message_start","message":{"id":"msg_t","type":"message","role":"assistant","model":"claude-sonnet-4-6","content":[],"stop_reason":null,"usage":{"input_tokens":25,"output_tokens":1}}}`,
		``,
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_01Abc","name":"get_weather"}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"location\":"}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"Jakarta\"}"}}`,
		``,
		`event: content_block_stop`,
		`data: {"type":"content_block_stop","index":0}`,
		``,
		`event: message_delta`,
		`data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":15}}`,
		``,
		`event: message_stop`,
		`data: {"type":"message_stop"}`,
		``,
	}, "\n")

	srv, p := newTestProvider(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(stream))
	})
	defer srv.Close()

	var deltas []chat.StreamDelta
	err := p.Stream(context.Background(), chat.ChatRequest{
		Model:    "claude-sonnet-4-6",
		Messages: []chat.Message{{Role: chat.RoleUser, Content: "weather?"}},
	}, func(d chat.StreamDelta) error {
		deltas = append(deltas, d)
		return nil
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if len(deltas) != 1 {
		t.Fatalf("deltas len = %d, want 1", len(deltas))
	}
	last := deltas[0]
	if last.FinishReason != "tool_calls" {
		t.Errorf("finish_reason = %q, want tool_calls", last.FinishReason)
	}
	if len(last.ToolCalls) != 1 {
		t.Fatalf("tool_calls len = %d, want 1", len(last.ToolCalls))
	}
	tc := last.ToolCalls[0]
	if tc.ID != "toolu_01Abc" {
		t.Errorf("tool call id = %q", tc.ID)
	}
	if tc.Name != "get_weather" {
		t.Errorf("tool call name = %q", tc.Name)
	}
	var args map[string]any
	if err := json.Unmarshal(tc.Arguments, &args); err != nil {
		t.Fatalf("unmarshal arguments: %v", err)
	}
	if args["location"] != "Jakarta" {
		t.Errorf("arguments = %v", args)
	}
}
