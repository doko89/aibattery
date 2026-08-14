package adapter

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/aibattery/router/internal/chat"
)

// TestGeminiComplete_RequestMappingAndParsing verifies the non-streaming
// request-body translation (contents, systemInstruction, generationConfig) and
// the response parts[].text parsing.
func TestGeminiComplete_RequestMappingAndParsing(t *testing.T) {
	var gotPath, gotAuth, gotContentType string
	var gotBodies []geminiRequest

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAuth = r.Header.Get("x-goog-api-key")
		gotContentType = r.Header.Get("Content-Type")
		b, _ := io.ReadAll(r.Body)
		var body geminiRequest
		if err := json.Unmarshal(b, &body); err != nil {
			t.Fatalf("decode request body: %v", err)
		}
		gotBodies = append(gotBodies, body)
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{
			"candidates":[{"content":{"role":"model","parts":[{"text":"Hello"},{"text":" world"}]},"finishReason":"STOP"}],
			"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":20,"totalTokenCount":30},
			"responseId":"resp-1"
		}`)
	}))
	defer srv.Close()

	p := NewGeminiProvider(srv.URL, "test-key", 5*time.Second)

	temp := 0.7
	max := 100
	resp, err := p.Complete(context.Background(), chat.ChatRequest{
		Model: "models/gemini-2.5-flash",
		Messages: []chat.Message{
			{Role: chat.RoleUser, Content: "hi"},
			{Role: chat.RoleAssistant, Content: "hello"},
			{Role: chat.RoleUser, Content: ""},
		},
		System:      "be concise",
		Temperature: &temp,
		MaxTokens:   &max,
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}

	// Endpoint: leading "models/" stripped, :generateContent suffix.
	if gotPath != "/models/gemini-2.5-flash:generateContent" {
		t.Errorf("path = %q", gotPath)
	}
	if gotAuth != "test-key" {
		t.Errorf("auth header = %q", gotAuth)
	}
	if gotContentType != "application/json" {
		t.Errorf("content-type = %q", gotContentType)
	}

	// contents mapping: user->user, assistant->model, empty->empty text part.
	if len(gotBodies[0].Contents) != 3 {
		t.Fatalf("contents len = %d", len(gotBodies[0].Contents))
	}
	if gotBodies[0].Contents[0].Role != "user" || gotBodies[0].Contents[0].Parts[0].Text != "hi" {
		t.Errorf("contents[0] = %+v", gotBodies[0].Contents[0])
	}
	if gotBodies[0].Contents[1].Role != "model" || gotBodies[0].Contents[1].Parts[0].Text != "hello" {
		t.Errorf("contents[1] = %+v", gotBodies[0].Contents[1])
	}
	if gotBodies[0].Contents[2].Role != "user" || gotBodies[0].Contents[2].Parts[0].Text != "" {
		t.Errorf("contents[2] = %+v", gotBodies[0].Contents[2])
	}

	// systemInstruction only when System set.
	if gotBodies[0].SystemInstruction == nil || gotBodies[0].SystemInstruction.Parts[0].Text != "be concise" {
		t.Errorf("systemInstruction = %+v", gotBodies[0].SystemInstruction)
	}

	// generationConfig only when set.
	if gotBodies[0].GenerationConfig == nil {
		t.Fatal("generationConfig missing")
	}
	if gotBodies[0].GenerationConfig.Temperature == nil || *gotBodies[0].GenerationConfig.Temperature != 0.7 {
		t.Errorf("temperature = %v", gotBodies[0].GenerationConfig.Temperature)
	}
	if gotBodies[0].GenerationConfig.MaxOutputTokens == nil || *gotBodies[0].GenerationConfig.MaxOutputTokens != 100 {
		t.Errorf("maxOutputTokens = %v", gotBodies[0].GenerationConfig.MaxOutputTokens)
	}

	// thinkingConfig: Gemini 3 family maps effort to thinkingLevel and forces
	// temperature to 1.0.
	effort := "high"
	if _, err := p.Complete(context.Background(), chat.ChatRequest{
		Model:           "gemini-3-pro",
		Messages:        []chat.Message{{Role: chat.RoleUser, Content: "hi"}},
		Temperature:     &temp, // 0.7, must be overridden to 1.0
		ReasoningEffort: &effort,
	}); err != nil {
		t.Fatalf("Complete (gemini-3-pro): %v", err)
	}
	g3 := gotBodies[1].GenerationConfig
	if g3 == nil || g3.ThinkingConfig == nil || g3.ThinkingConfig.ThinkingLevel != "high" {
		t.Errorf("gemini-3-pro thinkingConfig = %+v", g3)
	}
	if g3.Temperature == nil || *g3.Temperature != 1.0 {
		t.Errorf("gemini-3-pro temperature = %v, want 1.0", g3.Temperature)
	}

	// thinkingConfig: 2.5 family maps effort to a token budget, no thinkingLevel.
	effort2 := "medium"
	if _, err := p.Complete(context.Background(), chat.ChatRequest{
		Model:           "gemini-2.5-flash",
		Messages:        []chat.Message{{Role: chat.RoleUser, Content: "hi"}},
		ReasoningEffort: &effort2,
	}); err != nil {
		t.Fatalf("Complete (gemini-2.5-flash): %v", err)
	}
	g25 := gotBodies[2].GenerationConfig
	if g25 == nil || g25.ThinkingConfig == nil || g25.ThinkingConfig.ThinkingBudget != 4096 {
		t.Errorf("gemini-2.5-flash thinkingConfig = %+v", g25)
	}
	if g25.ThinkingConfig != nil && g25.ThinkingConfig.ThinkingLevel != "" {
		t.Errorf("gemini-2.5-flash thinkingLevel = %q, want empty", g25.ThinkingConfig.ThinkingLevel)
	}

	// Response parsing: parts joined, finishReason mapped, usage, id.
	if resp.Content != "Hello world" {
		t.Errorf("content = %q", resp.Content)
	}
	if resp.FinishReason != "stop" {
		t.Errorf("finishReason = %q", resp.FinishReason)
	}
	if resp.ID != "resp-1" {
		t.Errorf("id = %q", resp.ID)
	}
	if resp.Usage.PromptTokens != 10 || resp.Usage.CompletionTokens != 20 || resp.Usage.TotalTokens != 30 {
		t.Errorf("usage = %+v", resp.Usage)
	}
}

// TestGeminiComplete_OmitsOptionals verifies systemInstruction and
// generationConfig are omitted when unset.
func TestGeminiComplete_OmitsOptionals(t *testing.T) {
	var gotBody geminiRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &gotBody)
		io.WriteString(w, `{"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`)
	}))
	defer srv.Close()

	p := NewGeminiProvider(srv.URL, "k", 5*time.Second)
	if _, err := p.Complete(context.Background(), chat.ChatRequest{
		Model:    "gemini-2.5-flash",
		Messages: []chat.Message{{Role: chat.RoleUser, Content: "hi"}},
	}); err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if gotBody.SystemInstruction != nil {
		t.Errorf("systemInstruction should be nil, got %+v", gotBody.SystemInstruction)
	}
	if gotBody.GenerationConfig != nil {
		t.Errorf("generationConfig should be nil, got %+v", gotBody.GenerationConfig)
	}
}

// TestGeminiStream_ChunkParsing verifies SSE chunk parsing, text accumulation,
// finishReason mapping, usage capture, and the final delta.
func TestGeminiStream_ChunkParsing(t *testing.T) {
	var gotPath, gotQuery string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotQuery = r.URL.RawQuery
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hel\"}]}}]}\n\n")
		io.WriteString(w, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"lo\"}]}}]}\n\n")
		io.WriteString(w, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\" world\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":5,\"candidatesTokenCount\":7,\"totalTokenCount\":12}}\n\n")
	}))
	defer srv.Close()

	p := NewGeminiProvider(srv.URL, "k", 5*time.Second)

	var deltas []chat.StreamDelta
	err := p.Stream(context.Background(), chat.ChatRequest{
		Model:    "gemini-2.5-flash",
		Messages: []chat.Message{{Role: chat.RoleUser, Content: "hi"}},
	}, func(d chat.StreamDelta) error {
		deltas = append(deltas, d)
		return nil
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}

	if gotPath != "/models/gemini-2.5-flash:streamGenerateContent" {
		t.Errorf("path = %q", gotPath)
	}
	if gotQuery != "alt=sse" {
		t.Errorf("query = %q", gotQuery)
	}

	// 3 text deltas + 1 final delta.
	if len(deltas) != 4 {
		t.Fatalf("deltas len = %d: %+v", len(deltas), deltas)
	}
	if deltas[0].Delta != "Hel" || deltas[1].Delta != "lo" || deltas[2].Delta != " world" {
		t.Errorf("text deltas = %+v", deltas[:3])
	}
	final := deltas[3]
	if final.FinishReason != "stop" {
		t.Errorf("final finishReason = %q", final.FinishReason)
	}
	if final.Usage == nil || final.Usage.PromptTokens != 5 || final.Usage.CompletionTokens != 7 || final.Usage.TotalTokens != 12 {
		t.Errorf("final usage = %+v", final.Usage)
	}
}

// TestGeminiStream_ErrorAfterFirstDelta verifies that a transport error after
// content is delivered surfaces as a FinishReason "error" delta and Stream
// returns nil.
func TestGeminiStream_ErrorAfterFirstDelta(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"partial\"}]}}]}\n\n")
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		// Abruptly close the underlying connection mid-stream to simulate a
		// transport error after the first delta.
		if hj, ok := w.(http.Hijacker); ok {
			conn, _, err := hj.Hijack()
			if err == nil {
				_ = conn.Close()
			}
		}
	}))
	defer srv.Close()

	p := NewGeminiProvider(srv.URL, "k", 5*time.Second)

	var deltas []chat.StreamDelta
	err := p.Stream(context.Background(), chat.ChatRequest{
		Model:    "gemini-2.5-flash",
		Messages: []chat.Message{{Role: chat.RoleUser, Content: "hi"}},
	}, func(d chat.StreamDelta) error {
		deltas = append(deltas, d)
		return nil
	})
	if err != nil {
		t.Fatalf("Stream should return nil after content, got %v", err)
	}
	if len(deltas) < 2 {
		t.Fatalf("expected at least text + error delta, got %+v", deltas)
	}
	if deltas[0].Delta != "partial" {
		t.Errorf("first delta = %+v", deltas[0])
	}
	if deltas[len(deltas)-1].FinishReason != "error" {
		t.Errorf("last delta finishReason = %q", deltas[len(deltas)-1].FinishReason)
	}
}

// TestGeminiFinishReasonMapping sanity-checks the finish_reason mapping table.
func TestGeminiFinishReasonMapping(t *testing.T) {
	cases := map[string]string{
		"STOP":       "stop",
		"MAX_TOKENS": "length",
		"SAFETY":     "content_filter",
		"RECITATION": "content_filter",
		"OTHER":      "other",
		"LANGUAGE":   "LANGUAGE", // passthrough
		"":           "",
	}
	for in, want := range cases {
		if got := geminiFinishReason(in); got != want {
			t.Errorf("geminiFinishReason(%q) = %q, want %q", in, got, want)
		}
	}
}

// TestGeminiName verifies the provider identifier.
func TestGeminiName(t *testing.T) {
	p := NewGeminiProvider("http://x", "k", time.Second)
	if p.Name() != "gemini" {
		t.Errorf("Name() = %q", p.Name())
	}
}

// TestGeminiEndpoint verifies model prefix stripping and trailing-slash
// handling on the base URL.
func TestGeminiEndpoint(t *testing.T) {
	p := &geminiProvider{baseURL: "https://example.com/"}
	if got := p.endpoint("models/gemini-2.5-flash", false); got != "https://example.com/models/gemini-2.5-flash:generateContent" {
		t.Errorf("non-stream endpoint = %q", got)
	}
	if got := p.endpoint("gemini-2.5-flash", true); got != "https://example.com/models/gemini-2.5-flash:streamGenerateContent?alt=sse" {
		t.Errorf("stream endpoint = %q", got)
	}
}

// TestGeminiStream_EmptyTextParts verifies empty text parts are skipped and a
// final delta is still emitted.
func TestGeminiStream_EmptyTextParts(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"\"}]}}]}\n\n")
		io.WriteString(w, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"\"}]},\"finishReason\":\"MAX_TOKENS\"}]}\n\n")
	}))
	defer srv.Close()

	p := NewGeminiProvider(srv.URL, "k", 5*time.Second)
	var deltas []chat.StreamDelta
	if err := p.Stream(context.Background(), chat.ChatRequest{
		Model:    "gemini-2.5-flash",
		Messages: []chat.Message{{Role: chat.RoleUser, Content: "hi"}},
	}, func(d chat.StreamDelta) error {
		deltas = append(deltas, d)
		return nil
	}); err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if len(deltas) != 1 {
		t.Fatalf("expected only final delta, got %+v", deltas)
	}
	if deltas[0].FinishReason != "length" {
		t.Errorf("final finishReason = %q", deltas[0].FinishReason)
	}
	if strings.TrimSpace(deltas[0].Delta) != "" {
		t.Errorf("final delta should carry no text, got %q", deltas[0].Delta)
	}
}

// TestGeminiRequest_ToolsSerialization verifies that a non-empty Tools list is
// serialized into the Gemini tools->functionDeclarations shape with
// name/description/parameters, and that existing fields are preserved.
func TestGeminiRequest_ToolsSerialization(t *testing.T) {
	var gotBody geminiRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(b, &gotBody); err != nil {
			t.Fatalf("decode request body: %v", err)
		}
		io.WriteString(w, `{"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`)
	}))
	defer srv.Close()

	p := NewGeminiProvider(srv.URL, "k", 5*time.Second)
	if _, err := p.Complete(context.Background(), chat.ChatRequest{
		Model:    "gemini-2.5-flash",
		Messages: []chat.Message{{Role: chat.RoleUser, Content: "hi"}},
		System:   "be concise",
		Tools: []chat.Tool{
			{
				Name:        "get_weather",
				Description: "Get current weather",
				InputSchema: map[string]any{
					"type":       "object",
					"properties": map[string]any{"location": map[string]any{"type": "string"}},
					"required":   []any{"location"},
				},
			},
		},
	}); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	if len(gotBody.Tools) != 1 {
		t.Fatalf("tools len = %d", len(gotBody.Tools))
	}
	fds := gotBody.Tools[0].FunctionDeclarations
	if len(fds) != 1 {
		t.Fatalf("functionDeclarations len = %d", len(fds))
	}
	fd := fds[0]
	if fd.Name != "get_weather" {
		t.Errorf("name = %q", fd.Name)
	}
	if fd.Description != "Get current weather" {
		t.Errorf("description = %q", fd.Description)
	}
	if fd.Parameters == nil || fd.Parameters["type"] != "object" {
		t.Errorf("parameters = %+v", fd.Parameters)
	}
	// Existing behavior preserved.
	if gotBody.SystemInstruction == nil || gotBody.SystemInstruction.Parts[0].Text != "be concise" {
		t.Errorf("systemInstruction = %+v", gotBody.SystemInstruction)
	}
}

// TestGeminiComplete_FunctionCallParsing verifies that a non-streaming response
// carrying functionCall parts maps to ChatResponse.ToolCalls and forces
// FinishReason to "tool_calls".
func TestGeminiComplete_FunctionCallParsing(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{
			"candidates":[{"content":{"role":"model","parts":[
				{"text":"Let me check"},
				{"functionCall":{"name":"get_weather","args":{"location":"Boston"}}}
			]},"finishReason":"STOP"}],
			"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":3,"totalTokenCount":8},
			"responseId":"resp-fc"
		}`)
	}))
	defer srv.Close()

	p := NewGeminiProvider(srv.URL, "k", 5*time.Second)
	resp, err := p.Complete(context.Background(), chat.ChatRequest{
		Model:    "gemini-2.5-flash",
		Messages: []chat.Message{{Role: chat.RoleUser, Content: "weather in Boston?"}},
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}

	if resp.FinishReason != "tool_calls" {
		t.Errorf("finishReason = %q, want tool_calls", resp.FinishReason)
	}
	if resp.Content != "Let me check" {
		t.Errorf("content = %q", resp.Content)
	}
	if len(resp.ToolCalls) != 1 {
		t.Fatalf("toolCalls len = %d", len(resp.ToolCalls))
	}
	tc := resp.ToolCalls[0]
	if tc.ID != "" {
		t.Errorf("toolCall id = %q, want empty", tc.ID)
	}
	if tc.Name != "get_weather" {
		t.Errorf("toolCall name = %q", tc.Name)
	}
	var args map[string]any
	if err := json.Unmarshal(tc.Arguments, &args); err != nil {
		t.Fatalf("unmarshal arguments: %v", err)
	}
	if args["location"] != "Boston" {
		t.Errorf("arguments = %+v", args)
	}
}

// TestGeminiStream_FunctionCall verifies that a streaming chunk carrying a
// functionCall part yields a final StreamDelta with FinishReason "tool_calls"
// and the collected ToolCalls.
func TestGeminiStream_FunctionCall(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		io.WriteString(w, "data: {\"candidates\":[{\"content\":{\"parts\":[{\"functionCall\":{\"name\":\"get_weather\",\"args\":{\"location\":\"Jakarta\"}}}]},\"finishReason\":\"STOP\"}]}\n\n")
	}))
	defer srv.Close()

	p := NewGeminiProvider(srv.URL, "k", 5*time.Second)
	var deltas []chat.StreamDelta
	if err := p.Stream(context.Background(), chat.ChatRequest{
		Model:    "gemini-2.5-flash",
		Messages: []chat.Message{{Role: chat.RoleUser, Content: "weather?"}},
	}, func(d chat.StreamDelta) error {
		deltas = append(deltas, d)
		return nil
	}); err != nil {
		t.Fatalf("Stream: %v", err)
	}

	if len(deltas) != 1 {
		t.Fatalf("expected only final delta, got %+v", deltas)
	}
	final := deltas[0]
	if final.FinishReason != "tool_calls" {
		t.Errorf("final finishReason = %q, want tool_calls", final.FinishReason)
	}
	if len(final.ToolCalls) != 1 {
		t.Fatalf("final toolCalls = %+v", final.ToolCalls)
	}
	tc := final.ToolCalls[0]
	if tc.Name != "get_weather" {
		t.Errorf("toolCall name = %q", tc.Name)
	}
	var args map[string]any
	if err := json.Unmarshal(tc.Arguments, &args); err != nil {
		t.Fatalf("unmarshal arguments: %v", err)
	}
	if args["location"] != "Jakarta" {
		t.Errorf("arguments = %+v", args)
	}
}
