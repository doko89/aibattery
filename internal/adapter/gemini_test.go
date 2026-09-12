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

// geminiCapture records what a fake Gemini upstream received.
type geminiCapture struct {
	path        string
	auth        string
	contentType string
	bodies      []geminiRequest
}

// newGeminiCaptureServer starts a fake Gemini upstream that records every
// request into cap and replies with respBody.
func newGeminiCaptureServer(t *testing.T, cap *geminiCapture, respBody string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cap.path = r.URL.Path
		cap.auth = r.Header.Get("x-goog-api-key")
		cap.contentType = r.Header.Get("Content-Type")
		b, _ := io.ReadAll(r.Body)
		var body geminiRequest
		if err := json.Unmarshal(b, &body); err != nil {
			t.Errorf("decode request body: %v", err)
		}
		cap.bodies = append(cap.bodies, body)
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, respBody)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func assertGeminiEndpoint(t *testing.T, cap *geminiCapture, wantPath, wantAuth string) {
	t.Helper()
	if cap.path != wantPath {
		t.Errorf("path = %q", cap.path)
	}
	if cap.auth != wantAuth {
		t.Errorf("auth header = %q", cap.auth)
	}
	if cap.contentType != "application/json" {
		t.Errorf("content-type = %q", cap.contentType)
	}
}

func assertGeminiContentsMapping(t *testing.T, body geminiRequest) {
	t.Helper()
	if len(body.Contents) != 3 {
		t.Fatalf("contents len = %d", len(body.Contents))
	}
	want := []struct {
		role string
		text string
	}{
		{"user", "hi"},
		{"model", "hello"},
		{"user", ""},
	}
	for i, w := range want {
		if body.Contents[i].Role != w.role || body.Contents[i].Parts[0].Text != w.text {
			t.Errorf("contents[%d] = %+v", i, body.Contents[i])
		}
	}
}

func assertGeminiSystemInstruction(t *testing.T, body geminiRequest, want string) {
	t.Helper()
	if body.SystemInstruction == nil || body.SystemInstruction.Parts[0].Text != want {
		t.Errorf("systemInstruction = %+v", body.SystemInstruction)
	}
}

func assertGeminiTempAndMaxTokens(t *testing.T, body geminiRequest, wantTemp float64, wantMax int) {
	t.Helper()
	if body.GenerationConfig == nil {
		t.Fatal("generationConfig missing")
	}
	if body.GenerationConfig.Temperature == nil || *body.GenerationConfig.Temperature != wantTemp {
		t.Errorf("temperature = %v", body.GenerationConfig.Temperature)
	}
	if body.GenerationConfig.MaxOutputTokens == nil || *body.GenerationConfig.MaxOutputTokens != wantMax {
		t.Errorf("maxOutputTokens = %v", body.GenerationConfig.MaxOutputTokens)
	}
}

func assertGeminiResponseParsing(t *testing.T, resp chat.ChatResponse) {
	t.Helper()
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

// TestGeminiComplete_RequestMappingAndParsing verifies the non-streaming
// request-body translation (contents, systemInstruction, generationConfig) and
// the response parts[].text parsing.
func TestGeminiComplete_RequestMappingAndParsing(t *testing.T) {
	var cap geminiCapture
	srv := newGeminiCaptureServer(t, &cap, `{
		"candidates":[{"content":{"role":"model","parts":[{"text":"Hello"},{"text":" world"}]},"finishReason":"STOP"}],
		"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":20,"totalTokenCount":30},
		"responseId":"resp-1"
	}`)
	p := NewGeminiProvider(srv.URL, "test-key", 5*time.Second)

	temp := 0.7
	maxTokens := 100
	resp, err := p.Complete(context.Background(), chat.ChatRequest{
		Model: "models/gemini-2.5-flash",
		Messages: []chat.Message{
			{Role: chat.RoleUser, Content: "hi"},
			{Role: chat.RoleAssistant, Content: "hello"},
			{Role: chat.RoleUser, Content: ""},
		},
		System:      "be concise",
		Temperature: &temp,
		MaxTokens:   &maxTokens,
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}

	assertGeminiEndpoint(t, &cap, "/models/gemini-2.5-flash:generateContent", "test-key")
	assertGeminiContentsMapping(t, cap.bodies[0])
	assertGeminiSystemInstruction(t, cap.bodies[0], "be concise")
	assertGeminiTempAndMaxTokens(t, cap.bodies[0], 0.7, 100)
	assertGeminiResponseParsing(t, resp)
}

// TestGeminiComplete_ThinkingLevelGemini3 verifies the Gemini 3 family maps
// effort to thinkingLevel and forces temperature to 1.0.
func TestGeminiComplete_ThinkingLevelGemini3(t *testing.T) {
	var cap geminiCapture
	srv := newGeminiCaptureServer(t, &cap, `{
		"candidates":[{"content":{"role":"model","parts":[{"text":"Hello"},{"text":" world"}]},"finishReason":"STOP"}],
		"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":20,"totalTokenCount":30},
		"responseId":"resp-1"
	}`)
	p := NewGeminiProvider(srv.URL, "test-key", 5*time.Second)

	temp := 0.7 // must be overridden to 1.0
	effort := "high"
	if _, err := p.Complete(context.Background(), chat.ChatRequest{
		Model:           "gemini-3-pro",
		Messages:        []chat.Message{{Role: chat.RoleUser, Content: "hi"}},
		Temperature:     &temp,
		ReasoningEffort: &effort,
	}); err != nil {
		t.Fatalf("Complete (gemini-3-pro): %v", err)
	}
	assertGeminiThinkingLevel(t, cap.bodies[0].GenerationConfig, "high")
	assertGeminiForcedTemperature(t, cap.bodies[0].GenerationConfig, 1.0)
}

func assertGeminiThinkingLevel(t *testing.T, cfg *geminiGenConfig, want string) {
	t.Helper()
	if cfg == nil || cfg.ThinkingConfig == nil || cfg.ThinkingConfig.ThinkingLevel != want {
		t.Errorf("gemini-3-pro thinkingConfig = %+v", cfg)
	}
}

func assertGeminiForcedTemperature(t *testing.T, cfg *geminiGenConfig, want float64) {
	t.Helper()
	if cfg == nil {
		t.Errorf("gemini-3-pro temperature = <nil config>, want 1.0")
		return
	}
	if cfg.Temperature == nil || *cfg.Temperature != want {
		t.Errorf("gemini-3-pro temperature = %v, want 1.0", cfg.Temperature)
	}
}

// TestGeminiComplete_ThinkingBudget25 verifies the 2.5 family maps effort to
// a token budget with no thinkingLevel.
func TestGeminiComplete_ThinkingBudget25(t *testing.T) {
	var cap geminiCapture
	srv := newGeminiCaptureServer(t, &cap, `{
		"candidates":[{"content":{"role":"model","parts":[{"text":"Hello"},{"text":" world"}]},"finishReason":"STOP"}],
		"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":20,"totalTokenCount":30},
		"responseId":"resp-1"
	}`)
	p := NewGeminiProvider(srv.URL, "test-key", 5*time.Second)

	effort := "medium"
	if _, err := p.Complete(context.Background(), chat.ChatRequest{
		Model:           "gemini-2.5-flash",
		Messages:        []chat.Message{{Role: chat.RoleUser, Content: "hi"}},
		ReasoningEffort: &effort,
	}); err != nil {
		t.Fatalf("Complete (gemini-2.5-flash): %v", err)
	}
	assertGeminiThinkingBudget(t, cap.bodies[0].GenerationConfig, 4096)
}

func assertGeminiThinkingBudget(t *testing.T, cfg *geminiGenConfig, want int) {
	t.Helper()
	if cfg == nil || cfg.ThinkingConfig == nil || cfg.ThinkingConfig.ThinkingBudget != want {
		t.Errorf("gemini-2.5-flash thinkingConfig = %+v", cfg)
	}
	checkGeminiNoThinkingLevel(t, cfg)
}

func checkGeminiNoThinkingLevel(t *testing.T, cfg *geminiGenConfig) {
	t.Helper()
	if cfg == nil {
		return
	}
	if cfg.ThinkingConfig != nil && cfg.ThinkingConfig.ThinkingLevel != "" {
		t.Errorf("gemini-2.5-flash thinkingLevel = %q, want empty", cfg.ThinkingConfig.ThinkingLevel)
	}
}

// TestGeminiComplete_RateLimit429 verifies a 429 upstream response surfaces as
// a typed chat.RateLimitError so the API layer can apply cooldown.
func TestGeminiComplete_RateLimit429(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		io.WriteString(w, `{"error":{"code":429,"message":"rate limited","status":"RESOURCE_EXHAUSTED"}}`)
	}))
	defer srv.Close()

	p := NewGeminiProvider(srv.URL, "test-key", 5*time.Second)
	_, err := p.Complete(context.Background(), chat.ChatRequest{
		Model:    "gemini-2.5-flash",
		Messages: []chat.Message{{Role: chat.RoleUser, Content: "hi"}},
	})
	if err == nil {
		t.Fatal("expected error for 429")
	}
	if !chat.IsRateLimit(err) {
		t.Fatalf("error = %v, want IsRateLimit to be true", err)
	}
	if !strings.Contains(err.Error(), "status 429") {
		t.Errorf("error = %q, want to contain status 429", err)
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

// TestGemini_ToolRoundTripBody verifies that a multi-turn tool conversation is
// serialized with assistant functionCall parts and user-role functionResponse
// parts carrying matching ids.
func TestGemini_ToolRoundTripBody(t *testing.T) {
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
		Model: "gemini-2.5-flash",
		Messages: []chat.Message{
			{Role: chat.RoleUser, Content: "hi"},
			{
				Role:    chat.RoleAssistant,
				Content: "searching",
				ToolCalls: []chat.ToolCall{{
					ID:        "call_1",
					Name:      "search_web",
					Arguments: json.RawMessage(`{"query":"x"}`),
				}},
			},
			{Role: chat.Role("tool"), ToolCallID: "call_1", Content: "result"},
		},
	}); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	if len(gotBody.Contents) != 3 {
		t.Fatalf("contents len = %d", len(gotBody.Contents))
	}

	// Assistant message: text part + functionCall part with parsed args and id.
	model := gotBody.Contents[1]
	if model.Role != "model" {
		t.Errorf("assistant role = %q, want model", model.Role)
	}
	if len(model.Parts) != 2 {
		t.Fatalf("assistant parts len = %d, want 2 (text + functionCall)", len(model.Parts))
	}
	if model.Parts[0].Text != "searching" {
		t.Errorf("assistant text part = %q", model.Parts[0].Text)
	}
	fc := model.Parts[1].FunctionCall
	if fc == nil {
		t.Fatal("assistant parts[1] is not a functionCall")
	}
	if fc.Name != "search_web" || fc.ID != "call_1" {
		t.Errorf("functionCall = %+v, want name search_web id call_1", fc)
	}
	if args := fc.Arguments; len(args) != 1 || args["query"] != "x" {
		t.Errorf("functionCall args = %+v", fc.Arguments)
	}

	// Tool-result message: user role + functionResponse part matching the id.
	tool := gotBody.Contents[2]
	if tool.Role != "user" {
		t.Errorf("tool-result role = %q, want user", tool.Role)
	}
	if len(tool.Parts) != 1 {
		t.Fatalf("tool-result parts len = %d", len(tool.Parts))
	}
	fr := tool.Parts[0].FunctionResponse
	if fr == nil {
		t.Fatal("tool-result parts[0] is not a functionResponse")
	}
	if fr.ID != "call_1" {
		t.Errorf("functionResponse id = %q", fr.ID)
	}
	if fr.Name != "search_web" {
		t.Errorf("functionResponse name = %q, want search_web (matched from assistant ToolCall.Name)", fr.Name)
	}
	if len(fr.Response) == 0 {
		t.Errorf("functionResponse response = %+v, want non-empty", fr.Response)
	}
}

// TestGemini_ToolResultJSONResponse verifies that a tool result whose content
// is a JSON object is used verbatim as the functionResponse object.
func TestGemini_ToolResultJSONResponse(t *testing.T) {
	var gotBody geminiRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, &gotBody)
		io.WriteString(w, `{"candidates":[{"content":{"parts":[{"text":"ok"}]},"finishReason":"STOP"}]}`)
	}))
	defer srv.Close()

	p := NewGeminiProvider(srv.URL, "k", 5*time.Second)
	if _, err := p.Complete(context.Background(), chat.ChatRequest{
		Model: "gemini-2.5-flash",
		Messages: []chat.Message{
			{
				Role: chat.RoleAssistant,
				ToolCalls: []chat.ToolCall{{
					ID:        "c2",
					Name:      "get_weather",
					Arguments: json.RawMessage(`{}`),
				}},
			},
			{Role: chat.Role("tool"), ToolCallID: "c2", Content: `{"temp":22,"unit":"C"}`},
		},
	}); err != nil {
		t.Fatalf("Complete: %v", err)
	}

	fr := gotBody.Contents[1].Parts[0].FunctionResponse
	if fr == nil {
		t.Fatal("missing functionResponse part")
	}
	if fr.Name != "get_weather" {
		t.Errorf("name = %q", fr.Name)
	}
	if len(fr.Response) != 2 || fr.Response["temp"] != float64(22) {
		t.Errorf("response = %+v, want parsed JSON object", fr.Response)
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
