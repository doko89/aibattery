// Package adapter implements concrete chat.Provider adapters that translate
// the canonical chat domain model into each provider's native wire format.
package adapter

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/aibattery/router/internal/chat"
)

// openAIProvider is a chat.Provider that talks to any OpenAI-compatible
// /v1/chat/completions endpoint.
type openAIProvider struct {
	baseURL string
	apiKey  string
	client  *http.Client
	timeout time.Duration
}

// NewOpenAIProvider returns a chat.Provider backed by an OpenAI-compatible
// chat completions endpoint. baseURL should include the /v1 prefix (e.g.
// "https://api.openai.com/v1"); the trailing "/chat/completions" is appended.
// timeout bounds each non-streaming completion; streaming uses a fixed,
// generous budget (streamTimeout) because reasoning models can think for
// minutes before the first delta.
func NewOpenAIProvider(baseURL, apiKey string, timeout time.Duration) chat.Provider {
	return &openAIProvider{
		baseURL: strings.TrimRight(baseURL, "/"),
		apiKey:  apiKey,
		client:  &http.Client{},
		timeout: timeout,
	}
}

// Name returns the provider identifier.
func (p *openAIProvider) Name() string { return "openai" }

// streamTimeout bounds the total time a streaming request may run. Reasoning
// models can think for minutes before the first delta, so this is deliberately
// generous; http.Client.Timeout cannot be used because it is a total deadline
// that would kill healthy long streams.
// ponytail: fixed 5m streaming budget; make configurable if providers need more
const streamTimeout = 5 * time.Minute

// ---- wire types -----------------------------------------------------------

// openAIMessage is a single message in the OpenAI request body.
type openAIMessage struct {
	Role             string           `json:"role"`
	Content          string           `json:"content"`
	ToolCallID       string           `json:"tool_call_id,omitempty"`
	ToolCalls        []openAIToolCall `json:"tool_calls,omitempty"`
	ReasoningContent string           `json:"reasoning_content,omitempty"`
}

// openAIFunction is the function definition inside an openAITool.
type openAIFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

// openAITool is a single tool (function) definition offered to the model.
type openAITool struct {
	Type     string         `json:"type"`
	Function openAIFunction `json:"function"`
}

// openAIRequest is the request body sent to /chat/completions. omitempty
// ensures absent optional fields are omitted from the wire.
type openAIRequest struct {
	Model           string          `json:"model"`
	Messages        []openAIMessage `json:"messages"`
	Stream          bool            `json:"stream,omitempty"`
	Temperature     *float64        `json:"temperature,omitempty"`
	MaxTokens       *int            `json:"max_tokens,omitempty"`
	ReasoningEffort *string         `json:"reasoning_effort,omitempty"`
	Tools           []openAITool    `json:"tools,omitempty"`
}

// openAIUsage mirrors the OpenAI usage object.
type openAIUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// openAIToolCall is a single tool call in a non-streaming response message.
type openAIToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// openAIResponseMessage is the assistant message in a non-streaming response.
type openAIResponseMessage struct {
	Role             string           `json:"role"`
	Content          string           `json:"content"`
	ToolCalls        []openAIToolCall `json:"tool_calls"`
	ReasoningContent string           `json:"reasoning_content,omitempty"`
}

// openAIChoice is a single choice in a non-streaming response.
type openAIChoice struct {
	Message      openAIResponseMessage `json:"message"`
	FinishReason string                `json:"finish_reason"`
}

// openAIResponse is the non-streaming chat completions response.
type openAIResponse struct {
	ID      string         `json:"id"`
	Model   string         `json:"model"`
	Choices []openAIChoice `json:"choices"`
	Usage   openAIUsage    `json:"usage"`
}

// openAIStreamToolCall is a single tool-call delta inside a streaming chunk.
type openAIStreamToolCall struct {
	Index    int    `json:"index"`
	ID       string `json:"id"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// openAIStreamDelta is the delta object inside a streaming chunk.
type openAIStreamDelta struct {
	Content          string                 `json:"content"`
	ToolCalls        []openAIStreamToolCall `json:"tool_calls"`
	ReasoningContent string                 `json:"reasoning_content,omitempty"`
}

// openAIStreamChoice is a single choice inside a streaming chunk.
type openAIStreamChoice struct {
	Delta        openAIStreamDelta `json:"delta"`
	FinishReason string            `json:"finish_reason"`
}

// openAIStreamChunk is a single SSE data payload during streaming.
type openAIStreamChunk struct {
	ID      string               `json:"id"`
	Model   string               `json:"model"`
	Choices []openAIStreamChoice `json:"choices"`
	Usage   *openAIUsage         `json:"usage"`
}

// ---- request helpers ------------------------------------------------------

// buildRequest marshals a canonical ChatRequest into the OpenAI wire body.
func buildRequest(req chat.ChatRequest) ([]byte, error) {
	body := openAIRequest{
		Model:       req.Model,
		Messages:    openAIMessages(req.Messages),
		Stream:      req.Stream,
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
	}
	applyOpenAIReasoning(req, &body)
	if len(req.Tools) > 0 {
		body.Tools = openAITools(req.Tools)
	}
	return json.Marshal(body)
}

// openAIMessages converts canonical messages to the OpenAI wire shape.
func openAIMessages(messages []chat.Message) []openAIMessage {
	out := make([]openAIMessage, 0, len(messages))
	for _, m := range messages {
		out = append(out, openAIMessageFor(m))
	}
	return out
}

// openAIMessageFor converts one canonical message.
func openAIMessageFor(m chat.Message) openAIMessage {
	msg := openAIMessage{Role: string(m.Role), Content: m.Content}
	if m.ToolCallID != "" {
		msg.ToolCallID = m.ToolCallID
	}
	if m.ReasoningContent != "" {
		msg.ReasoningContent = m.ReasoningContent
	}
	if len(m.ToolCalls) > 0 {
		msg.ToolCalls = openAIToolCalls(m.ToolCalls)
	}
	return msg
}

// openAIToolCalls converts canonical tool calls to the OpenAI wire shape.
func openAIToolCalls(calls []chat.ToolCall) []openAIToolCall {
	out := make([]openAIToolCall, 0, len(calls))
	for _, tc := range calls {
		out = append(out, openAIToolCallFor(tc))
	}
	return out
}

// openAIToolCallFor converts one canonical tool call; empty arguments become
// "{}" so the wire never carries a missing arguments string.
func openAIToolCallFor(tc chat.ToolCall) openAIToolCall {
	args := string(tc.Arguments)
	if len(tc.Arguments) == 0 {
		args = "{}"
	}
	return openAIToolCall{
		ID:   tc.ID,
		Type: "function",
		Function: struct {
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		}{Name: tc.Name, Arguments: args},
	}
}

// applyOpenAIReasoning sets reasoning_effort and drops temperature, which
// reasoning models reject (dropped uniformly by design).
func applyOpenAIReasoning(req chat.ChatRequest, body *openAIRequest) {
	if req.ReasoningEffort == nil {
		return
	}
	// Reasoning models reject temperature; drop it uniformly when reasoning
	// effort is active (GPT-5.2 accepts it, but we drop by design).
	body.ReasoningEffort = req.ReasoningEffort
	body.Temperature = nil
}

// openAITools converts canonical tool definitions to the OpenAI wire shape.
func openAITools(tools []chat.Tool) []openAITool {
	out := make([]openAITool, 0, len(tools))
	for _, t := range tools {
		out = append(out, openAITool{
			Type: "function",
			Function: openAIFunction{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.InputSchema,
			},
		})
	}
	return out
}

// mapUsage converts an OpenAI usage object into the canonical chat.Usage.
func mapUsage(u openAIUsage) chat.Usage {
	return chat.Usage{
		PromptTokens:     u.PromptTokens,
		CompletionTokens: u.CompletionTokens,
		TotalTokens:      u.TotalTokens,
	}
}

// ---- Complete (non-streaming) ---------------------------------------------

// Complete performs a non-streaming completion and returns the parsed result.
func (p *openAIProvider) Complete(ctx context.Context, req chat.ChatRequest) (chat.ChatResponse, error) {
	ctx, cancel := context.WithTimeout(ctx, p.timeout)
	defer cancel()
	payload, err := buildRequest(req)
	if err != nil {
		return chat.ChatResponse{}, fmt.Errorf("provider openai: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.endpoint(), bytes.NewReader(payload))
	if err != nil {
		return chat.ChatResponse{}, fmt.Errorf("provider openai: build request: %w", err)
	}
	p.setHeaders(httpReq, false)

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return chat.ChatResponse{}, fmt.Errorf("provider openai: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)
		if resp.StatusCode == http.StatusTooManyRequests {
			return chat.ChatResponse{}, &chat.RateLimitError{Provider: p.Name(), StatusCode: resp.StatusCode, Message: strings.TrimSpace(string(body))}
		}
		return chat.ChatResponse{}, &chat.ProviderError{Provider: p.Name(), StatusCode: resp.StatusCode, Message: strings.TrimSpace(string(body))}
	}

	var parsed openAIResponse
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return chat.ChatResponse{}, fmt.Errorf("provider openai: decode response: %w", err)
	}

	out := chat.ChatResponse{
		ID:    parsed.ID,
		Model: parsed.Model,
		Usage: mapUsage(parsed.Usage),
	}
	if len(parsed.Choices) > 0 {
		choice := parsed.Choices[0]
		out.Content = choice.Message.Content
		out.FinishReason = choice.FinishReason
		out.ReasoningContent = choice.Message.ReasoningContent
		if len(choice.Message.ToolCalls) > 0 {
			calls := make([]chat.ToolCall, 0, len(choice.Message.ToolCalls))
			for _, tc := range choice.Message.ToolCalls {
				calls = append(calls, chat.ToolCall{
					ID:        tc.ID,
					Name:      tc.Function.Name,
					Arguments: json.RawMessage(tc.Function.Arguments),
				})
			}
			out.ToolCalls = calls
		}
	}
	return out, nil
}

// ---- Stream (SSE) ---------------------------------------------------------

// toolCallAccum accumulates the fragmented pieces of a single streaming tool
// call (identified by its delta index) until the stream finishes.
type toolCallAccum struct {
	id   string
	name string
	args strings.Builder
}

// buildToolCallDelta assembles the final StreamDelta carrying any accumulated
// tool calls and the stream's accumulated reasoning_content. When no calls
// were collected it returns a plain finish delta.
func buildToolCallDelta(finishReason string, usage *chat.Usage, acc map[int]*toolCallAccum, reasoning string) chat.StreamDelta {
	delta := chat.StreamDelta{FinishReason: finishReason, Usage: usage}
	if reasoning != "" {
		delta.ReasoningContent = reasoning
	}
	if len(acc) == 0 {
		return delta
	}
	indices := make([]int, 0, len(acc))
	for i := range acc {
		indices = append(indices, i)
	}
	sort.Ints(indices)
	calls := make([]chat.ToolCall, 0, len(indices))
	for _, i := range indices {
		a := acc[i]
		args := a.args.String()
		if args == "" {
			args = "{}"
		}
		calls = append(calls, chat.ToolCall{
			ID:        a.id,
			Name:      a.name,
			Arguments: json.RawMessage(args),
		})
	}
	delta.ToolCalls = calls
	return delta
}

// Stream performs a streaming completion, calling emit for each delta and a
// final chunk carrying the finish reason.
func (p *openAIProvider) Stream(ctx context.Context, req chat.ChatRequest, emit chat.StreamFunc) error {
	ctx, cancel := context.WithTimeout(ctx, streamTimeout)
	defer cancel()
	req.Stream = true
	resp, err := p.doStreamRequest(ctx, req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	return consumeOpenAIStream(resp.Body, emit)
}

// doStreamRequest issues the streaming HTTP request and maps non-2xx
// responses to typed errors. The caller owns closing the response body.
func (p *openAIProvider) doStreamRequest(ctx context.Context, req chat.ChatRequest) (*http.Response, error) {
	payload, err := buildRequest(req)
	if err != nil {
		return nil, fmt.Errorf("provider openai: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.endpoint(), bytes.NewReader(payload))
	if err != nil {
		return nil, fmt.Errorf("provider openai: build request: %w", err)
	}
	p.setHeaders(httpReq, true)

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("provider openai: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode == http.StatusTooManyRequests {
			return nil, &chat.RateLimitError{Provider: p.Name(), StatusCode: resp.StatusCode, Message: strings.TrimSpace(string(body))}
		}
		return nil, &chat.ProviderError{Provider: p.Name(), StatusCode: resp.StatusCode, Message: strings.TrimSpace(string(body))}
	}
	return resp, nil
}

// consumeOpenAIStream parses the SSE frame stream and drives emit.
func consumeOpenAIStream(r io.Reader, emit chat.StreamFunc) error {
	st := newOpenAIStreamState()
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	for scanner.Scan() {
		payload, ok := openAIStreamPayload(scanner.Text())
		if !ok {
			continue
		}
		done, err := handleOpenAIStreamPayload(payload, st, emit)
		if err != nil {
			return err
		}
		if done {
			return nil
		}
	}

	if err := scanner.Err(); err != nil {
		return handleOpenAIScanError(err, st, emit)
	}

	return finishOpenAIStream(st, emit)
}

// openAIStreamState accumulates streaming progress across SSE frames.
type openAIStreamState struct {
	emitted bool
	acc     map[int]*toolCallAccum
	// reasoning is stream-global: DeepSeek streams reasoning before any
	// tool-call delta.
	reasoning strings.Builder
}

// newOpenAIStreamState returns an initialized stream state.
func newOpenAIStreamState() *openAIStreamState {
	return &openAIStreamState{acc: make(map[int]*toolCallAccum)}
}

// openAIStreamPayload extracts the payload of a `data:` SSE line; ok is
// false for non-data lines and empty payloads (both are skipped).
func openAIStreamPayload(line string) (string, bool) {
	if !strings.HasPrefix(line, "data:") {
		return "", false
	}
	payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
	if payload == "" {
		return "", false
	}
	return payload, true
}

// handleOpenAIStreamPayload processes one SSE payload. done is true when the
// stream is finished and the caller must return.
func handleOpenAIStreamPayload(payload string, st *openAIStreamState, emit chat.StreamFunc) (bool, error) {
	if payload == "[DONE]" {
		handleOpenAIDone(st, emit)
		return true, nil
	}
	chunk, done, err := parseOpenAIChunk(payload, st, emit)
	if err != nil || done || chunk == nil {
		return done, err
	}
	return applyOpenAIChoice(chunk, st, emit)
}

// handleOpenAIDone handles the [DONE] sentinel: pending tool calls are
// flushed with a stop delta, otherwise the stream simply ends.
func handleOpenAIDone(st *openAIStreamState, emit chat.StreamFunc) {
	if len(st.acc) > 0 {
		_ = emit(buildToolCallDelta("stop", nil, st.acc, st.reasoning.String()))
	}
}

// parseOpenAIChunk decodes one frame. done is true when the frame ended the
// stream (malformed payload after content: error delta already emitted).
func parseOpenAIChunk(payload string, st *openAIStreamState, emit chat.StreamFunc) (*openAIStreamChunk, bool, error) {
	var chunk openAIStreamChunk
	if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
		if !st.emitted {
			return nil, false, fmt.Errorf("provider openai: decode chunk: %w", err)
		}
		_ = emit(chat.StreamDelta{FinishReason: "error"})
		return nil, true, nil
	}
	return &chunk, false, nil
}

// applyOpenAIChoice folds one chunk's first choice into text deltas, pending
// tool calls, and the terminal finish chunk. done is true when a finish
// reason ended the stream.
func applyOpenAIChoice(chunk *openAIStreamChunk, st *openAIStreamState, emit chat.StreamFunc) (bool, error) {
	if len(chunk.Choices) == 0 {
		return false, nil
	}
	choice := chunk.Choices[0]

	if err := emitOpenAIContent(choice, st, emit); err != nil {
		return false, err
	}
	accumulateOpenAIReasoning(choice, st)
	accumulateOpenAIToolCalls(choice, st)

	if choice.FinishReason == "" {
		return false, nil
	}
	emitOpenAIFinish(chunk, choice, st, emit)
	return true, nil
}

// emitOpenAIContent emits one text delta and marks content delivered.
func emitOpenAIContent(choice openAIStreamChoice, st *openAIStreamState, emit chat.StreamFunc) error {
	if choice.Delta.Content == "" {
		return nil
	}
	st.emitted = true
	return emit(chat.StreamDelta{Delta: choice.Delta.Content})
}

// accumulateOpenAIReasoning appends reasoning_content to the stream-global
// buffer.
func accumulateOpenAIReasoning(choice openAIStreamChoice, st *openAIStreamState) {
	if choice.Delta.ReasoningContent != "" {
		st.reasoning.WriteString(choice.Delta.ReasoningContent)
	}
}

// accumulateOpenAIToolCalls merges tool-call deltas into the accumulators.
func accumulateOpenAIToolCalls(choice openAIStreamChoice, st *openAIStreamState) {
	for _, tc := range choice.Delta.ToolCalls {
		accumulateOpenAIToolCall(tc, st)
	}
}

// accumulateOpenAIToolCall merges one tool-call delta fragment.
func accumulateOpenAIToolCall(tc openAIStreamToolCall, st *openAIStreamState) {
	a := st.acc[tc.Index]
	if a == nil {
		a = &toolCallAccum{}
		st.acc[tc.Index] = a
	}
	if tc.ID != "" {
		a.id = tc.ID
	}
	if tc.Function.Name != "" {
		a.name = tc.Function.Name
	}
	if tc.Function.Arguments != "" {
		a.args.WriteString(tc.Function.Arguments)
	}
}

// emitOpenAIFinish emits the terminal chunk carrying the finish reason, any
// usage, pending tool calls, and accumulated reasoning.
func emitOpenAIFinish(chunk *openAIStreamChunk, choice openAIStreamChoice, st *openAIStreamState, emit chat.StreamFunc) {
	var usage *chat.Usage
	if chunk.Usage != nil {
		u := mapUsage(*chunk.Usage)
		usage = &u
	}
	_ = emit(buildToolCallDelta(choice.FinishReason, usage, st.acc, st.reasoning.String()))
}

// handleOpenAIScanError maps a scanner failure to an emitted error delta
// (after content) or a returned error (before any content).
func handleOpenAIScanError(err error, st *openAIStreamState, emit chat.StreamFunc) error {
	if !st.emitted {
		return fmt.Errorf("provider openai: read stream: %w", err)
	}
	_ = emit(chat.StreamDelta{FinishReason: "error"})
	return nil
}

// finishOpenAIStream emits the closing chunk when the stream ends without an
// explicit finish chunk.
func finishOpenAIStream(st *openAIStreamState, emit chat.StreamFunc) error {
	// Stream ended without an explicit finish chunk.
	if !st.emitted && len(st.acc) == 0 {
		return fmt.Errorf("provider openai: stream ended without content")
	}
	_ = emit(buildToolCallDelta("stop", nil, st.acc, st.reasoning.String()))
	return nil
}

// ---- shared helpers -------------------------------------------------------

// endpoint returns the full chat completions URL.
func (p *openAIProvider) endpoint() string {
	return p.baseURL + "/chat/completions"
}

// setHeaders applies the common headers; streaming additionally requests SSE.
func (p *openAIProvider) setHeaders(r *http.Request, streaming bool) {
	r.Header.Set("Authorization", "Bearer "+p.apiKey)
	r.Header.Set("Content-Type", "application/json")
	if streaming {
		r.Header.Set("Accept", "text/event-stream")
	}
}
