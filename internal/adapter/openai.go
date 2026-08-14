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
	messages := make([]openAIMessage, 0, len(req.Messages))
	for _, m := range req.Messages {
		msg := openAIMessage{Role: string(m.Role), Content: m.Content}
		if m.ToolCallID != "" {
			msg.ToolCallID = m.ToolCallID
		}
		if m.ReasoningContent != "" {
			msg.ReasoningContent = m.ReasoningContent
		}
		if len(m.ToolCalls) > 0 {
			calls := make([]openAIToolCall, 0, len(m.ToolCalls))
			for _, tc := range m.ToolCalls {
				args := string(tc.Arguments)
				if len(tc.Arguments) == 0 {
					args = "{}"
				}
				calls = append(calls, openAIToolCall{
					ID:   tc.ID,
					Type: "function",
					Function: struct {
						Name      string `json:"name"`
						Arguments string `json:"arguments"`
					}{Name: tc.Name, Arguments: args},
				})
			}
			msg.ToolCalls = calls
		}
		messages = append(messages, msg)
	}

	body := openAIRequest{
		Model:       req.Model,
		Messages:    messages,
		Stream:      req.Stream,
		Temperature: req.Temperature,
		MaxTokens:   req.MaxTokens,
	}
	// Reasoning models reject temperature; drop it uniformly when reasoning
	// effort is active (GPT-5.2 accepts it, but we drop by design).
	if req.ReasoningEffort != nil {
		body.ReasoningEffort = req.ReasoningEffort
		body.Temperature = nil
	}
	if len(req.Tools) > 0 {
		tools := make([]openAITool, 0, len(req.Tools))
		for _, t := range req.Tools {
			tools = append(tools, openAITool{
				Type: "function",
				Function: openAIFunction{
					Name:        t.Name,
					Description: t.Description,
					Parameters:  t.InputSchema,
				},
			})
		}
		body.Tools = tools
	}
	return json.Marshal(body)
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
	payload, err := buildRequest(req)
	if err != nil {
		return fmt.Errorf("provider openai: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.endpoint(), bytes.NewReader(payload))
	if err != nil {
		return fmt.Errorf("provider openai: build request: %w", err)
	}
	p.setHeaders(httpReq, true)

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return fmt.Errorf("provider openai: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(resp.Body)
		if resp.StatusCode == http.StatusTooManyRequests {
			return &chat.RateLimitError{Provider: p.Name(), StatusCode: resp.StatusCode, Message: strings.TrimSpace(string(body))}
		}
		return &chat.ProviderError{Provider: p.Name(), StatusCode: resp.StatusCode, Message: strings.TrimSpace(string(body))}
	}

	emitted := false
	acc := make(map[int]*toolCallAccum)
	var reasoning strings.Builder // stream-global: DeepSeek streams reasoning before any tool-call delta
	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "" {
			continue
		}
		if payload == "[DONE]" {
			if len(acc) > 0 {
				_ = emit(buildToolCallDelta("stop", nil, acc, reasoning.String()))
			}
			return nil
		}

		var chunk openAIStreamChunk
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			if !emitted {
				return fmt.Errorf("provider openai: decode chunk: %w", err)
			}
			_ = emit(chat.StreamDelta{FinishReason: "error"})
			return nil
		}

		if len(chunk.Choices) == 0 {
			continue
		}
		choice := chunk.Choices[0]

		if choice.Delta.Content != "" {
			emitted = true
			if err := emit(chat.StreamDelta{Delta: choice.Delta.Content}); err != nil {
				return err
			}
		}

		if choice.Delta.ReasoningContent != "" {
			reasoning.WriteString(choice.Delta.ReasoningContent)
		}

		for _, tc := range choice.Delta.ToolCalls {
			a := acc[tc.Index]
			if a == nil {
				a = &toolCallAccum{}
				acc[tc.Index] = a
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

		if choice.FinishReason != "" {
			var usage *chat.Usage
			if chunk.Usage != nil {
				u := mapUsage(*chunk.Usage)
				usage = &u
			}
			_ = emit(buildToolCallDelta(choice.FinishReason, usage, acc, reasoning.String()))
			return nil
		}
	}

	if err := scanner.Err(); err != nil {
		if !emitted {
			return fmt.Errorf("provider openai: read stream: %w", err)
		}
		_ = emit(chat.StreamDelta{FinishReason: "error"})
		return nil
	}

	// Stream ended without an explicit finish chunk.
	if !emitted && len(acc) == 0 {
		return fmt.Errorf("provider openai: stream ended without content")
	}
	_ = emit(buildToolCallDelta("stop", nil, acc, reasoning.String()))
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
