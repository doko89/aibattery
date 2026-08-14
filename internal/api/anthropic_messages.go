package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"strings"

	"github.com/aibattery/router/internal/chat"
	"github.com/aibattery/router/internal/routing"
)

// anthropicMessageRequest is the Anthropic Messages API wire format accepted
// by POST /anthropic/v1/messages. The `model` field carries the VIRTUAL model
// name that the routing layer resolves to a concrete provider+model pair.
type anthropicMessageRequest struct {
	Model       string                 `json:"model"`
	MaxTokens   *int                   `json:"max_tokens"`
	Messages    []anthropicWireMessage `json:"messages"`
	System      string                 `json:"system"`
	Temperature *float64               `json:"temperature"`
	Stream      bool                   `json:"stream"`
	Tools       []anthropicWireTool    `json:"tools"`
}

// anthropicWireMessage is a single message. Content is kept as raw JSON so
// both a plain string and an array of content blocks decode.
type anthropicWireMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

// anthropicWireTool is a tool definition in the Anthropic wire shape.
type anthropicWireTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	InputSchema map[string]any `json:"input_schema"`
}

// anthropicMessageResponse is the Anthropic non-streaming message payload.
type anthropicMessageResponse struct {
	ID           string                  `json:"id"`
	Type         string                  `json:"type"`
	Role         string                  `json:"role"`
	Model        string                  `json:"model"`
	Content      []anthropicContentBlock `json:"content"`
	StopReason   string                  `json:"stop_reason"`
	StopSequence any                     `json:"stop_sequence"`
	Usage        anthropicUsage          `json:"usage"`
}

// anthropicContentBlock is one entry in the response content array: a text
// block or a tool_use block.
type anthropicContentBlock struct {
	Type  string `json:"type"`
	Text  string `json:"text,omitempty"`
	ID    string `json:"id,omitempty"`
	Name  string `json:"name,omitempty"`
	Input any    `json:"input,omitempty"`
}

// anthropicUsage reports token consumption in the Anthropic shape.
type anthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// anthropicErrorEnvelope is the Anthropic-compatible error envelope.
type anthropicErrorEnvelope struct {
	Type  string                `json:"type"`
	Error anthropicErrorDetails `json:"error"`
}

// anthropicErrorDetails carries the error type and message.
type anthropicErrorDetails struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

// writeAnthropicError writes an Anthropic error envelope with the given
// status and error type.
func writeAnthropicError(w http.ResponseWriter, status int, typ, message string) {
	writeJSON(w, status, anthropicErrorEnvelope{Type: "error", Error: anthropicErrorDetails{Type: typ, Message: message}})
}

// parseAnthropicMessageRequest translates the Anthropic wire body into a
// canonical chat.ChatRequest and reports whether streaming was requested.
// A missing max_tokens defaults to 1024 (Anthropic requires it, we are
// lenient). Tools sent by the client replace rather than extend the
// aggregated tool set.
func parseAnthropicMessageRequest(body []byte) (chat.ChatRequest, bool, error) {
	var wire anthropicMessageRequest
	if err := json.Unmarshal(body, &wire); err != nil {
		return chat.ChatRequest{}, false, err
	}
	if wire.Model == "" {
		return chat.ChatRequest{}, false, errors.New("model is required")
	}

	var messages []chat.Message
	for _, m := range wire.Messages {
		content, err := anthropicContentText(m.Content)
		if err != nil {
			return chat.ChatRequest{}, false, err
		}
		messages = append(messages, chat.Message{Role: chat.Role(m.Role), Content: content})
	}

	maxTokens := 1024
	if wire.MaxTokens != nil {
		maxTokens = *wire.MaxTokens
	}

	cReq := chat.ChatRequest{
		Model:       wire.Model,
		Messages:    messages,
		System:      wire.System,
		Temperature: wire.Temperature,
		MaxTokens:   &maxTokens,
		Stream:      wire.Stream,
	}
	if len(wire.Tools) > 0 {
		tools := make([]chat.Tool, 0, len(wire.Tools))
		for _, t := range wire.Tools {
			tools = append(tools, chat.Tool{Name: t.Name, Description: t.Description, InputSchema: t.InputSchema})
		}
		cReq.Tools = tools
	}
	return cReq, wire.Stream, nil
}

// anthropicContentText extracts the plain text from a message content that is
// either a string or an array of text blocks. Non-text blocks are skipped.
func anthropicContentText(raw json.RawMessage) (string, error) {
	if len(raw) == 0 {
		return "", nil
	}
	if raw[0] == '"' {
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			return "", err
		}
		return s, nil
	}
	var blocks []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return "", fmt.Errorf("invalid content: %w", err)
	}
	var sb strings.Builder
	for _, b := range blocks {
		if b.Type == "text" {
			sb.WriteString(b.Text)
		}
	}
	return sb.String(), nil
}

// handleAnthropicMessages serves POST /anthropic/v1/messages. It parses the
// Anthropic wire request, resolves the virtual model name to a
// routing.Selector, then dispatches to the streaming sibling (implemented in
// anthropic_stream.go) or the local non-streaming failover loop.
func (s *Server) handleAnthropicMessages(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "failed to read body: "+err.Error())
		return
	}
	cReq, stream, err := parseAnthropicMessageRequest(body)
	if err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}

	sel, err := s.deps.Registry.Select(cReq.Model)
	if err != nil {
		if errors.Is(err, routing.ErrModelNotFound) {
			writeAnthropicError(w, http.StatusNotFound, "invalid_request_error", fmt.Sprintf("model '%s' not found", cReq.Model))
			return
		}
		writeAnthropicError(w, http.StatusInternalServerError, "invalid_request_error", err.Error())
		return
	}

	if stream {
		s.anthropicStream(w, r, sel, cReq)
		return
	}

	// The aggregated tool set is offered only when the client sent none.
	if len(cReq.Tools) == 0 && s.deps.Tools != nil {
		cReq.Tools = s.deps.Tools.List()
	}
	s.completeAnthropic(w, r, sel, cReq)
}

// completeAnthropic runs the non-streaming failover loop, mirroring the
// OpenAI flow in chat_completions.go: forward the concrete model name to each
// candidate provider and return the first successful completion as an
// Anthropic message. If every candidate fails it responds 502.
func (s *Server) completeAnthropic(w http.ResponseWriter, r *http.Request, sel routing.Selector, cReq chat.ChatRequest) {
	ctx := r.Context()
	for _, cand := range sel.Begin() {
		cReq.Model = cand.Model
		p, ok := s.deps.Providers[cand.ProviderName]
		if !ok {
			sel.RecordFailure(cand)
			continue
		}
		resp, err := p.Complete(ctx, cReq)
		if err != nil {
			s.deps.Logger.Warn("provider completion failed",
				"provider", cand.ProviderName, "model", cand.Model, "error", err)
			sel.RecordFailure(cand)
			continue
		}
		sel.RecordSuccess(cand)
		writeAnthropicMessage(w, cand.Model, resp)
		return
	}
	writeAnthropicError(w, http.StatusBadGateway, "api_error", "all model candidates failed")
}

// writeAnthropicMessage serializes a canonical chat.ChatResponse into the
// Anthropic non-streaming message shape.
func writeAnthropicMessage(w http.ResponseWriter, model string, resp chat.ChatResponse) {
	content := make([]anthropicContentBlock, 0, 1+len(resp.ToolCalls))
	if resp.Content != "" {
		content = append(content, anthropicContentBlock{Type: "text", Text: resp.Content})
	}
	for _, tc := range resp.ToolCalls {
		var input any = map[string]any{}
		if len(tc.Arguments) > 0 && json.Unmarshal(tc.Arguments, &input) != nil {
			input = map[string]any{}
		}
		content = append(content, anthropicContentBlock{Type: "tool_use", ID: tc.ID, Name: tc.Name, Input: input})
	}

	out := anthropicMessageResponse{
		ID:           fmt.Sprintf("msg_%x", rand.Uint64()),
		Type:         "message",
		Role:         "assistant",
		Model:        model,
		Content:      content,
		StopReason:   anthropicStopReason(resp.FinishReason),
		StopSequence: nil,
		Usage: anthropicUsage{
			InputTokens:  resp.Usage.PromptTokens,
			OutputTokens: resp.Usage.CompletionTokens,
		},
	}
	writeJSON(w, http.StatusOK, out)
}

// anthropicStopReason maps a canonical FinishReason to the Anthropic
// stop_reason vocabulary.
func anthropicStopReason(finish string) string {
	switch finish {
	case "stop":
		return "end_turn"
	case "length":
		return "max_tokens"
	case "tool_calls":
		return "tool_use"
	case "error":
		return "error"
	default:
		return "end_turn"
	}
}
