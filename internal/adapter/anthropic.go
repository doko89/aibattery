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
	"strings"
	"time"

	"github.com/aibattery/router/internal/chat"
)

// anthropicProvider adapts the canonical chat.Provider port to the Anthropic
// Messages API (POST /v1/messages).
type anthropicProvider struct {
	baseURL string
	apiKey  string
	version string
	client  *http.Client
}

// NewAnthropicProvider returns a chat.Provider backed by the Anthropic
// Messages API. baseURL is the API root (e.g. "https://api.anthropic.com/v1");
// the adapter appends "/messages". version is the anthropic-version header
// value (e.g. "2023-06-01"). timeout bounds each HTTP request.
func NewAnthropicProvider(baseURL, apiKey, version string, timeout time.Duration) chat.Provider {
	return &anthropicProvider{
		baseURL: baseURL,
		apiKey:  apiKey,
		version: version,
		client:  &http.Client{Timeout: timeout},
	}
}

// Name returns the provider identifier.
func (p *anthropicProvider) Name() string { return "anthropic" }

// endpoint returns the fully-qualified /v1/messages URL.
func (p *anthropicProvider) endpoint() string {
	return strings.TrimRight(p.baseURL, "/") + "/messages"
}

// ---- wire types -----------------------------------------------------------

// anthropicMessage is a member of the messages array; Content is a plain
// string or a slice of content blocks (tool_use / tool_result).
type anthropicMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"`
}

type anthropicRequest struct {
	Model       string             `json:"model"`
	MaxTokens   int                `json:"max_tokens"`
	Temperature *float64           `json:"temperature,omitempty"`
	Stream      bool               `json:"stream,omitempty"`
	System      string             `json:"system,omitempty"`
	Messages    []anthropicMessage `json:"messages"`
	Tools       []anthropicTool    `json:"tools,omitempty"`
	Thinking    *anthropicThinking `json:"thinking,omitempty"`
}

// anthropicThinking controls Anthropic's thinking mode. "disabled" turns it
// off; "enabled" (legacy models only) requests extended thinking with a token
// budget; "adaptive" (Claude 4.6+) lets the model self-manage thinking.
type anthropicThinking struct {
	Type         string `json:"type"`
	BudgetTokens int    `json:"budget_tokens,omitempty"`
}

// anthropicTool is a function definition in Anthropic's native wire shape
// (direct name/description/input_schema — no "type":"function" wrapper).
type anthropicTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	InputSchema map[string]any `json:"input_schema"`
}

type anthropicBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     *map[string]any `json:"input,omitempty"`
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   string          `json:"content,omitempty"`
}

type anthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

type anthropicResponse struct {
	ID         string           `json:"id"`
	Type       string           `json:"type"`
	Role       string           `json:"role"`
	Model      string           `json:"model"`
	Content    []anthropicBlock `json:"content"`
	StopReason string           `json:"stop_reason"`
	Usage      anthropicUsage   `json:"usage"`
}

// anthropicStreamEvent is a union of the SSE event payloads the adapter cares
// about. Only the fields relevant to the current event type are populated.
type anthropicStreamEvent struct {
	Type string `json:"type"`
	// Index identifies the content block a start/delta/stop event refers to.
	Index *int `json:"index"`
	// message_start carries the initial message (usage.input_tokens).
	Message *struct {
		Usage anthropicUsage `json:"usage"`
	} `json:"message"`
	// content_block_start carries the block header (type, and for tool_use
	// blocks the id and name).
	ContentBlock *struct {
		Type string `json:"type"`
		ID   string `json:"id"`
		Name string `json:"name"`
	} `json:"content_block"`
	// content_block_delta and message_delta both use the "delta" field but
	// with disjoint shapes; a single struct captures both.
	Delta *struct {
		Type        string `json:"type"`
		Text        string `json:"text"`
		PartialJSON string `json:"partial_json"`
		StopReason  string `json:"stop_reason"`
	} `json:"delta"`
	// message_delta carries cumulative usage.output_tokens.
	Usage *anthropicUsage `json:"usage"`
	// error event payload.
	Error *struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

// streamToolBlock accumulates a streaming tool_use block: id+name from
// content_block_start and the concatenated partial_json fragments.
type streamToolBlock struct {
	id   string
	name string
	sb   strings.Builder
}

// ---- request translation --------------------------------------------------

// buildRequest translates a canonical ChatRequest into the Anthropic wire
// shape.
func buildAnthropicRequest(req chat.ChatRequest) anthropicRequest {
	maxTokens := 1024
	if req.MaxTokens != nil {
		maxTokens = *req.MaxTokens
	}

	out := anthropicRequest{
		Model:     req.Model,
		MaxTokens: maxTokens,
		Stream:    req.Stream,
		System:    req.System,
		Messages:  make([]anthropicMessage, 0, len(req.Messages)),
	}

	if req.Temperature != nil {
		out.Temperature = req.Temperature
	}

	if req.ReasoningEffort != nil {
		out.Thinking = anthropicThinkingFor(*req.ReasoningEffort, req.Model)
		// budget_tokens counts toward max_tokens; Anthropic rejects a budget
		// >= max_tokens, so raise max_tokens above the budget when needed.
		if out.Thinking.Type == "enabled" && out.MaxTokens <= out.Thinking.BudgetTokens {
			out.MaxTokens = out.Thinking.BudgetTokens + 1
		}
	}

	if len(req.Tools) > 0 {
		out.Tools = make([]anthropicTool, 0, len(req.Tools))
		for _, t := range req.Tools {
			schema := t.InputSchema
			if schema == nil {
				schema = map[string]any{"type": "object"}
			}
			out.Tools = append(out.Tools, anthropicTool{
				Name:        t.Name,
				Description: t.Description,
				InputSchema: schema,
			})
		}
	}

	for _, m := range req.Messages {
		role := anthropicRole(m.Role)
		if role == "" {
			// RoleSystem is promoted to the top-level "system" field and is
			// not a valid member of the messages array.
			continue
		}
		wire := anthropicMessage{Role: role}
		switch {
		case m.Role == chat.Role("tool"):
			wire.Content = []anthropicBlock{{
				Type:      "tool_result",
				ToolUseID: m.ToolCallID,
				Content:   m.Content,
			}}
		case len(m.ToolCalls) > 0:
			blocks := make([]anthropicBlock, 0, len(m.ToolCalls)+1)
			if m.Content != "" {
				blocks = append(blocks, anthropicBlock{Type: "text", Text: m.Content})
			}
			for _, tc := range m.ToolCalls {
				var input map[string]any
				if err := json.Unmarshal(tc.Arguments, &input); err != nil || input == nil {
					input = map[string]any{}
				}
				blocks = append(blocks, anthropicBlock{
					Type:  "tool_use",
					ID:    tc.ID,
					Name:  tc.Name,
					Input: &input,
				})
			}
			wire.Content = blocks
		default:
			wire.Content = m.Content
		}
		out.Messages = append(out.Messages, wire)
	}

	return out
}

// anthropicThinkingFor maps a canonical reasoning effort to the Anthropic
// thinking parameter. "none" disables thinking; legacy models get extended
// thinking ("enabled" + budget); modern models (Claude 4.6+) get "adaptive".
// Anthropic deprecated extended thinking on Claude 4.6 and rejects it with
// HTTP 400 on 4.7+, so the model must be classified before choosing the mode.
func anthropicThinkingFor(effort, model string) *anthropicThinking {
	if effort == "none" {
		return &anthropicThinking{Type: "disabled"}
	}
	if isLegacyThinkingModel(model) {
		return &anthropicThinking{Type: "enabled", BudgetTokens: budgetForEffort(effort)}
	}
	return &anthropicThinking{Type: "adaptive"}
}

// isLegacyThinkingModel reports whether the model predates adaptive thinking
// (Claude 4.6). Legacy naming: claude-*-4-5 and earlier (claude-3-5-sonnet,
// claude-3-7, claude-2, claude-1). Anything else — 4-6, 4-7, 5.x, aliases —
// is treated as modern and must use "adaptive".
func isLegacyThinkingModel(model string) bool {
	for _, s := range []string{"4-5", "4.5", "3-", "claude-2", "claude-1"} {
		if strings.Contains(model, s) {
			return true
		}
	}
	return false
}

// budgetForEffort maps a reasoning effort to an extended-thinking token
// budget. Unknown efforts default to the medium tier.
func budgetForEffort(effort string) int {
	switch effort {
	case "minimal":
		return 1024
	case "low":
		return 2048
	case "high":
		return 16000
	case "xhigh", "max":
		return 32000
	default: // "medium" and unknown values
		return 8192
	}
}

// mapRole translates a canonical Role into the Anthropic role string. It
// returns "" for roles Anthropic does not accept inside messages (system).
func anthropicRole(r chat.Role) string {
	switch r {
	case chat.RoleUser:
		return "user"
	case chat.RoleAssistant:
		return "assistant"
	case chat.Role("tool"):
		// Tool results are sent as user messages; the tool_result content
		// block itself carries the semantics.
		return "user"
	default:
		return ""
	}
}

// mapFinishReason translates an Anthropic stop_reason into the canonical
// finish reason vocabulary.
func anthropicFinishReason(s string) string {
	switch s {
	case "end_turn", "stop_sequence":
		return "stop"
	case "max_tokens":
		return "length"
	case "tool_use":
		return "tool_calls"
	case "refusal":
		return "content_filter"
	default:
		return s
	}
}

// ---- Complete ----------------------------------------------------------------

// Complete performs a non-streaming completion against /v1/messages.
func (p *anthropicProvider) Complete(ctx context.Context, req chat.ChatRequest) (chat.ChatResponse, error) {
	body, err := json.Marshal(buildAnthropicRequest(req))
	if err != nil {
		return chat.ChatResponse{}, fmt.Errorf("anthropic: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.endpoint(), bytes.NewReader(body))
	if err != nil {
		return chat.ChatResponse{}, fmt.Errorf("anthropic: build request: %w", err)
	}
	p.setHeaders(httpReq, false)

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return chat.ChatResponse{}, fmt.Errorf("anthropic: request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return chat.ChatResponse{}, p.statusError(resp)
	}

	var wire anthropicResponse
	if err := json.NewDecoder(resp.Body).Decode(&wire); err != nil {
		return chat.ChatResponse{}, fmt.Errorf("anthropic: decode response: %w", err)
	}

	var sb strings.Builder
	var toolCalls []chat.ToolCall
	for _, block := range wire.Content {
		switch block.Type {
		case "text":
			sb.WriteString(block.Text)
		case "tool_use":
			args, err := json.Marshal(block.Input)
			if err != nil {
				args = json.RawMessage("{}")
			}
			toolCalls = append(toolCalls, chat.ToolCall{
				ID:        block.ID,
				Name:      block.Name,
				Arguments: args,
			})
		}
	}

	usage := chat.Usage{
		PromptTokens:     wire.Usage.InputTokens,
		CompletionTokens: wire.Usage.OutputTokens,
		TotalTokens:      wire.Usage.InputTokens + wire.Usage.OutputTokens,
	}

	out := chat.ChatResponse{
		ID:           wire.ID,
		Model:        wire.Model,
		Content:      sb.String(),
		FinishReason: anthropicFinishReason(wire.StopReason),
		Usage:        usage,
	}
	if len(toolCalls) > 0 {
		out.ToolCalls = toolCalls
	}
	return out, nil
}

// ---- Stream -------------------------------------------------------------

// Stream performs a streaming completion against /v1/messages, emitting each
// text delta and a final chunk carrying the finish reason and usage.
func (p *anthropicProvider) Stream(ctx context.Context, req chat.ChatRequest, emit chat.StreamFunc) error {
	req.Stream = true
	body, err := json.Marshal(buildAnthropicRequest(req))
	if err != nil {
		return fmt.Errorf("anthropic: marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.endpoint(), bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("anthropic: build request: %w", err)
	}
	p.setHeaders(httpReq, true)

	resp, err := p.client.Do(httpReq)
	if err != nil {
		return fmt.Errorf("anthropic: stream request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return p.statusError(resp)
	}

	return p.consumeStream(resp.Body, emit)
}

// consumeStream parses the SSE frame stream and drives emit.
func (p *anthropicProvider) consumeStream(r io.Reader, emit chat.StreamFunc) error {
	var (
		delivered    bool
		inputTokens  int
		outputTokens int
		stopReason   string
		toolBlocks   map[int]*streamToolBlock
	)

	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	var (
		eventName string
		dataLines []string
	)

	flush := func() error {
		if len(dataLines) == 0 {
			return nil
		}
		payload := strings.Join(dataLines, "\n")
		dataLines = dataLines[:0]

		var ev anthropicStreamEvent
		if err := json.Unmarshal([]byte(payload), &ev); err != nil {
			// Ignore malformed frames; keep streaming.
			return nil
		}

		switch eventName {
		case "message_start":
			if ev.Message != nil {
				inputTokens = ev.Message.Usage.InputTokens
			}
		case "content_block_start":
			if ev.ContentBlock != nil && ev.ContentBlock.Type == "tool_use" && ev.Index != nil {
				if toolBlocks == nil {
					toolBlocks = make(map[int]*streamToolBlock)
				}
				toolBlocks[*ev.Index] = &streamToolBlock{
					id:   ev.ContentBlock.ID,
					name: ev.ContentBlock.Name,
				}
				delivered = true
			}
		case "content_block_delta":
			if ev.Delta == nil {
				return nil
			}
			switch ev.Delta.Type {
			case "text_delta":
				delivered = true
				if err := emit(chat.StreamDelta{Delta: ev.Delta.Text}); err != nil {
					return err
				}
			case "input_json_delta":
				if ev.Index != nil {
					if tb, ok := toolBlocks[*ev.Index]; ok {
						tb.sb.WriteString(ev.Delta.PartialJSON)
					}
				}
			}
		case "message_delta":
			if ev.Delta != nil {
				stopReason = ev.Delta.StopReason
			}
			if ev.Usage != nil {
				outputTokens = ev.Usage.OutputTokens
			}
		case "message_stop":
			usage := &chat.Usage{
				PromptTokens:     inputTokens,
				CompletionTokens: outputTokens,
				TotalTokens:      inputTokens + outputTokens,
			}
			if len(toolBlocks) > 0 {
				calls := make([]chat.ToolCall, 0, len(toolBlocks))
				for _, tb := range toolBlocks {
					raw := tb.sb.String()
					if raw == "" {
						raw = "{}"
					}
					calls = append(calls, chat.ToolCall{
						ID:        tb.id,
						Name:      tb.name,
						Arguments: json.RawMessage(raw),
					})
				}
				return emit(chat.StreamDelta{
					FinishReason: "tool_calls",
					Usage:        usage,
					ToolCalls:    calls,
				})
			}
			return emit(chat.StreamDelta{
				FinishReason: anthropicFinishReason(stopReason),
				Usage:        usage,
			})
		case "error":
			if delivered {
				return emit(chat.StreamDelta{FinishReason: "error"})
			}
			msg := "anthropic: stream error"
			if ev.Error != nil && ev.Error.Message != "" {
				msg = "anthropic: stream error: " + ev.Error.Message
			}
			return fmt.Errorf("%s", msg)
		}
		return nil
	}

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			if err := flush(); err != nil {
				return err
			}
			eventName = ""
			continue
		}
		if strings.HasPrefix(line, "event:") {
			eventName = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
			continue
		}
		if strings.HasPrefix(line, "data:") {
			dataLines = append(dataLines, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
			continue
		}
		// Ignore comments and other SSE fields.
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("anthropic: read stream: %w", err)
	}

	// A trailing frame may not be followed by a blank line (bufio.Scanner
	// does not emit a final empty token), so flush any pending frame.
	if err := flush(); err != nil {
		return err
	}

	// Stream ended without a message_stop event.
	if !delivered {
		return fmt.Errorf("anthropic: stream ended before any content")
	}
	return nil
}

// ---- helpers ----------------------------------------------------------------

func (p *anthropicProvider) setHeaders(r *http.Request, streaming bool) {
	r.Header.Set("x-api-key", p.apiKey)
	r.Header.Set("anthropic-version", p.version)
	r.Header.Set("Content-Type", "application/json")
	if streaming {
		r.Header.Set("Accept", "text/event-stream")
	}
}

// statusError builds an error from a non-200 response, attempting to surface
// the Anthropic error message.
func (p *anthropicProvider) statusError(resp *http.Response) error {
	body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
	msg := strings.TrimSpace(string(body))
	if msg == "" {
		msg = resp.Status
	}
	if resp.StatusCode == http.StatusTooManyRequests {
		return &chat.RateLimitError{Provider: p.Name(), StatusCode: resp.StatusCode, Message: msg}
	}
	return fmt.Errorf("anthropic: %s: %s", resp.Status, msg)
}
