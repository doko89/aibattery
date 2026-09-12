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
	timeout time.Duration
}

// NewAnthropicProvider returns a chat.Provider backed by the Anthropic
// Messages API. baseURL is the API root (e.g. "https://api.anthropic.com/v1");
// the adapter appends "/messages". version is the anthropic-version header
// value (e.g. "2023-06-01"). timeout bounds each non-streaming completion;
// streaming uses the shared streamTimeout budget.
func NewAnthropicProvider(baseURL, apiKey, version string, timeout time.Duration) chat.Provider {
	return &anthropicProvider{
		baseURL: baseURL,
		apiKey:  apiKey,
		version: version,
		client:  &http.Client{},
		timeout: timeout,
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
	out := anthropicRequest{
		Model:     req.Model,
		MaxTokens: anthropicMaxTokens(req),
		Stream:    req.Stream,
		System:    req.System,
		Messages:  make([]anthropicMessage, 0, len(req.Messages)),
	}

	if req.Temperature != nil {
		out.Temperature = req.Temperature
	}

	applyAnthropicThinking(req, &out)

	if len(req.Tools) > 0 {
		out.Tools = anthropicTools(req.Tools)
	}

	out.Messages = anthropicMessages(req.Messages)

	return out
}

// anthropicMaxTokens resolves the effective max_tokens (default 1024).
func anthropicMaxTokens(req chat.ChatRequest) int {
	if req.MaxTokens != nil {
		return *req.MaxTokens
	}
	return 1024
}

// applyAnthropicThinking sets the thinking parameter and bumps max_tokens
// above the thinking budget when Anthropic would reject budget >= max_tokens.
func applyAnthropicThinking(req chat.ChatRequest, out *anthropicRequest) {
	if req.ReasoningEffort == nil {
		return
	}
	out.Thinking = anthropicThinkingFor(*req.ReasoningEffort, req.Model)
	// budget_tokens counts toward max_tokens; Anthropic rejects a budget
	// >= max_tokens, so raise max_tokens above the budget when needed.
	if out.Thinking.Type == "enabled" && out.MaxTokens <= out.Thinking.BudgetTokens {
		out.MaxTokens = out.Thinking.BudgetTokens + 1
	}
}

// anthropicTools converts canonical tool definitions to the Anthropic wire
// shape.
func anthropicTools(tools []chat.Tool) []anthropicTool {
	out := make([]anthropicTool, 0, len(tools))
	for _, t := range tools {
		schema := t.InputSchema
		if schema == nil {
			schema = map[string]any{"type": "object"}
		}
		out = append(out, anthropicTool{
			Name:        t.Name,
			Description: t.Description,
			InputSchema: schema,
		})
	}
	return out
}

// anthropicMessages converts canonical messages, skipping roles Anthropic
// does not accept inside messages (system is promoted to top-level).
func anthropicMessages(messages []chat.Message) []anthropicMessage {
	out := make([]anthropicMessage, 0, len(messages))
	for _, m := range messages {
		wire, ok := anthropicMessageFor(m)
		if !ok {
			// RoleSystem is promoted to the top-level "system" field and is
			// not a valid member of the messages array.
			continue
		}
		out = append(out, wire)
	}
	return out
}

// anthropicMessageFor converts one canonical message; ok is false when the
// role has no Anthropic messages-array representation.
func anthropicMessageFor(m chat.Message) (anthropicMessage, bool) {
	role := anthropicRole(m.Role)
	if role == "" {
		return anthropicMessage{}, false
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
		wire.Content = anthropicToolUseBlocks(m)
	default:
		wire.Content = m.Content
	}
	return wire, true
}

// anthropicToolUseBlocks builds the text + tool_use blocks for an assistant
// message carrying tool calls.
func anthropicToolUseBlocks(m chat.Message) []anthropicBlock {
	blocks := make([]anthropicBlock, 0, len(m.ToolCalls)+1)
	if m.Content != "" {
		blocks = append(blocks, anthropicBlock{Type: "text", Text: m.Content})
	}
	for _, tc := range m.ToolCalls {
		input := anthropicToolInput(tc.Arguments)
		blocks = append(blocks, anthropicBlock{
			Type:  "tool_use",
			ID:    tc.ID,
			Name:  tc.Name,
			Input: &input,
		})
	}
	return blocks
}

// anthropicToolInput decodes raw tool arguments; invalid or null input falls
// back to an empty object so the wire never carries a bad payload.
func anthropicToolInput(raw json.RawMessage) map[string]any {
	var input map[string]any
	if err := json.Unmarshal(raw, &input); err != nil || input == nil {
		return map[string]any{}
	}
	return input
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
	ctx, cancel := context.WithTimeout(ctx, p.timeout)
	defer cancel()
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
	ctx, cancel := context.WithTimeout(ctx, streamTimeout)
	defer cancel()
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

// anthropicStreamState accumulates streaming progress across SSE frames.
type anthropicStreamState struct {
	delivered    bool
	inputTokens  int
	outputTokens int
	stopReason   string
	toolBlocks   map[int]*streamToolBlock
}

// consumeStream parses the SSE frame stream and drives emit.
func (p *anthropicProvider) consumeStream(r io.Reader, emit chat.StreamFunc) error {
	st := &anthropicStreamState{}

	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)

	var (
		eventName string
		dataLines []string
	)

	flush := func() error {
		payload, ok := anthropicFramePayload(&dataLines)
		if !ok {
			return nil
		}
		ev, ok := anthropicParseEvent(payload)
		if !ok {
			// Ignore malformed frames; keep streaming.
			return nil
		}
		return handleAnthropicStreamEvent(eventName, ev, st, emit)
	}

	for scanner.Scan() {
		if anthropicProcessLine(scanner.Text(), &eventName, &dataLines) {
			continue
		}
		if err := flush(); err != nil {
			return err
		}
		eventName = ""
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
	if !st.delivered {
		return fmt.Errorf("anthropic: stream ended before any content")
	}
	return nil
}

// anthropicFramePayload joins pending data lines into one payload and clears
// them. ok is false when there is no pending frame.
func anthropicFramePayload(dataLines *[]string) (string, bool) {
	if len(*dataLines) == 0 {
		return "", false
	}
	payload := strings.Join(*dataLines, "\n")
	*dataLines = (*dataLines)[:0]
	return payload, true
}

// anthropicParseEvent decodes one SSE payload; ok is false on malformed JSON.
func anthropicParseEvent(payload string) (anthropicStreamEvent, bool) {
	var ev anthropicStreamEvent
	if err := json.Unmarshal([]byte(payload), &ev); err != nil {
		return anthropicStreamEvent{}, false
	}
	return ev, true
}

// anthropicProcessLine routes one scanner line into the SSE frame being
// built. It returns true when the scan loop should continue without flushing
// (non-blank line); blank lines return false so the caller flushes.
func anthropicProcessLine(line string, eventName *string, dataLines *[]string) bool {
	if line == "" {
		return false
	}
	if strings.HasPrefix(line, "event:") {
		*eventName = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
		return true
	}
	if strings.HasPrefix(line, "data:") {
		*dataLines = append(*dataLines, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
		return true
	}
	// Ignore comments and other SSE fields.
	return true
}

// handleAnthropicStreamEvent dispatches one decoded SSE event.
func handleAnthropicStreamEvent(eventName string, ev anthropicStreamEvent, st *anthropicStreamState, emit chat.StreamFunc) error {
	switch eventName {
	case "message_start":
		handleAnthropicMessageStart(ev, st)
	case "content_block_start":
		handleAnthropicBlockStart(ev, st)
	case "content_block_delta":
		return handleAnthropicBlockDelta(ev, st, emit)
	case "message_delta":
		handleAnthropicMessageDelta(ev, st)
	case "message_stop":
		return handleAnthropicMessageStop(st, emit)
	case "error":
		return handleAnthropicStreamError(ev, st, emit)
	}
	return nil
}

// handleAnthropicMessageStart records usage.input_tokens from message_start.
func handleAnthropicMessageStart(ev anthropicStreamEvent, st *anthropicStreamState) {
	if ev.Message != nil {
		st.inputTokens = ev.Message.Usage.InputTokens
	}
}

// handleAnthropicBlockStart registers a streaming tool_use block header.
func handleAnthropicBlockStart(ev anthropicStreamEvent, st *anthropicStreamState) {
	if ev.ContentBlock == nil || ev.ContentBlock.Type != "tool_use" || ev.Index == nil {
		return
	}
	if st.toolBlocks == nil {
		st.toolBlocks = make(map[int]*streamToolBlock)
	}
	st.toolBlocks[*ev.Index] = &streamToolBlock{
		id:   ev.ContentBlock.ID,
		name: ev.ContentBlock.Name,
	}
	st.delivered = true
}

// handleAnthropicBlockDelta handles text and input_json deltas.
func handleAnthropicBlockDelta(ev anthropicStreamEvent, st *anthropicStreamState, emit chat.StreamFunc) error {
	if ev.Delta == nil {
		return nil
	}
	switch ev.Delta.Type {
	case "text_delta":
		return emitAnthropicTextDelta(ev, st, emit)
	case "input_json_delta":
		appendAnthropicPartialJSON(ev, st)
	}
	return nil
}

// emitAnthropicTextDelta emits one text delta and marks content delivered.
func emitAnthropicTextDelta(ev anthropicStreamEvent, st *anthropicStreamState, emit chat.StreamFunc) error {
	st.delivered = true
	return emit(chat.StreamDelta{Delta: ev.Delta.Text})
}

// appendAnthropicPartialJSON appends an input_json fragment to its tool block.
func appendAnthropicPartialJSON(ev anthropicStreamEvent, st *anthropicStreamState) {
	if ev.Index == nil {
		return
	}
	tb, ok := st.toolBlocks[*ev.Index]
	if !ok {
		return
	}
	tb.sb.WriteString(ev.Delta.PartialJSON)
}

// handleAnthropicMessageDelta records the stop reason and output tokens.
func handleAnthropicMessageDelta(ev anthropicStreamEvent, st *anthropicStreamState) {
	if ev.Delta != nil {
		st.stopReason = ev.Delta.StopReason
	}
	if ev.Usage != nil {
		st.outputTokens = ev.Usage.OutputTokens
	}
}

// handleAnthropicMessageStop emits the final chunk with finish reason, usage,
// and any accumulated tool calls.
func handleAnthropicMessageStop(st *anthropicStreamState, emit chat.StreamFunc) error {
	usage := &chat.Usage{
		PromptTokens:     st.inputTokens,
		CompletionTokens: st.outputTokens,
		TotalTokens:      st.inputTokens + st.outputTokens,
	}
	if len(st.toolBlocks) > 0 {
		return emit(chat.StreamDelta{
			FinishReason: "tool_calls",
			Usage:        usage,
			ToolCalls:    anthropicStreamToolCalls(st.toolBlocks),
		})
	}
	return emit(chat.StreamDelta{
		FinishReason: anthropicFinishReason(st.stopReason),
		Usage:        usage,
	})
}

// anthropicStreamToolCalls materializes accumulated tool blocks.
func anthropicStreamToolCalls(toolBlocks map[int]*streamToolBlock) []chat.ToolCall {
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
	return calls
}

// handleAnthropicStreamError maps an error event to an emitted error delta
// (after content) or a returned error (before any content).
func handleAnthropicStreamError(ev anthropicStreamEvent, st *anthropicStreamState, emit chat.StreamFunc) error {
	if st.delivered {
		return emit(chat.StreamDelta{FinishReason: "error"})
	}
	msg := "anthropic: stream error"
	if ev.Error != nil && ev.Error.Message != "" {
		msg = "anthropic: stream error: " + ev.Error.Message
	}
	return fmt.Errorf("%s", msg)
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
	return &chat.ProviderError{Provider: p.Name(), StatusCode: resp.StatusCode, Message: msg}
}
