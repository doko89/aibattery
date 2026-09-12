package api

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"net/http"

	"github.com/aibattery/router/internal/chat"
	"github.com/aibattery/router/internal/routing"
)

// anthropicSSEWriter serializes canonical chat.StreamDelta values into
// Anthropic Messages-format SSE events. It owns the message identity and the
// block sequencing: text blocks first, then tool_use blocks, indices
// 0,1,2,... across the whole stream.
type anthropicSSEWriter struct {
	w        http.ResponseWriter
	flusher  http.Flusher
	model    string
	id       string
	started  bool // message_start emitted
	textOpen bool // a text content block is currently open
	index    int
	usage    *chat.Usage // last-seen usage, for message_start input_tokens
}

func newAnthropicSSEWriter(w http.ResponseWriter, model string) *anthropicSSEWriter {
	return &anthropicSSEWriter{
		w:       w,
		flusher: w.(http.Flusher),
		model:   model,
		id:      fmt.Sprintf("msg_%x", rand.Uint64()),
	}
}

// event writes one `event: <name>\ndata: <json>\n\n` frame and flushes.
func (a *anthropicSSEWriter) event(name string, data any) error {
	b, err := json.Marshal(data)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(a.w, "event: %s\ndata: %s\n\n", name, b); err != nil {
		return err
	}
	a.flusher.Flush()
	return nil
}

// messageStart emits message_start exactly once. input_tokens comes from the
// last-seen usage (typically 0: usage arrives on the final chunk, after the
// start event has already been sent).
func (a *anthropicSSEWriter) messageStart() error {
	if a.started {
		return nil
	}
	a.started = true
	input := 0
	if a.usage != nil {
		input = a.usage.PromptTokens
	}
	return a.event("message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id": a.id, "type": "message", "role": "assistant",
			"model": a.model, "content": []any{},
			"usage": map[string]any{"input_tokens": input, "output_tokens": 0},
		},
	})
}

// openTextBlock starts a text content block (no-op if one is already open).
func (a *anthropicSSEWriter) openTextBlock() error {
	if a.textOpen {
		return nil
	}
	a.textOpen = true
	return a.event("content_block_start", map[string]any{
		"type": "content_block_start", "index": a.index,
		"content_block": map[string]any{"type": "text", "text": ""},
	})
}

func (a *anthropicSSEWriter) textDelta(text string) error {
	return a.event("content_block_delta", map[string]any{
		"type": "content_block_delta", "index": a.index,
		"delta": map[string]any{"type": "text_delta", "text": text},
	})
}

// closeTextBlock stops the open text block and advances the index.
func (a *anthropicSSEWriter) closeTextBlock() error {
	if !a.textOpen {
		return nil
	}
	a.textOpen = false
	a.index++
	return a.event("content_block_stop", map[string]any{"type": "content_block_stop", "index": a.index - 1})
}

// toolUseBlock emits the start/delta/stop triple for one tool_use block. The
// whole arguments JSON is sent in a single input_json_delta.
func (a *anthropicSSEWriter) toolUseBlock(tc chat.ToolCall) error {
	block := map[string]any{"type": "tool_use", "id": tc.ID, "name": tc.Name, "input": map[string]any{}}
	if err := a.event("content_block_start", map[string]any{"type": "content_block_start", "index": a.index, "content_block": block}); err != nil {
		return err
	}
	if err := a.event("content_block_delta", map[string]any{
		"type": "content_block_delta", "index": a.index,
		"delta": map[string]any{"type": "input_json_delta", "partial_json": string(tc.Arguments)},
	}); err != nil {
		return err
	}
	a.index++
	return a.event("content_block_stop", map[string]any{"type": "content_block_stop", "index": a.index - 1})
}

func (a *anthropicSSEWriter) messageDelta(stopReason string, u *chat.Usage) error {
	output := 0
	if u != nil {
		output = u.CompletionTokens
	}
	return a.event("message_delta", map[string]any{
		"type":  "message_delta",
		"delta": map[string]any{"stop_reason": stopReason, "stop_sequence": nil},
		"usage": map[string]any{"output_tokens": output},
	})
}

func (a *anthropicSSEWriter) messageStop() error {
	return a.event("message_stop", map[string]any{"type": "message_stop"})
}

// errorEvent emits the Anthropic error envelope as an SSE error frame.
func (a *anthropicSSEWriter) errorEvent(message string) error {
	return a.event("error", anthropicErrorEnvelope{Type: "error", Error: anthropicErrorDetails{Type: "api_error", Message: message}})
}

// emitDelta translates one canonical StreamDelta into Anthropic SSE events:
// a message_start, incremental text deltas, then on the final chunk the tail
// (close text block, tool_use blocks, message_delta, message_stop).
func (a *anthropicSSEWriter) emitDelta(d chat.StreamDelta) error {
	if d.Usage != nil {
		a.usage = d.Usage
	}
	if err := a.messageStart(); err != nil {
		return err
	}
	if d.Delta != "" {
		if err := a.openTextBlock(); err != nil {
			return err
		}
		if err := a.textDelta(d.Delta); err != nil {
			return err
		}
	}
	if d.FinishReason == "" {
		return nil
	}
	if err := a.closeTextBlock(); err != nil {
		return err
	}
	for _, tc := range d.ToolCalls {
		if err := a.toolUseBlock(tc); err != nil {
			return err
		}
	}
	if err := a.messageDelta(anthropicStopReason(d.FinishReason), d.Usage); err != nil {
		return err
	}
	return a.messageStop()
}

// writeResponseBurst emits a completed non-streaming ChatResponse as the
// Anthropic message event sequence: message_start, a text block when content
// is present, one tool_use block per call, then the message_delta/message_stop
// tail. Used to stream the result of the internal server-tool execution loop,
// whose response was never streamed.
func (a *anthropicSSEWriter) writeResponseBurst(resp chat.ChatResponse) error {
	if resp.Usage != (chat.Usage{}) {
		a.usage = &resp.Usage
	}
	if err := a.messageStart(); err != nil {
		return err
	}
	if resp.Content != "" {
		if err := a.openTextBlock(); err != nil {
			return err
		}
		if err := a.textDelta(resp.Content); err != nil {
			return err
		}
		if err := a.closeTextBlock(); err != nil {
			return err
		}
	}
	for _, tc := range resp.ToolCalls {
		if err := a.toolUseBlock(tc); err != nil {
			return err
		}
	}
	if err := a.messageDelta(anthropicStopReason(resp.FinishReason), a.usage); err != nil {
		return err
	}
	return a.messageStop()
}

// anthropicStream serves the streaming branch of POST /anthropic/v1/messages.
// It mirrors the failover loop of streamCompletion: candidates are tried in
// order and a candidate is only failed over when it errored before delivering
// any content. The upstream stream is buffered until its final chunk so
// server-owned tool calls can be executed internally instead of being
// forwarded to a client that cannot run them.
func (s *Server) anthropicStream(w http.ResponseWriter, r *http.Request, sel routing.Selector, cReq chat.ChatRequest, clientToolNames map[string]bool) {
	ctx := r.Context()
	virtualModel := cReq.Model
	session := sessionKey(r)

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}

	sw := newAnthropicSSEWriter(w, cReq.Model)
	candidates := sel.Begin()
	if len(candidates) == 0 {
		_ = sw.errorEvent("all model candidates are down")
		_ = sw.messageStop()
		return
	}

	if s.runAnthropicCandidates(ctx, sw, sel, &cReq, candidates, clientToolNames, session, virtualModel) {
		return
	}
	// Every candidate failed before delivering any content.
	_ = sw.errorEvent("all model candidates failed")
	_ = sw.messageStop()
}

// runAnthropicCandidates tries each candidate in order. It reports whether any
// candidate delivered content (or errored after content), making the stream
// final.
func (s *Server) runAnthropicCandidates(ctx context.Context, sw *anthropicSSEWriter, sel routing.Selector, cReq *chat.ChatRequest, candidates []routing.Candidate, clientToolNames map[string]bool, session, virtualModel string) bool {
	for _, cand := range candidates {
		cReq.Model = cand.Model
		p, ok := s.deps.Providers[cand.ProviderName]
		if !ok {
			sel.RecordFailure(cand)
			continue
		}
		got := captureStream(ctx, p, *cReq)
		if got.err != nil && !got.delivered {
			s.deps.Logger.Warn("anthropic stream failed before content",
				"virtual_model", virtualModel, "provider", cand.ProviderName, "model", cand.Model, "error", got.err)
			sel.RecordFailure(cand)
			continue
		}
		s.recordAnthropicOutcome(sel, cand, got.endedInError)
		s.deps.Logger.Info("stream served",
			"virtual_model", virtualModel,
			"provider", cand.ProviderName,
			"model", cand.Model)
		s.serveAnthropicCapture(ctx, sw, p, cReq, cand, got, clientToolNames, session, virtualModel)
		return true
	}
	return false
}

// recordAnthropicOutcome records a delivered stream: an error-terminated
// stream counts as a failure, a clean one as a success.
func (s *Server) recordAnthropicOutcome(sel routing.Selector, cand routing.Candidate, endedInError bool) {
	if endedInError {
		sel.RecordFailure(cand)
		return
	}
	sel.RecordSuccess(cand)
}

// serveAnthropicCapture serves one candidate's buffered stream: server-owned
// tool calls are executed internally as an event burst, everything else is
// replayed exactly as it arrived.
func (s *Server) serveAnthropicCapture(ctx context.Context, sw *anthropicSSEWriter, p chat.Provider, cReq *chat.ChatRequest, cand routing.Candidate, got streamCapture, clientToolNames map[string]bool, session, virtualModel string) {
	if s.tryServeAnthropicToolBurst(ctx, sw, p, cReq, cand, got, clientToolNames, session, virtualModel) {
		return
	}
	// Replay the buffered deltas (plain content, client-owned or mixed
	// tool calls, or an error-terminated stream) exactly as they arrived.
	for _, d := range got.deltas {
		if err := sw.emitDelta(d); err != nil {
			s.deps.Logger.Warn("anthropic stream write failed",
				"virtual_model", virtualModel, "provider", cand.ProviderName, "model", cand.Model, "error", err)
			break
		}
	}
}

// tryServeAnthropicToolBurst executes server-owned tool calls internally and
// emits the result as an Anthropic event burst. It reports whether the burst
// path was taken (so the caller must not replay the buffered deltas).
func (s *Server) tryServeAnthropicToolBurst(ctx context.Context, sw *anthropicSSEWriter, p chat.Provider, cReq *chat.ChatRequest, cand routing.Candidate, got streamCapture, clientToolNames map[string]bool, session, virtualModel string) bool {
	if got.err != nil || got.endedInError {
		return false
	}
	// The stream completed cleanly: the buffered final chunk decides
	// whether the tool calls belong to the server and must be executed
	// internally instead of being forwarded.
	// ponytail: streaming is buffered until the final chunk to decide tool ownership; a lookahead could stream text deltas but risks leaking server tool_calls
	resp := responseFromDeltas(got.deltas)
	s.rememberReasoning(session, &resp)
	if !s.allServerToolCalls(resp.ToolCalls, clientToolNames) {
		return false
	}
	final, err := s.executeServerTools(ctx, p, cReq, &resp, clientToolNames, session)
	if err != nil {
		s.deps.Logger.Warn("anthropic provider completion failed during tool loop",
			"virtual_model", virtualModel, "provider", cand.ProviderName, "model", cand.Model, "error", err)
		_ = sw.errorEvent("tool execution failed: " + err.Error())
		_ = sw.messageStop()
		return true
	}
	if err := sw.writeResponseBurst(*final); err != nil {
		s.deps.Logger.Warn("anthropic stream write failed",
			"virtual_model", virtualModel, "provider", cand.ProviderName, "model", cand.Model, "error", err)
	}
	return true
}
