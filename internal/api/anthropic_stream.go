package api

import (
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

// anthropicStream serves the streaming branch of POST /anthropic/v1/messages.
// It mirrors the failover loop of streamCompletion: candidates are tried in
// order and a candidate is only failed over when it errored before delivering
// any content. Canonical StreamDelta chunks are translated into Anthropic SSE
// events; the final chunk (FinishReason set) closes the open text block,
// emits any tool_use blocks, then the message_delta/message_stop tail.
func (s *Server) anthropicStream(w http.ResponseWriter, r *http.Request, sel routing.Selector, cReq chat.ChatRequest) {
	ctx := r.Context()

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

	for _, cand := range candidates {
		cReq.Model = cand.Model
		p, ok := s.deps.Providers[cand.ProviderName]
		if !ok {
			sel.RecordFailure(cand)
			continue
		}

		delivered := false
		endedInError := false
		streamErr := p.Stream(ctx, cReq, func(d chat.StreamDelta) error {
			delivered = true
			if d.FinishReason == "error" {
				endedInError = true
			}
			if d.Usage != nil {
				sw.usage = d.Usage
			}
			if err := sw.messageStart(); err != nil {
				return err
			}
			if d.Delta != "" {
				if err := sw.openTextBlock(); err != nil {
					return err
				}
				if err := sw.textDelta(d.Delta); err != nil {
					return err
				}
			}
			if d.FinishReason == "" {
				return nil
			}
			// Final chunk: close any open text block, then tool_use blocks,
			// then the message tail.
			if err := sw.closeTextBlock(); err != nil {
				return err
			}
			for _, tc := range d.ToolCalls {
				if err := sw.toolUseBlock(tc); err != nil {
					return err
				}
			}
			if err := sw.messageDelta(anthropicStopReason(d.FinishReason), d.Usage); err != nil {
				return err
			}
			return sw.messageStop()
		})

		if streamErr != nil && !delivered {
			s.deps.Logger.Warn("anthropic stream failed before content",
				"provider", cand.ProviderName, "model", cand.Model, "error", streamErr)
			sel.RecordFailure(cand)
			continue
		}

		if endedInError {
			sel.RecordFailure(cand)
		} else {
			sel.RecordSuccess(cand)
		}
		return
	}

	// Every candidate failed before delivering any content.
	_ = sw.errorEvent("all model candidates failed")
	_ = sw.messageStop()
}
