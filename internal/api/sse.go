package api

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"net/http"
	"time"

	"github.com/aibattery/router/internal/chat"
)

// sseWriter normalizes canonical chat.StreamDelta values into OpenAI
// `chat.completion.chunk` SSE frames. It owns the stream identity (id,
// created) and the first-chunk role injection.
type sseWriter struct {
	w       http.ResponseWriter
	flusher http.Flusher
	id      string
	created int64
	first   bool
}

// newSSEWriter builds an SSE writer for a fresh stream.
func newSSEWriter(w http.ResponseWriter) *sseWriter {
	return &sseWriter{
		w:       w,
		flusher: w.(http.Flusher),
		id:      fmt.Sprintf("cmpl-%x", rand.Uint64()),
		created: time.Now().Unix(),
		first:   true,
	}
}

// writeChunk serializes a single chat.StreamDelta into an OpenAI
// `chat.completion.chunk` frame and flushes it. The first chunk carries
// delta.role="assistant"; finish_reason is only set on the final chunk; usage
// is attached to the final chunk when present.
func (s *sseWriter) writeChunk(model string, d chat.StreamDelta) error {
	delta := map[string]any{}
	if s.first {
		delta["role"] = "assistant"
		s.first = false
	}
	if d.Delta != "" {
		delta["content"] = d.Delta
	}
	if len(d.ToolCalls) > 0 {
		calls := make([]map[string]any, 0, len(d.ToolCalls))
		for _, tc := range d.ToolCalls {
			calls = append(calls, map[string]any{
				"id":   tc.ID,
				"type": "function",
				"function": map[string]any{
					"name":      tc.Name,
					"arguments": string(tc.Arguments), // JSON-encoded string per OpenAI wire
				},
			})
		}
		delta["tool_calls"] = calls
	}

	var finishReason any
	if d.FinishReason != "" {
		finishReason = d.FinishReason
	}

	chunk := map[string]any{
		"id":      s.id,
		"object":  "chat.completion.chunk",
		"created": s.created,
		"model":   model,
		"choices": []any{map[string]any{
			"index":         0,
			"delta":         delta,
			"finish_reason": finishReason,
		}},
	}
	if d.Usage != nil {
		chunk["usage"] = map[string]any{
			"prompt_tokens":     d.Usage.PromptTokens,
			"completion_tokens": d.Usage.CompletionTokens,
			"total_tokens":      d.Usage.TotalTokens,
		}
	}

	b, err := json.Marshal(chunk)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(s.w, "data: %s\n\n", b); err != nil {
		return err
	}
	s.flusher.Flush()
	return nil
}

// writeError emits an SSE error frame (used when no candidate could be
// selected, e.g. all models in cooldown).
func (s *sseWriter) writeError(message string) error {
	errResp := errorResponse{Error: apiError{Message: message, Type: "upstream_error"}}
	b, err := json.Marshal(errResp)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(s.w, "data: %s\n\n", b); err != nil {
		return err
	}
	s.flusher.Flush()
	return nil
}

// writeDone terminates the SSE stream.
func (s *sseWriter) writeDone() error {
	if _, err := fmt.Fprintf(s.w, "data: [DONE]\n\n"); err != nil {
		return err
	}
	s.flusher.Flush()
	return nil
}