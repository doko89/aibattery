package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"math/rand"
	"net/http"
	"strings"
	"time"

	"github.com/aibattery/router/internal/chat"
	"github.com/aibattery/router/internal/routing"
)

// handleChatCompletions serves POST /v1/chat/completions. It validates the
// OpenAI wire request, resolves the virtual model name to a routing.Selector,
// then dispatches to the non-streaming or streaming orchestration loop.
func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	var req chatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body: "+err.Error(), "invalid_request_error", "")
		return
	}
	if req.Model == "" {
		writeError(w, http.StatusBadRequest, "model is required", "invalid_request_error", "")
		return
	}
	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, "messages must not be empty", "invalid_request_error", "")
		return
	}
	if req.ReasoningEffort != nil && !validReasoningEffort(*req.ReasoningEffort) {
		writeError(w, http.StatusBadRequest,
			fmt.Sprintf("invalid reasoning_effort %q: must be one of none, minimal, low, medium, high, xhigh, max", *req.ReasoningEffort),
			"invalid_request_error", "invalid_reasoning_effort")
		return
	}

	sel, err := s.deps.Registry.Select(req.Model)
	if err != nil {
		if errors.Is(err, routing.ErrModelNotFound) {
			writeError(w, http.StatusNotFound, fmt.Sprintf("model '%s' not found", req.Model), "invalid_request_error", "model_not_found")
			return
		}
		writeError(w, http.StatusInternalServerError, err.Error(), "invalid_request_error", "")
		return
	}

	cReq := buildChatRequest(req)
	if s.deps.Tools != nil {
		cReq.Tools = s.deps.Tools.List()
	}
	if req.Stream {
		s.streamCompletion(w, r, sel, cReq)
		return
	}
	s.complete(w, r, sel, cReq)
}

// validReasoningEffort reports whether v is a canonical reasoning effort
// accepted by the wire API. Values are lowercase per the OpenAI spec.
func validReasoningEffort(v string) bool {
	switch v {
	case "none", "minimal", "low", "medium", "high", "xhigh", "max":
		return true
	}
	return false
}

// buildChatRequest translates the OpenAI wire request into the canonical
// chat.ChatRequest. System-role messages are concatenated into the System
// field; the virtual model name is preserved as the initial Model value (the
// orchestration loop overwrites it with each concrete candidate model).
func buildChatRequest(req chatCompletionRequest) chat.ChatRequest {
	var system []string
	var messages []chat.Message
	for _, m := range req.Messages {
		if m.Role == "system" {
			system = append(system, m.Content)
			continue
		}
		messages = append(messages, chat.Message{Role: chat.Role(m.Role), Content: m.Content})
	}

	var maxTokens *int
	if req.MaxTokens != nil {
		v := *req.MaxTokens
		maxTokens = &v
	}

	return chat.ChatRequest{
		Model:           req.Model,
		Messages:        messages,
		System:          strings.Join(system, "\n"),
		Temperature:     req.Temperature,
		MaxTokens:       maxTokens,
		ReasoningEffort: req.ReasoningEffort,
		Stream:          req.Stream,
	}
}

// complete runs the non-streaming failover loop. It iterates the selector's
// candidates, forwarding the concrete model name to each provider, and returns
// the first successful completion as OpenAI JSON. If every candidate fails it
// responds 502.
func (s *Server) complete(w http.ResponseWriter, r *http.Request, sel routing.Selector, cReq chat.ChatRequest) {
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
		writeCompletion(w, cand.Model, resp)
		return
	}
	writeError(w, http.StatusBadGateway, "all model candidates failed", "upstream_error", "")
}

// writeCompletion serializes a canonical chat.ChatResponse into the OpenAI
// non-streaming completion shape.
func writeCompletion(w http.ResponseWriter, model string, resp chat.ChatResponse) {
	finish := resp.FinishReason
	if finish == "" {
		finish = "stop"
	}
	out := chatCompletionResponse{
		ID:      fmt.Sprintf("cmpl-%x", rand.Uint64()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []choice{{
			Index:        0,
			Message:      respMessage{Role: "assistant", Content: resp.Content, ToolCalls: toRespToolCalls(resp.ToolCalls)},
			FinishReason: finish,
		}},
		Usage: &usage{
			PromptTokens:     resp.Usage.PromptTokens,
			CompletionTokens: resp.Usage.CompletionTokens,
			TotalTokens:      resp.Usage.TotalTokens,
		},
	}
	writeJSON(w, http.StatusOK, out)
}

// toRespToolCalls translates canonical tool calls into the OpenAI wire shape.
// Arguments is JSON-encoded so it serializes as a string on the wire.
func toRespToolCalls(tcs []chat.ToolCall) []respToolCall {
	if len(tcs) == 0 {
		return nil
	}
	out := make([]respToolCall, 0, len(tcs))
	for _, tc := range tcs {
		out = append(out, respToolCall{
			ID:   tc.ID,
			Type: "function",
			Function: respToolCallFunction{
				Name:      tc.Name,
				Arguments: string(tc.Arguments),
			},
		})
	}
	return out
}

// streamCompletion runs the streaming failover loop. It writes SSE headers,
// then iterates candidates. A candidate is only failed over when it errored
// before delivering any content; once content is delivered the stream is
// final and cannot be un-sent.
func (s *Server) streamCompletion(w http.ResponseWriter, r *http.Request, sel routing.Selector, cReq chat.ChatRequest) {
	ctx := r.Context()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}

	sw := newSSEWriter(w)
	candidates := sel.Begin()
	if len(candidates) == 0 {
		sw.writeError("all model candidates are down")
		sw.writeDone()
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
			return sw.writeChunk(cand.Model, d)
		})

		if streamErr != nil && !delivered {
			s.deps.Logger.Warn("provider stream failed before content",
				"provider", cand.ProviderName, "model", cand.Model, "error", streamErr)
			sel.RecordFailure(cand)
			continue
		}

		if endedInError {
			sel.RecordFailure(cand)
		} else {
			sel.RecordSuccess(cand)
		}
		break
	}

	sw.writeDone()
}
