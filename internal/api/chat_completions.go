package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math/rand"
	"net/http"
	"strings"
	"sync"
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

	cReq, err := buildChatRequest(req)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body: "+err.Error(), "invalid_request_error", "")
		return
	}
	clientToolNames := clientToolNamesSet(cReq.Tools)
	if s.deps.Tools != nil {
		cReq.Tools = mergeTools(cReq.Tools, s.deps.Tools.List())
	}
	if req.Stream {
		s.streamCompletion(w, r, sel, cReq, clientToolNames)
		return
	}
	s.complete(w, r, sel, cReq, clientToolNames)
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
func buildChatRequest(req chatCompletionRequest) (chat.ChatRequest, error) {
	var system []string
	var messages []chat.Message
	for _, m := range req.Messages {
		content, err := anthropicContentText(m.Content)
		if err != nil {
			return chat.ChatRequest{}, err
		}
		if m.Role == "system" {
			system = append(system, content)
			continue
		}
		msg := chat.Message{Role: chat.Role(m.Role), Content: content, ToolCallID: m.ToolCallID, ReasoningContent: m.ReasoningContent}
		if len(m.ToolCalls) > 0 {
			msg.ToolCalls = make([]chat.ToolCall, 0, len(m.ToolCalls))
			for _, tc := range m.ToolCalls {
				call := chat.ToolCall{ID: tc.ID, Name: tc.Function.Name}
				if tc.Function.Arguments != "" && json.Valid([]byte(tc.Function.Arguments)) {
					call.Arguments = json.RawMessage(tc.Function.Arguments)
				}
				msg.ToolCalls = append(msg.ToolCalls, call)
			}
		}
		// Clients (OpenCode/OpenAI SDKs) never send reasoning_content back; echo
		// the value captured from the upstream response that produced these tool
		// calls, keyed by the first call's ID, or thinking-mode upstreams reject
		// the turn (DeepSeek 400 "reasoning_content must be passed back").
		if m.Role == "assistant" && len(msg.ToolCalls) > 0 && msg.ReasoningContent == "" {
			msg.ReasoningContent = reasoningByCallID.lookup(msg.ToolCalls[0].ID)
		}
		messages = append(messages, msg)
	}

	var maxTokens *int
	if req.MaxTokens != nil {
		v := *req.MaxTokens
		maxTokens = &v
	}

	cReq := chat.ChatRequest{
		Model:           req.Model,
		Messages:        messages,
		System:          strings.Join(system, "\n"),
		Temperature:     req.Temperature,
		MaxTokens:       maxTokens,
		ReasoningEffort: req.ReasoningEffort,
		Stream:          req.Stream,
	}
	if len(req.Tools) > 0 {
		cReq.Tools = make([]chat.Tool, 0, len(req.Tools))
		for _, t := range req.Tools {
			cReq.Tools = append(cReq.Tools, chat.Tool{
				Name:        t.Function.Name,
				Description: t.Function.Description,
				InputSchema: t.Function.Parameters,
			})
		}
	}
	return cReq, nil
}

// mergeTools appends the server tools not already present by Name to the
// client's tool list. A client tool with the same name wins: its definition is
// kept and the server duplicate skipped, so the client keeps executing the
// tools it declares itself.
func mergeTools(client, server []chat.Tool) []chat.Tool {
	if len(server) == 0 {
		return client
	}
	have := make(map[string]bool, len(client))
	for _, t := range client {
		have[t.Name] = true
	}
	out := make([]chat.Tool, 0, len(client)+len(server))
	out = append(out, client...)
	for _, t := range server {
		if !have[t.Name] {
			out = append(out, t)
		}
	}
	return out
}

// clientToolNamesSet indexes the names of the tools the client itself sent, so
// server-owned tool detection can exclude names the client executes.
func clientToolNamesSet(tools []chat.Tool) map[string]bool {
	names := make(map[string]bool, len(tools))
	for _, t := range tools {
		names[t.Name] = true
	}
	return names
}

// serverOwned reports whether a tool call targets a server-registered tool
// that the client did not itself declare. Calls to tools the client provided
// (or that are absent from the registry) are the client's to execute.
func (s *Server) serverOwned(tc chat.ToolCall, clientToolNames map[string]bool) bool {
	return s.deps.Tools != nil && s.deps.Tools.Has(tc.Name) && !clientToolNames[tc.Name]
}

// allServerToolCalls reports whether calls is non-empty and every call is
// server-owned. Mixed batches (client and server calls together) are passed
// through to the client untouched, so they never trigger the internal loop.
func (s *Server) allServerToolCalls(calls []chat.ToolCall, clientToolNames map[string]bool) bool {
	if len(calls) == 0 {
		return false
	}
	for _, tc := range calls {
		if !s.serverOwned(tc, clientToolNames) {
			return false
		}
	}
	return true
}

// maxServerToolIterations caps the internal tool-execution loop.
// ponytail: hard cap 5, raise if models need longer tool chains
const maxServerToolIterations = 5

// maxAttempts is how many times a candidate is tried in total for non-429
// errors before failing over. 429 is never retried — it goes straight to
// cooldown + failover.
const maxAttempts = 3

// reasoningCache remembers reasoning_content per tool_call_id so
// thinking-mode upstreams get it echoed on the next client turn.
// ponytail: single-instance in-memory map, cap 4096, reset when full;
// a TTL/eviction policy only matters at multi-instance scale.
type reasoningCache struct {
	mu sync.Mutex
	m  map[string]string
}

var reasoningByCallID = &reasoningCache{m: make(map[string]string)}

func (c *reasoningCache) remember(calls []chat.ToolCall, reasoning string) {
	if reasoning == "" {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(c.m) >= 4096 {
		c.m = make(map[string]string)
	}
	for _, tc := range calls {
		if tc.ID != "" {
			c.m[tc.ID] = reasoning
		}
	}
}

func (c *reasoningCache) lookup(callID string) string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.m[callID]
}

// rememberReasoning caches a response's reasoning_content keyed by the IDs of
// the tool calls it produced, so a later turn (client-driven or the internal
// tool loop) can echo it back to thinking-mode upstreams.
func (s *Server) rememberReasoning(resp *chat.ChatResponse) {
	if resp == nil || resp.ReasoningContent == "" {
		return
	}
	reasoningByCallID.remember(resp.ToolCalls, resp.ReasoningContent)
}

// executeServerTools runs the internal tool-execution loop for a completed
// response. When every tool call is server-owned, each call is executed
// against the tool registry and the request extended with the assistant
// tool-call message and its tool result, then the same provider is called
// again with the extended conversation. It returns the first response that is
// not all server-owned (client-owned or mixed batches pass through to the
// client untouched), or resp unchanged when the iteration cap is exhausted.
func (s *Server) executeServerTools(ctx context.Context, p chat.Provider, cReq *chat.ChatRequest, resp *chat.ChatResponse, clientToolNames map[string]bool) (*chat.ChatResponse, error) {
	for i := 0; i < maxServerToolIterations; i++ {
		if !s.allServerToolCalls(resp.ToolCalls, clientToolNames) {
			return resp, nil
		}
		for _, tc := range resp.ToolCalls {
			out, err := s.deps.Tools.Call(ctx, tc.Name, tc.Arguments)
			if err != nil {
				out = err.Error()
			}
			cReq.Messages = append(cReq.Messages,
				chat.Message{Role: chat.RoleAssistant, ReasoningContent: resp.ReasoningContent, ToolCalls: []chat.ToolCall{tc}},
				chat.Message{Role: chat.RoleTool, ToolCallID: tc.ID, Content: out},
			)
		}
		// The loop reuses the client's request, which may have Stream set
		// (streaming client); the continuation must be a plain completion or
		// the upstream answers with SSE and Complete fails to decode it.
		cReq.Stream = false
		next, err := p.Complete(ctx, *cReq)
		if err != nil {
			return nil, err
		}
		resp = &next
		s.rememberReasoning(resp)
	}
	s.rememberReasoning(resp)
	return resp, nil
}

// responseFromDeltas collapses a fully-buffered stream into the canonical
// ChatResponse the final chunk implies: concatenated text, and the final
// chunk's finish reason, tool calls and usage.
func responseFromDeltas(deltas []chat.StreamDelta) chat.ChatResponse {
	var resp chat.ChatResponse
	for _, d := range deltas {
		resp.Content += d.Delta
		if d.FinishReason != "" {
			resp.FinishReason = d.FinishReason
			resp.ToolCalls = d.ToolCalls
			resp.ReasoningContent = d.ReasoningContent
			if d.Usage != nil {
				resp.Usage = *d.Usage
			}
		}
	}
	return resp
}

// complete runs the non-streaming failover loop. It iterates the selector's
// candidates, forwarding the concrete model name to each provider, and returns
// the first successful completion as OpenAI JSON. Server-owned tool calls in a
// completion are executed internally by the router. If every candidate fails it
// responds 502.
func (s *Server) complete(w http.ResponseWriter, r *http.Request, sel routing.Selector, cReq chat.ChatRequest, clientToolNames map[string]bool) {
	ctx := r.Context()
	virtualModel := cReq.Model
	for _, cand := range sel.Begin() {
		cReq.Model = cand.Model
		p, ok := s.deps.Providers[cand.ProviderName]
		if !ok {
			sel.RecordFailure(cand)
			continue
		}
		resp, err := p.Complete(ctx, cReq)
		if err != nil {
			// 429 is never retried: it goes straight to cooldown + failover.
			if chat.IsRateLimit(err) {
				s.deps.Logger.Warn("provider rate limited",
					"virtual_model", virtualModel, "provider", cand.ProviderName, "model", cand.Model, "error", err)
				sel.RecordFailure(cand)
				continue
			}
			// Non-429: retry the same candidate up to maxAttempts total tries
			// before failing over. No cooldown — it may recover quickly.
			for attempt := 1; attempt < maxAttempts; attempt++ {
				s.deps.Logger.Warn("provider completion failed, retrying",
					"virtual_model", virtualModel, "provider", cand.ProviderName, "model", cand.Model, "attempt", attempt, "error", err)
				resp, err = p.Complete(ctx, cReq)
				if err == nil {
					break
				}
			}
			if err != nil {
				s.deps.Logger.Warn("provider completion failed after retries",
					"virtual_model", virtualModel, "provider", cand.ProviderName, "model", cand.Model, "error", err)
				continue
			}
		}
		s.rememberReasoning(&resp)
		final, err := s.executeServerTools(ctx, p, &cReq, &resp, clientToolNames)
		if err != nil {
			s.deps.Logger.Warn("provider completion failed during tool loop",
				"virtual_model", virtualModel, "provider", cand.ProviderName, "model", cand.Model, "error", err)
			sel.RecordFailure(cand)
			continue
		}
		sel.RecordSuccess(cand)
		s.deps.Logger.Info("completion served",
			"virtual_model", virtualModel,
			"provider", cand.ProviderName,
			"model", cand.Model)
		writeCompletion(w, cand.Model, *final)
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
// final and cannot be un-sent. The upstream stream is buffered until its final
// chunk so server-owned tool calls can be executed internally instead of being
// forwarded to a client that cannot run them.
func (s *Server) streamCompletion(w http.ResponseWriter, r *http.Request, sel routing.Selector, cReq chat.ChatRequest, clientToolNames map[string]bool) {
	ctx := r.Context()
	virtualModel := cReq.Model

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

	succeeded := false
	for _, cand := range candidates {
		cReq.Model = cand.Model
		p, ok := s.deps.Providers[cand.ProviderName]
		if !ok {
			sel.RecordFailure(cand)
			continue
		}

		delivered := false
		endedInError := false
		var deltas []chat.StreamDelta
		streamErr := p.Stream(ctx, cReq, func(d chat.StreamDelta) error {
			delivered = true
			if d.FinishReason == "error" {
				endedInError = true
			}
			deltas = append(deltas, d)
			return nil
		})

		if streamErr != nil && !delivered {
			// 429 is never retried: straight to cooldown + failover.
			if chat.IsRateLimit(streamErr) {
				s.deps.Logger.Warn("provider stream rate limited",
					"virtual_model", virtualModel, "provider", cand.ProviderName, "model", cand.Model, "error", streamErr)
				sel.RecordFailure(cand)
				continue
			}
			// Non-429 pre-content failure: retry the same candidate up to
			// maxAttempts total tries before failing over.
			for attempt := 1; attempt < maxAttempts && streamErr != nil && !delivered; attempt++ {
				s.deps.Logger.Warn("provider stream failed before content, retrying",
					"virtual_model", virtualModel, "provider", cand.ProviderName, "model", cand.Model, "attempt", attempt, "error", streamErr)
				delivered = false
				endedInError = false
				deltas = nil
				streamErr = p.Stream(ctx, cReq, func(d chat.StreamDelta) error {
					delivered = true
					if d.FinishReason == "error" {
						endedInError = true
					}
					deltas = append(deltas, d)
					return nil
				})
			}
			if streamErr != nil && !delivered {
				s.deps.Logger.Warn("provider stream failed before content after retries",
					"virtual_model", virtualModel, "provider", cand.ProviderName, "model", cand.Model, "error", streamErr)
				continue
			}
		}

		if !endedInError {
			sel.RecordSuccess(cand)
		}
		// A candidate delivered content (or errored after content): the stream
		// is final and cannot be failed over, so the loop must not continue.
		succeeded = true
		s.deps.Logger.Info("stream served",
			"virtual_model", virtualModel,
			"provider", cand.ProviderName,
			"model", cand.Model)

		if streamErr == nil && !endedInError {
			// The stream completed cleanly: the buffered final chunk decides
			// whether the tool calls belong to the server and must be executed
			// internally instead of being forwarded.
			// ponytail: streaming is buffered until the final chunk to decide tool ownership; a lookahead could stream text deltas but risks leaking server tool_calls
			resp := responseFromDeltas(deltas)
			s.rememberReasoning(&resp)
			if s.allServerToolCalls(resp.ToolCalls, clientToolNames) {
				final, err := s.executeServerTools(ctx, p, &cReq, &resp, clientToolNames)
				if err != nil {
					s.deps.Logger.Warn("provider completion failed during tool loop",
						"virtual_model", virtualModel, "provider", cand.ProviderName, "model", cand.Model, "error", err)
					_ = sw.writeError("tool execution failed: " + err.Error())
				} else if err := sw.writeResponseBurst(cand.Model, *final); err != nil {
					s.deps.Logger.Warn("stream write failed",
						"virtual_model", virtualModel, "provider", cand.ProviderName, "model", cand.Model, "error", err)
				}
				break
			}
		}

		// Replay the buffered deltas (plain content, client-owned or mixed
		// tool calls, or an error-terminated stream) exactly as they arrived.
		for _, d := range deltas {
			if err := sw.writeChunk(cand.Model, d); err != nil {
				s.deps.Logger.Warn("stream write failed",
					"virtual_model", virtualModel, "provider", cand.ProviderName, "model", cand.Model, "error", err)
				break
			}
		}
		break
	}

	// Every candidate failed before delivering any content: surface an error
	// frame instead of a misleading empty 200 stream.
	if !succeeded {
		sw.writeError("all model candidates failed")
	}
	sw.writeDone()
}
