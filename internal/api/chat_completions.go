package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math/rand"
	"net/http"
	"os"
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

	cReq, err := s.buildChatRequest(req, sessionKey(r))
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
func (s *Server) buildChatRequest(req chatCompletionRequest, session string) (chat.ChatRequest, error) {
	system, messages, err := s.buildChatMessages(req.Messages, session)
	if err != nil {
		return chat.ChatRequest{}, err
	}

	cReq := chat.ChatRequest{
		Model:           req.Model,
		Messages:        messages,
		System:          system,
		Temperature:     req.Temperature,
		MaxTokens:       copyMaxTokens(req.MaxTokens),
		ReasoningEffort: req.ReasoningEffort,
		Stream:          req.Stream,
	}
	cReq.Tools = convertWireTools(req.Tools)
	return cReq, nil
}

// buildChatMessages converts wire messages into the system prompt and the
// canonical message list.
func (s *Server) buildChatMessages(wire []chatMessage, session string) (string, []chat.Message, error) {
	var system []string
	var messages []chat.Message
	for _, m := range wire {
		msg, sysText, isSystem, err := s.convertWireMessage(m, session)
		if err != nil {
			return "", nil, err
		}
		if isSystem {
			system = append(system, sysText)
			continue
		}
		messages = append(messages, msg)
	}
	return strings.Join(system, "\n"), messages, nil
}

// convertWireMessage translates one wire message. System-role messages return
// isSystem with their text; all other roles return the canonical message.
func (s *Server) convertWireMessage(m chatMessage, session string) (chat.Message, string, bool, error) {
	content, err := anthropicContentText(m.Content)
	if err != nil {
		return chat.Message{}, "", false, err
	}
	if m.Role == "system" {
		return chat.Message{}, content, true, nil
	}
	msg := chat.Message{Role: chat.Role(m.Role), Content: content, ToolCallID: m.ToolCallID, ReasoningContent: m.ReasoningContent}
	msg.ToolCalls = toChatToolCalls(m.ToolCalls)
	s.fillReasoningEcho(session, &msg)
	return msg, "", false, nil
}

// toChatToolCalls translates wire tool calls into canonical calls. Arguments
// is kept only when it is a non-empty valid JSON document.
func toChatToolCalls(tcs []wireMessageToolCall) []chat.ToolCall {
	if len(tcs) == 0 {
		return nil
	}
	out := make([]chat.ToolCall, 0, len(tcs))
	for _, tc := range tcs {
		call := chat.ToolCall{ID: tc.ID, Name: tc.Function.Name}
		if isJSONArguments(tc.Function.Arguments) {
			call.Arguments = json.RawMessage(tc.Function.Arguments)
		}
		out = append(out, call)
	}
	return out
}

// isJSONArguments reports whether s is a non-empty valid JSON document.
func isJSONArguments(s string) bool {
	if s == "" {
		return false
	}
	return json.Valid([]byte(s))
}

// fillReasoningEcho echoes cached reasoning_content onto an assistant
// tool-call turn that carries none. Clients (OpenCode/OpenAI SDKs) never send
// reasoning_content back; the value captured from the upstream response that
// produced these tool calls is echoed, keyed by the first call's ID, or
// thinking-mode upstreams reject the turn (DeepSeek 400
// "reasoning_content must be passed back"). When the cache has nothing (calls
// produced by a non-thinking provider, e.g. GLM without thinking enabled), a
// placeholder keeps the turn routable.
func (s *Server) fillReasoningEcho(session string, msg *chat.Message) {
	if msg.Role != chat.Role("assistant") {
		return
	}
	if len(msg.ToolCalls) == 0 {
		return
	}
	if msg.ReasoningContent != "" {
		return
	}
	id := msg.ToolCalls[0].ID
	if r := reasoningByCallID.lookup(session, id); r != "" {
		msg.ReasoningContent = r
		s.deps.Logger.Debug("reasoning echo",
			"session", session, "call_id", id, "found", true)
		return
	}
	msg.ReasoningContent = reasoningPlaceholder
	s.deps.Logger.Debug("reasoning echo",
		"session", session, "call_id", id, "found", false, "injected", true)
}

// copyMaxTokens copies an optional max-tokens value.
func copyMaxTokens(in *int) *int {
	if in == nil {
		return nil
	}
	v := *in
	return &v
}

// convertWireTools translates wire tool definitions into canonical tools.
func convertWireTools(in []wireTool) []chat.Tool {
	if len(in) == 0 {
		return nil
	}
	out := make([]chat.Tool, 0, len(in))
	for _, t := range in {
		out = append(out, chat.Tool{
			Name:        t.Function.Name,
			Description: t.Function.Description,
			InputSchema: t.Function.Parameters,
		})
	}
	return out
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

// reasoningPlaceholder is injected as the assistant message's reasoning_content
// when a tool-call turn arrives with no captured reasoning and the client sent
// none. Thinking-mode upstreams (DeepSeek reasoner, GLM thinking) hard-reject
// tool-call turns that omit reasoning_content (400 "must be passed back"), so a
// minimal non-empty marker keeps the turn routable across a failover chain that
// mixes thinking and non-thinking providers. It is never cached as real
// reasoning, only placed on the wire for the current request.
const reasoningPlaceholder = "[reasoning omitted by client]"

// reasoningCache remembers reasoning_content per (session, tool_call_id) so
// thinking-mode upstreams get it echoed on the next client turn. Entries are
// keyed by session first so concurrent conversations never leak each other's
// reasoning; tool_call_id then disambiguates within a session. When path is
// set the cache is persisted to disk (atomic write) so it survives restarts;
// a TTL/eviction policy only matters at multi-instance scale.
type reasoningCache struct {
	mu   sync.Mutex
	path string
	m    map[string]map[string]string
}

// newReasoningCache returns a cache optionally backed by the file at path
// (empty path = in-memory only). Existing entries are loaded on construction.
func newReasoningCache(path string) *reasoningCache {
	c := &reasoningCache{path: path, m: make(map[string]map[string]string)}
	if path == "" {
		return c
	}
	b, err := os.ReadFile(path)
	if err == nil {
		_ = json.Unmarshal(b, &c.m)
	}
	return c
}

// maxReasoningEntries caps total cached entries; the cache resets when full to
// bound memory and disk growth.
const maxReasoningEntries = 4096

// save persists the cache to path with an atomic write (temp file + rename).
// The caller must hold mu.
func (c *reasoningCache) save() {
	if c.path == "" {
		return
	}
	b, err := json.Marshal(c.m)
	if err != nil {
		return
	}
	tmp := c.path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o600); err != nil {
		return
	}
	_ = os.Rename(tmp, c.path)
}

var reasoningByCallID = newReasoningCache("")

func (c *reasoningCache) remember(session string, calls []chat.ToolCall, reasoning string) {
	if reasoning == "" {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	total := 0
	per, ok := c.m[session]
	if !ok {
		per = make(map[string]string)
		c.m[session] = per
	}
	for _, tc := range calls {
		if tc.ID != "" {
			per[tc.ID] = reasoning
		}
	}
	for _, s := range c.m {
		total += len(s)
	}
	if total > maxReasoningEntries {
		c.m = make(map[string]map[string]string)
		c.m[session] = per
	}
	c.save()
}

func (c *reasoningCache) lookup(session, callID string) string {
	c.mu.Lock()
	defer c.mu.Unlock()
	if per, ok := c.m[session]; ok {
		return per[callID]
	}
	return ""
}

// sessionKey extracts the caller's conversation identifier used to isolate the
// reasoning cache per session. An empty key means the global bucket — safe
// because upstream tool_call IDs are globally unique.
func sessionKey(r *http.Request) string {
	return r.Header.Get("X-Session-ID")
}

// rememberReasoning caches a response's reasoning_content keyed by session and
// the IDs of the tool calls it produced, so a later turn (client-driven or the
// internal tool loop) can echo it back to thinking-mode upstreams.
func (s *Server) rememberReasoning(session string, resp *chat.ChatResponse) {
	if resp == nil || resp.ReasoningContent == "" {
		// A served response with tool calls but no reasoning means the
		// producing provider is not in thinking mode; the echo cache stays
		// empty for those call IDs and the placeholder path will cover them.
		if resp != nil && len(resp.ToolCalls) > 0 {
			s.deps.Logger.Debug("no reasoning to cache",
				"session", session, "calls", len(resp.ToolCalls))
		}
		return
	}
	reasoningByCallID.remember(session, resp.ToolCalls, resp.ReasoningContent)
	s.deps.Logger.Debug("reasoning cached",
		"session", session, "calls", len(resp.ToolCalls), "len", len(resp.ReasoningContent))
}

// executeServerTools runs the internal tool-execution loop for a completed
// response. When every tool call is server-owned, each call is executed
// against the tool registry and the request extended with the assistant
// tool-call message and its tool result, then the same provider is called
// again with the extended conversation. It returns the first response that is
// not all server-owned (client-owned or mixed batches pass through to the
// client untouched), or resp unchanged when the iteration cap is exhausted.
func (s *Server) executeServerTools(ctx context.Context, p chat.Provider, cReq *chat.ChatRequest, resp *chat.ChatResponse, clientToolNames map[string]bool, session string) (*chat.ChatResponse, error) {
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
		s.rememberReasoning(session, resp)
	}
	s.rememberReasoning(session, resp)
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
	session := sessionKey(r)
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
			// 4xx (permanent, e.g. invalid request) fails over immediately:
			// retrying a 400/401/403/404 can never succeed.
			if !chat.IsRetryable(err) {
				s.deps.Logger.Warn("provider completion failed (permanent)",
					"virtual_model", virtualModel, "provider", cand.ProviderName, "model", cand.Model, "error", err)
				continue
			}
			// Transient (5xx, timeout, connection): retry the same candidate up
			// to maxAttempts total tries before failing over. No cooldown.
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
		s.rememberReasoning(session, &resp)
		final, err := s.executeServerTools(ctx, p, &cReq, &resp, clientToolNames, session)
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
	session := sessionKey(r)

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

	if s.runStreamCandidates(ctx, sw, sel, &cReq, candidates, clientToolNames, session, virtualModel) {
		sw.writeDone()
		return
	}
	// Every candidate failed before delivering any content: surface an error
	// frame instead of a misleading empty 200 stream.
	sw.writeError("all model candidates failed")
	sw.writeDone()
}

// streamCapture is one buffered upstream stream attempt.
type streamCapture struct {
	deltas       []chat.StreamDelta
	delivered    bool
	endedInError bool
	err          error
}

// captureStream buffers one upstream stream attempt.
func captureStream(ctx context.Context, p chat.Provider, cReq chat.ChatRequest) streamCapture {
	var cap streamCapture
	cap.err = p.Stream(ctx, cReq, func(d chat.StreamDelta) error {
		cap.delivered = true
		if d.FinishReason == "error" {
			cap.endedInError = true
		}
		cap.deltas = append(cap.deltas, d)
		return nil
	})
	return cap
}

// runStreamCandidates tries each candidate in order. It reports whether any
// candidate delivered content (or errored after content), making the stream
// final.
func (s *Server) runStreamCandidates(ctx context.Context, sw *sseWriter, sel routing.Selector, cReq *chat.ChatRequest, candidates []routing.Candidate, clientToolNames map[string]bool, session, virtualModel string) bool {
	for _, cand := range candidates {
		cReq.Model = cand.Model
		p, ok := s.deps.Providers[cand.ProviderName]
		if !ok {
			sel.RecordFailure(cand)
			continue
		}
		cap, serve := s.streamWithRetry(ctx, p, *cReq, cand, virtualModel, sel)
		if !serve {
			continue
		}
		if !cap.endedInError {
			sel.RecordSuccess(cand)
		}
		// A candidate delivered content (or errored after content): the stream
		// is final and cannot be failed over, so the loop must not continue.
		s.deps.Logger.Info("stream served",
			"virtual_model", virtualModel,
			"provider", cand.ProviderName,
			"model", cand.Model)
		s.serveStreamCapture(ctx, sw, p, cReq, cand, cap, clientToolNames, session, virtualModel)
		return true
	}
	return false
}

// streamWithRetry captures one candidate's stream, retrying transient
// pre-content failures on the same candidate. It reports serve=false when the
// candidate failed before delivering anything and the loop must fail over.
func (s *Server) streamWithRetry(ctx context.Context, p chat.Provider, cReq chat.ChatRequest, cand routing.Candidate, virtualModel string, sel routing.Selector) (streamCapture, bool) {
	cap := captureStream(ctx, p, cReq)
	if cap.err == nil || cap.delivered {
		return cap, true
	}
	// 429 is never retried: straight to cooldown + failover.
	if chat.IsRateLimit(cap.err) {
		s.deps.Logger.Warn("provider stream rate limited",
			"virtual_model", virtualModel, "provider", cand.ProviderName, "model", cand.Model, "error", cap.err)
		sel.RecordFailure(cand)
		return cap, false
	}
	// 4xx (permanent) fails over immediately — retrying can never
	// succeed; only 5xx/timeout/connection (transient) is retried.
	if !chat.IsRetryable(cap.err) {
		s.deps.Logger.Warn("provider stream failed before content (permanent)",
			"virtual_model", virtualModel, "provider", cand.ProviderName, "model", cand.Model, "error", cap.err)
		return cap, false
	}
	// Transient pre-content failure: retry the same candidate up to
	// maxAttempts total tries before failing over.
	cap = s.retryStreamAfterFailure(ctx, p, cReq, cap, cand, virtualModel)
	if cap.err != nil && !cap.delivered {
		s.deps.Logger.Warn("provider stream failed before content after retries",
			"virtual_model", virtualModel, "provider", cand.ProviderName, "model", cand.Model, "error", cap.err)
		return cap, false
	}
	return cap, true
}

// retryStreamAfterFailure retries a transient pre-content stream failure on
// the same candidate up to maxAttempts total tries.
func (s *Server) retryStreamAfterFailure(ctx context.Context, p chat.Provider, cReq chat.ChatRequest, cap streamCapture, cand routing.Candidate, virtualModel string) streamCapture {
	for attempt := 1; attempt < maxAttempts; attempt++ {
		if cap.err == nil || cap.delivered {
			return cap
		}
		s.deps.Logger.Warn("provider stream failed before content, retrying",
			"virtual_model", virtualModel, "provider", cand.ProviderName, "model", cand.Model, "attempt", attempt, "error", cap.err)
		cap = captureStream(ctx, p, cReq)
	}
	return cap
}

// serveStreamCapture serves one candidate's buffered stream: server-owned tool
// calls are executed internally as an SSE burst, everything else is replayed
// exactly as it arrived.
func (s *Server) serveStreamCapture(ctx context.Context, sw *sseWriter, p chat.Provider, cReq *chat.ChatRequest, cand routing.Candidate, cap streamCapture, clientToolNames map[string]bool, session, virtualModel string) {
	if s.tryServeToolBurst(ctx, sw, p, cReq, cand, cap, clientToolNames, session, virtualModel) {
		return
	}
	// Replay the buffered deltas (plain content, client-owned or mixed
	// tool calls, or an error-terminated stream) exactly as they arrived.
	s.replayStreamDeltas(sw, cand, cap.deltas, virtualModel)
}

// tryServeToolBurst executes server-owned tool calls internally and emits the
// result as an SSE burst. It reports whether the burst path was taken (so the
// caller must not replay the buffered deltas).
func (s *Server) tryServeToolBurst(ctx context.Context, sw *sseWriter, p chat.Provider, cReq *chat.ChatRequest, cand routing.Candidate, cap streamCapture, clientToolNames map[string]bool, session, virtualModel string) bool {
	if cap.err != nil || cap.endedInError {
		return false
	}
	// The stream completed cleanly: the buffered final chunk decides
	// whether the tool calls belong to the server and must be executed
	// internally instead of being forwarded.
	// ponytail: streaming is buffered until the final chunk to decide tool ownership; a lookahead could stream text deltas but risks leaking server tool_calls
	resp := responseFromDeltas(cap.deltas)
	s.rememberReasoning(session, &resp)
	if !s.allServerToolCalls(resp.ToolCalls, clientToolNames) {
		return false
	}
	final, err := s.executeServerTools(ctx, p, cReq, &resp, clientToolNames, session)
	if err != nil {
		s.deps.Logger.Warn("provider completion failed during tool loop",
			"virtual_model", virtualModel, "provider", cand.ProviderName, "model", cand.Model, "error", err)
		_ = sw.writeError("tool execution failed: " + err.Error())
		return true
	}
	if err := sw.writeResponseBurst(cand.Model, *final); err != nil {
		s.deps.Logger.Warn("stream write failed",
			"virtual_model", virtualModel, "provider", cand.ProviderName, "model", cand.Model, "error", err)
	}
	return true
}

// replayStreamDeltas writes each buffered delta as an SSE chunk in order.
func (s *Server) replayStreamDeltas(sw *sseWriter, cand routing.Candidate, deltas []chat.StreamDelta, virtualModel string) {
	for _, d := range deltas {
		if err := sw.writeChunk(cand.Model, d); err != nil {
			s.deps.Logger.Warn("stream write failed",
				"virtual_model", virtualModel, "provider", cand.ProviderName, "model", cand.Model, "error", err)
			break
		}
	}
}
