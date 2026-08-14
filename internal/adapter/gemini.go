// Package adapter implements the chat.Provider domain port for concrete
// provider providers. Each provider translates the canonical ChatRequest into
// the provider's native wire format and maps the provider's response back into
// the canonical ChatResponse / StreamDelta types.
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

// geminiProvider adapts chat.Provider for Google Gemini's REST
// generateContent / streamGenerateContent API.
type geminiProvider struct {
	baseURL    string
	apiKey     string
	httpClient *http.Client
	timeout    time.Duration
}

// NewGeminiProvider returns a chat.Provider backed by the Google Gemini API.
// baseURL is the API root (e.g. https://generativelanguage.googleapis.com) and
// apiKey is sent via the x-goog-api-key header. timeout bounds each
// non-streaming completion; streaming uses the shared streamTimeout budget.
func NewGeminiProvider(baseURL, apiKey string, timeout time.Duration) chat.Provider {
	return &geminiProvider{
		baseURL:    baseURL,
		apiKey:     apiKey,
		httpClient: &http.Client{},
		timeout:    timeout,
	}
}

// Name returns the provider's unique identifier.
func (p *geminiProvider) Name() string { return "gemini" }

// --- wire types ------------------------------------------------------------

type geminiRequest struct {
	Contents          []geminiContent  `json:"contents"`
	SystemInstruction *geminiContent   `json:"systemInstruction,omitempty"`
	GenerationConfig  *geminiGenConfig `json:"generationConfig,omitempty"`
	Tools             []geminiTool     `json:"tools,omitempty"`
}

type geminiTool struct {
	FunctionDeclarations []geminiFunctionDeclaration `json:"functionDeclarations"`
}

type geminiFunctionDeclaration struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

type geminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	Text             string                  `json:"text,omitempty"`
	FunctionCall     *geminiFunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *geminiFunctionResponse `json:"functionResponse,omitempty"`
}

type geminiFunctionCall struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"args"`
	// Gemini requires functionCall.id to match the answering functionResponse.id.
	ID string `json:"id,omitempty"`
}

type geminiFunctionResponse struct {
	Name     string         `json:"name"`
	Response map[string]any `json:"response"`
	ID       string         `json:"id,omitempty"`
}

type geminiGenConfig struct {
	Temperature     *float64              `json:"temperature,omitempty"`
	MaxOutputTokens *int                  `json:"maxOutputTokens,omitempty"`
	ThinkingConfig  *geminiThinkingConfig `json:"thinkingConfig,omitempty"`
}

// geminiThinkingConfig mirrors Gemini's generationConfig.thinkingConfig.
// Gemini 3 models take a thinkingLevel string enum; earlier families take a
// thinkingBudget token count. Only one field is set per model family.
type geminiThinkingConfig struct {
	ThinkingBudget int    `json:"thinkingBudget,omitempty"`
	ThinkingLevel  string `json:"thinkingLevel,omitempty"`
}

type geminiResponse struct {
	Candidates    []geminiCandidate `json:"candidates"`
	UsageMetadata geminiUsageMeta   `json:"usageMetadata"`
	ResponseID    string            `json:"responseId"`
}

type geminiCandidate struct {
	Content      geminiContent `json:"content"`
	FinishReason string        `json:"finishReason"`
}

type geminiUsageMeta struct {
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
	TotalTokenCount      int `json:"totalTokenCount"`
}

// --- request translation ---------------------------------------------------

// buildGeminiRequest translates a canonical ChatRequest into the Gemini
// generateContent wire body.
func buildGeminiRequest(req chat.ChatRequest) ([]byte, error) {
	gr := geminiRequest{}
	for _, m := range req.Messages {
		gr.Contents = append(gr.Contents, geminiContent{
			Role:  geminiRole(m.Role),
			Parts: geminiPartsFor(m, req.Messages),
		})
	}
	if req.System != "" {
		gr.SystemInstruction = &geminiContent{Parts: []geminiPart{{Text: req.System}}}
	}
	if req.Temperature != nil || req.MaxTokens != nil || req.ReasoningEffort != nil {
		gc := &geminiGenConfig{}
		if req.Temperature != nil {
			gc.Temperature = req.Temperature
		}
		if req.MaxTokens != nil {
			gc.MaxOutputTokens = req.MaxTokens
		}
		if req.ReasoningEffort != nil {
			gc.ThinkingConfig = geminiThinkingConfigFor(req.Model, *req.ReasoningEffort)
			// Gemini 3 thinking requires temperature == 1.0; override any
			// caller-supplied value so the request is not rejected.
			if isGemini3Model(req.Model) && req.Temperature != nil && *req.Temperature != 1.0 {
				one := 1.0
				gc.Temperature = &one
			}
		}
		gr.GenerationConfig = gc
	}
	for _, t := range req.Tools {
		gr.Tools = append(gr.Tools, geminiTool{
			FunctionDeclarations: []geminiFunctionDeclaration{{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.InputSchema,
			}},
		})
	}
	return json.Marshal(gr)
}

// geminiPartsFor translates one canonical Message into Gemini parts:
//   - tool-result messages (role "tool") become a single functionResponse part
//     in a "user" content; the response is the content parsed as a JSON object
//     when possible, else {"result": content}.
//   - assistant messages with ToolCalls emit one functionCall part per call
//     (plus a text part when Content is non-empty).
//   - everything else stays a plain text part.
func geminiPartsFor(m chat.Message, messages []chat.Message) []geminiPart {
	if m.Role == chat.Role("tool") {
		return []geminiPart{{FunctionResponse: &geminiFunctionResponse{
			Name:     toolNameFor(m, messages),
			Response: toolResponseFor(m.Content),
			ID:       m.ToolCallID,
		}}}
	}

	if len(m.ToolCalls) == 0 {
		return []geminiPart{{Text: m.Content}}
	}

	parts := make([]geminiPart, 0, 1+len(m.ToolCalls))
	if m.Content != "" {
		parts = append(parts, geminiPart{Text: m.Content})
	}
	for _, tc := range m.ToolCalls {
		parts = append(parts, geminiPart{FunctionCall: &geminiFunctionCall{
			Name:      tc.Name,
			Arguments: toolArguments(tc.Arguments),
			ID:        tc.ID,
		}})
	}
	return parts
}

// toolArguments unmarshals a tool call's raw JSON into a map; invalid or empty
// input falls back to an empty object so the wire never carries a bad payload.
func toolArguments(raw json.RawMessage) map[string]any {
	var args map[string]any
	if len(raw) == 0 || json.Unmarshal(raw, &args) != nil {
		return map[string]any{}
	}
	return args
}

// toolResponseFor turns a tool result's content into the functionResponse
// object: a valid JSON object is used as-is, anything else becomes
// {"result": content}.
func toolResponseFor(content string) map[string]any {
	if content == "" {
		return map[string]any{"result": content}
	}
	var obj map[string]any
	if err := json.Unmarshal([]byte(content), &obj); err == nil && obj != nil {
		return obj
	}
	return map[string]any{"result": content}
}

// toolNameFor resolves the tool name for a functionResponse. Canonical tool
// messages carry no name (only ToolCallID + Content), so the name is matched
// from the assistant ToolCall with the same ID earlier in the conversation.
// Falls back to a placeholder when no match exists.
func toolNameFor(m chat.Message, messages []chat.Message) string {
	for _, prev := range messages {
		for _, tc := range prev.ToolCalls {
			if tc.ID != "" && tc.ID == m.ToolCallID {
				return tc.Name
			}
		}
	}
	return "functionCallResponse"
}

// geminiThinkingConfigFor maps a canonical reasoning effort onto Gemini's
// thinkingConfig, model-family-aware:
//   - Gemini 3 (isGemini3Model) takes a thinkingLevel string enum.
//   - Earlier families (2.x) take a thinkingBudget token count. Budget 0 would
//     disable thinking on Flash-only models and 2.5 Pro cannot disable it at
//     all — 1024 is the safe floor, so "none"/"minimal"/"low" all map there.
func geminiThinkingConfigFor(model, effort string) *geminiThinkingConfig {
	if isGemini3Model(model) {
		level := map[string]string{
			"none": "minimal", "minimal": "minimal",
			"low": "low", "medium": "medium",
			"high": "high", "xhigh": "high", "max": "high",
		}[effort]
		if level == "" {
			level = "medium"
		}
		return &geminiThinkingConfig{ThinkingLevel: level}
	}
	budget := map[string]int{
		"none": 1024, "minimal": 1024, "low": 1024,
		"medium": 4096, "high": 16384,
		"xhigh": 32768, "max": 32768,
	}[effort]
	if budget == 0 {
		budget = 4096
	}
	return &geminiThinkingConfig{ThinkingBudget: budget}
}

// isGemini3Model reports whether the model belongs to the Gemini 3 family
// (gemini-3-pro, gemini-3.5-flash, gemini-3.1, ...), which switches thinking
// config from a token budget to a thinkingLevel enum.
func isGemini3Model(model string) bool {
	return strings.Contains(model, "3")
}

// geminiRole maps a canonical Role to the Gemini role. An empty message maps to
// an empty string part (the caller passes Content through unchanged).
func geminiRole(r chat.Role) string {
	switch r {
	case chat.RoleAssistant:
		return "model"
	default:
		return "user"
	}
}

// endpoint builds the Gemini URL for a model. A leading "models/" prefix on the
// model name is stripped so the concrete model name is forwarded.
func (p *geminiProvider) endpoint(model string, stream bool) string {
	model = strings.TrimPrefix(model, "models/")
	base := strings.TrimRight(p.baseURL, "/")
	if stream {
		return base + "/models/" + model + ":streamGenerateContent?alt=sse"
	}
	return base + "/models/" + model + ":generateContent"
}

// --- finish_reason mapping -------------------------------------------------

// geminiFinishReason maps a Gemini finishReason (UPPERCASE) to the canonical
// finish-reason vocabulary.
func geminiFinishReason(fr string) string {
	switch fr {
	case "STOP":
		return "stop"
	case "MAX_TOKENS":
		return "length"
	case "SAFETY", "RECITATION":
		return "content_filter"
	case "OTHER":
		return "other"
	default:
		return fr
	}
}

// geminiUsage maps Gemini usageMetadata into the canonical Usage.
func geminiUsage(u geminiUsageMeta) chat.Usage {
	return chat.Usage{
		PromptTokens:     u.PromptTokenCount,
		CompletionTokens: u.CandidatesTokenCount,
		TotalTokens:      u.TotalTokenCount,
	}
}

// --- Complete (non-streaming) ----------------------------------------------

// Complete performs a non-streaming completion.
func (p *geminiProvider) Complete(ctx context.Context, req chat.ChatRequest) (chat.ChatResponse, error) {
	ctx, cancel := context.WithTimeout(ctx, p.timeout)
	defer cancel()
	body, err := buildGeminiRequest(req)
	if err != nil {
		return chat.ChatResponse{}, fmt.Errorf("gemini: build request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.endpoint(req.Model, false), bytes.NewReader(body))
	if err != nil {
		return chat.ChatResponse{}, fmt.Errorf("gemini: new request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-goog-api-key", p.apiKey)

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return chat.ChatResponse{}, fmt.Errorf("gemini: request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		if resp.StatusCode == http.StatusTooManyRequests {
			return chat.ChatResponse{}, &chat.RateLimitError{Provider: p.Name(), StatusCode: resp.StatusCode, Message: strings.TrimSpace(string(b))}
		}
		return chat.ChatResponse{}, fmt.Errorf("gemini: status %d: %s", resp.StatusCode, strings.TrimSpace(string(b)))
	}

	var gr geminiResponse
	if err := json.NewDecoder(resp.Body).Decode(&gr); err != nil {
		return chat.ChatResponse{}, fmt.Errorf("gemini: decode response: %w", err)
	}

	out := chat.ChatResponse{Model: req.Model, ID: gr.ResponseID}
	if len(gr.Candidates) > 0 {
		c := gr.Candidates[0]
		var sb strings.Builder
		for _, part := range c.Content.Parts {
			sb.WriteString(part.Text)
			if part.FunctionCall != nil {
				fc := part.FunctionCall
				args, err := json.Marshal(fc.Arguments)
				if err != nil {
					return chat.ChatResponse{}, fmt.Errorf("gemini: marshal function args: %w", err)
				}
				out.ToolCalls = append(out.ToolCalls, chat.ToolCall{
					ID:        "",
					Name:      fc.Name,
					Arguments: args,
				})
			}
		}
		out.Content = sb.String()
		if len(out.ToolCalls) > 0 {
			// Gemini reports STOP even when it emits function calls; surface
			// the canonical tool_calls reason instead.
			out.FinishReason = "tool_calls"
		} else {
			out.FinishReason = geminiFinishReason(c.FinishReason)
		}
	}
	out.Usage = geminiUsage(gr.UsageMetadata)
	return out, nil
}

// --- Stream ----------------------------------------------------------------

// Stream performs a streaming completion over the :streamGenerateContent SSE
// endpoint. Each `data: {...}` frame is a partial GenerateContentResponse.
// There is no [DONE] sentinel; a final StreamDelta carrying the finish reason
// (and usage when present) is emitted at end of body.
func (p *geminiProvider) Stream(ctx context.Context, req chat.ChatRequest, emit chat.StreamFunc) error {
	ctx, cancel := context.WithTimeout(ctx, streamTimeout)
	defer cancel()
	body, err := buildGeminiRequest(req)
	if err != nil {
		return fmt.Errorf("gemini: build request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.endpoint(req.Model, true), bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("gemini: new request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("x-goog-api-key", p.apiKey)

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return fmt.Errorf("gemini: request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		if resp.StatusCode == http.StatusTooManyRequests {
			return &chat.RateLimitError{Provider: p.Name(), StatusCode: resp.StatusCode, Message: strings.TrimSpace(string(b))}
		}
		return fmt.Errorf("gemini: status %d: %s", resp.StatusCode, strings.TrimSpace(string(b)))
	}

	delivered := false
	var usage *chat.Usage
	finishReason := ""
	var toolCall *chat.ToolCall

	scanner := bufio.NewScanner(resp.Body)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")

		var chunk geminiResponse
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			if delivered {
				_ = emit(chat.StreamDelta{FinishReason: "error"})
				return nil
			}
			return fmt.Errorf("gemini: parse chunk: %w", err)
		}

		if len(chunk.Candidates) > 0 {
			c := chunk.Candidates[0]
			for _, part := range c.Content.Parts {
				if part.FunctionCall != nil {
					fc := part.FunctionCall
					args, err := json.Marshal(fc.Arguments)
					if err != nil {
						return fmt.Errorf("gemini: marshal function args: %w", err)
					}
					toolCall = &chat.ToolCall{ID: "", Name: fc.Name, Arguments: args}
					continue
				}
				if part.Text == "" {
					continue
				}
				delivered = true
				if err := emit(chat.StreamDelta{Delta: part.Text}); err != nil {
					return err
				}
			}
			if c.FinishReason != "" {
				finishReason = geminiFinishReason(c.FinishReason)
			}
		}

		if chunk.UsageMetadata.PromptTokenCount != 0 ||
			chunk.UsageMetadata.CandidatesTokenCount != 0 ||
			chunk.UsageMetadata.TotalTokenCount != 0 {
			u := geminiUsage(chunk.UsageMetadata)
			usage = &u
		}
	}

	if err := scanner.Err(); err != nil {
		if delivered {
			_ = emit(chat.StreamDelta{FinishReason: "error"})
			return nil
		}
		return fmt.Errorf("gemini: read stream: %w", err)
	}

	if toolCall != nil {
		return emit(chat.StreamDelta{FinishReason: "tool_calls", ToolCalls: []chat.ToolCall{*toolCall}})
	}
	return emit(chat.StreamDelta{FinishReason: finishReason, Usage: usage})
}
