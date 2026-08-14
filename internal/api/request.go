package api

import "encoding/json"

// chatCompletionRequest is the OpenAI-compatible wire format accepted by
// POST /v1/chat/completions. The `model` field carries the VIRTUAL model name
// that the routing layer resolves to a concrete provider+model pair.
type chatCompletionRequest struct {
	Model           string        `json:"model"`
	Messages        []chatMessage `json:"messages"`
	Stream          bool          `json:"stream"`
	Temperature     *float64      `json:"temperature"`
	MaxTokens       *int          `json:"max_tokens"`
	ReasoningEffort *string       `json:"reasoning_effort"`
	Tools           []wireTool    `json:"tools,omitempty"`
}

// wireTool is the OpenAI wire shape of a function definition offered to the
// model. When a client supplies tools, they are passed through instead of the
// router's aggregated MCP tool set.
type wireTool struct {
	Type     string           `json:"type"` // "function"
	Function wireToolFunction `json:"function"`
}

type wireToolFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
}

// chatMessage is a single message in the OpenAI wire format. Content is either
// plain text or an array of content parts; buildChatRequest normalizes it.
// ToolCallID and ToolCalls carry multi-turn function-calling state.
type chatMessage struct {
	Role             string                `json:"role"`
	Content          json.RawMessage       `json:"content"`
	ToolCallID       string                `json:"tool_call_id,omitempty"`
	ToolCalls        []wireMessageToolCall `json:"tool_calls,omitempty"`
	ReasoningContent string                `json:"reasoning_content,omitempty"`
}

type wireMessageToolCall struct {
	ID       string                      `json:"id"`
	Type     string                      `json:"type"` // "function"
	Function wireMessageToolCallFunction `json:"function"`
}

type wireMessageToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // JSON string per OpenAI wire
}
