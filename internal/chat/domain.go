// Package chat defines the canonical, provider-agnostic domain model for the
// AI router. It is the "heart" of the system: every adapter, routing strategy
// and API handler depends on these exact types. The package is deliberately
// framework-free — pure Go types and the standard library only.
package chat

// Role identifies the speaker of a chat message.
type Role string

const (
	// RoleSystem marks a system-level instruction message.
	RoleSystem Role = "system"
	// RoleUser marks a message from the end user.
	RoleUser Role = "user"
	// RoleAssistant marks a message produced by the model.
	RoleAssistant Role = "assistant"
	// RoleTool marks a message carrying the result of a tool call, linked to
	// the assistant tool call it answers via ToolCallID.
	RoleTool Role = "tool"
)

// Message is a single chat message. For v1 the content is plain text;
// multimodal content is deferred to a later iteration.
type Message struct {
	Role    Role
	Content string
	// ToolCallID links a role=="tool" message to the assistant tool call it
	// answers (OpenAI wire tool_call_id / Anthropic tool_result tool_use_id).
	ToolCallID string
	// ToolCalls carries assistant-role tool invocations back to the model
	// in multi-turn conversations. Arguments is the raw JSON object.
	ToolCalls []ToolCall
	// ReasoningContent is the model's chain-of-thought text (DeepSeek
	// `reasoning_content`), which thinking-mode APIs require to be echoed
	// back on assistant messages in multi-turn conversations.
	ReasoningContent string
}

// ChatRequest is the canonical request accepted by every provider. It is the
// single cross-provider contract that adapters translate into their native
// wire format. It carries typed core fields plus an Extra map for untyped
// pass-through of provider-specific parameters (tools, response_format, seed,
// etc.); adapters may read the keys they understand and ignore the rest.
type ChatRequest struct {
	Model       string
	Messages    []Message
	System      string
	Temperature *float64
	MaxTokens   *int
	// ReasoningEffort maps to OpenAI's reasoning_effort semantics: one of
	// "none", "minimal", "low", "medium", "high", "xhigh", "max". Nil means
	// not specified; adapters translate it to their native equivalent.
	ReasoningEffort *string
	Stream          bool
	// Extra is an untyped pass-through for provider-specific parameters.
	Extra map[string]any
	// Tools is the set of function definitions made available to the model
	// for function calling. It is provider-agnostic; adapters translate it
	// into their native tool schema. Nil means no tools are offered.
	Tools []Tool
}

// Usage reports token consumption for a completion.
type Usage struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
}

// ChatResponse is the canonical result of a non-streaming completion.
type ChatResponse struct {
	ID           string
	Model        string
	Content      string
	FinishReason string
	Usage        Usage
	// ToolCalls holds the model's requested tool invocations, if any.
	ToolCalls []ToolCall
}

// StreamDelta is a single chunk handed to a StreamFunc during streaming.
type StreamDelta struct {
	// Delta is the text produced in this chunk.
	Delta string
	// FinishReason is empty until the final chunk, where it carries the
	// provider's stop reason (e.g. "stop", "length", or "error").
	FinishReason string
	// Usage is cumulative token usage if the provider supplies it at stream
	// end; nil otherwise.
	Usage *Usage
	// ToolCalls is populated only on the FINAL chunk (when FinishReason ==
	// "tool_calls") if the streaming model produced tool calls; nil otherwise.
	ToolCalls []ToolCall
}

// StreamFunc receives each StreamDelta produced by a streaming completion.
type StreamFunc func(StreamDelta) error
