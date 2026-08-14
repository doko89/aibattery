package chat

import "encoding/json"

// Tool is a provider-agnostic function definition usable by a model for
// function calling. InputSchema is a JSON Schema object.
type Tool struct {
	Name        string
	Description string
	InputSchema map[string]any
}

// ToolCall is a model's request to invoke a tool.
type ToolCall struct {
	ID        string          // provider-specific call id, when available
	Name      string
	Arguments json.RawMessage // JSON object of arguments; may be empty {}
}