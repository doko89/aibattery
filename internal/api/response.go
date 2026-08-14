package api

// chatCompletionResponse is the OpenAI-compatible non-streaming completion
// payload returned by POST /v1/chat/completions.
type chatCompletionResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []choice `json:"choices"`
	Usage   *usage   `json:"usage,omitempty"`
}

// choice is a single completion choice.
type choice struct {
	Index        int         `json:"index"`
	Message      respMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

// respMessage is the assistant message inside a completion choice.
type respMessage struct {
	Role      string         `json:"role"`
	Content   string         `json:"content"`
	ToolCalls []respToolCall `json:"tool_calls,omitempty"`
}

// respToolCall is a model-requested tool invocation in the OpenAI wire shape.
type respToolCall struct {
	ID       string               `json:"id"`
	Type     string               `json:"type"`
	Function respToolCallFunction `json:"function"`
}

// respToolCallFunction carries the function name and JSON-encoded arguments.
type respToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// usage reports token consumption in the OpenAI shape.
type usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// modelListResponse is the payload for GET /v1/models.
type modelListResponse struct {
	Object string       `json:"object"`
	Data   []modelEntry `json:"data"`
}

// modelEntry is a single entry in the model list.
type modelEntry struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// errorResponse is the OpenAI-compatible error envelope.
type errorResponse struct {
	Error apiError `json:"error"`
}

// apiError carries the error message, type and optional code.
type apiError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code,omitempty"`
}