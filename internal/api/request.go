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
}

// chatMessage is a single message in the OpenAI wire format. Content is either
// plain text or an array of content parts; buildChatRequest normalizes it.
type chatMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}
