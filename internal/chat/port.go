package chat

import "context"

// Provider is the DOMAIN PORT — the only contract that adapter providers
// implement. Every concrete provider (OpenAI-compatible, Anthropic, Gemini,
// ...) must satisfy this interface so the routing layer can treat them
// uniformly.
type Provider interface {
	// Name returns the provider's unique identifier, matching the `name`
	// field of its ProviderConfig entry.
	Name() string

	// Complete returns a full non-streaming completion for req.
	Complete(ctx context.Context, req ChatRequest) (ChatResponse, error)

	// Stream calls emit for each delta produced while streaming, then emits a
	// final chunk with FinishReason set. It returns an error only if the whole
	// stream failed BEFORE any content was delivered. Transport errors that
	// occur after the first delta must be surfaced by emitting a StreamDelta
	// with FinishReason == "error"; in that case Stream returns nil.
	Stream(ctx context.Context, req ChatRequest, emit StreamFunc) error
}
