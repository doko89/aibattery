package chat

import (
	"errors"
	"fmt"
)

// RateLimitError marks an upstream HTTP 429 (rate limit) response so the
// API layer can apply per-model cooldown only on rate limits.
type RateLimitError struct {
	Provider   string // provider name, e.g. "openai"
	StatusCode int    // always 429 in practice
	Message    string // trimmed upstream body
}

func (e *RateLimitError) Error() string {
	return fmt.Sprintf("provider %s: status %d: %s", e.Provider, e.StatusCode, e.Message)
}

// IsRateLimit reports whether err is (or wraps) a RateLimitError.
func IsRateLimit(err error) bool {
	var rl *RateLimitError
	return errors.As(err, &rl)
}