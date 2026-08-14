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

// ProviderError marks any upstream HTTP error with its status code so the API
// layer can decide whether a retry is worthwhile. 5xx and network errors are
// transient (retry); 4xx are permanent (fail over immediately, no retry).
type ProviderError struct {
	Provider   string // provider name, e.g. "openai"
	StatusCode int    // upstream HTTP status code
	Message    string // trimmed upstream body
}

func (e *ProviderError) Error() string {
	return fmt.Sprintf("provider %s: status %d: %s", e.Provider, e.StatusCode, e.Message)
}

// IsRetryable reports whether err is a transient failure worth retrying:
// 5xx statuses and transport-level errors (timeout, connection reset) qualify;
// 4xx statuses and 429 (handled by cooldown) do not.
func IsRetryable(err error) bool {
	if IsRateLimit(err) {
		return false
	}
	var pe *ProviderError
	if errors.As(err, &pe) {
		return pe.StatusCode >= 500
	}
	// No HTTP status: network/timeout/parse error — transient, retry.
	return true
}