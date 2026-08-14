package tools

import (
	"context"
	"encoding/json"
	"time"

	"github.com/aibattery/router/internal/chat"
)

// echoText returns the "text" argument, defaulting to "abc" when absent.
func echoText(_ context.Context, args json.RawMessage) (string, error) {
	var in struct {
		Text string `json:"text"`
	}
	if len(args) > 0 {
		_ = json.Unmarshal(args, &in)
	}
	if in.Text == "" {
		in.Text = "abc"
	}
	return in.Text, nil
}

// getUTCTime returns the current UTC time as ISO 8601.
func getUTCTime(_ context.Context, _ json.RawMessage) (string, error) {
	return time.Now().UTC().Format(time.RFC3339), nil
}

// RegisterBuiltin registers the two built-in local tools (echo_text and
// get_utc_time) into reg. It returns ErrDuplicate if either name is already
// taken.
func RegisterBuiltin(reg *Registry) error {
	if err := reg.RegisterLocal(chat.Tool{
		Name:        "echo_text",
		Description: "Echoes back the provided text.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"text": map[string]any{"type": "string"},
			},
			"required": []any{"text"},
		},
	}, echoText); err != nil {
		return err
	}
	return reg.RegisterLocal(chat.Tool{
		Name:        "get_utc_time",
		Description: "Returns the current UTC time as ISO 8601.",
		InputSchema: map[string]any{"type": "object"},
	}, getUTCTime)
}