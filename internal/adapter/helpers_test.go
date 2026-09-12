package adapter

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/aibattery/router/internal/chat"
)

// Shared mock-upstream helpers for the adapter tests.
//
// Every fake upstream used to be spelled out as an inline
// httptest.NewServer + defer srv.Close() block (dozens of copies across
// openai_test.go, anthropic_test.go and gemini_test.go). The helpers below
// centralize only the server/request/stream *setup*; all behavioral
// assertions stay inline in the individual tests.

// newMockServer starts a test HTTP server with the given handler and
// registers its cleanup. It replaces the repeated
// httptest.NewServer + defer srv.Close() boilerplate.
func newMockServer(t *testing.T, handler http.HandlerFunc) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	return srv
}

// newStaticJSONServer serves body with 200 + application/json for every request.
func newStaticJSONServer(t *testing.T, body string) *httptest.Server {
	t.Helper()
	return newMockServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, body)
	})
}

// newStaticStatusServer serves body with the given HTTP status code.
func newStaticStatusServer(t *testing.T, status int, body string) *httptest.Server {
	t.Helper()
	return newMockServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(status)
		fmt.Fprint(w, body)
	})
}

// newCaptureJSONServer runs capture on each request, then serves respBody
// with 200 + application/json.
func newCaptureJSONServer(t *testing.T, respBody string, capture func(r *http.Request)) *httptest.Server {
	t.Helper()
	return newMockServer(t, func(w http.ResponseWriter, r *http.Request) {
		capture(r)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, respBody)
	})
}

// newSSEServer serves frames as a text/event-stream response.
func newSSEServer(t *testing.T, frames string) *httptest.Server {
	t.Helper()
	return newMockServer(t, func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, frames)
	})
}

// newCaptureSSEServer runs capture on each request, then serves SSE frames.
func newCaptureSSEServer(t *testing.T, frames string, capture func(r *http.Request)) *httptest.Server {
	t.Helper()
	return newMockServer(t, func(w http.ResponseWriter, r *http.Request) {
		capture(r)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, frames)
	})
}

// decodeTestBody decodes a JSON request body into dst and returns any error
// so each provider-specific capture helper can preserve its reporting policy.
func decodeTestBody(r *http.Request, dst any) error {
	return json.NewDecoder(r.Body).Decode(dst)
}

// simpleUserRequest builds the minimal one-turn user request reused by the
// tests that only need a well-formed request body.
func simpleUserRequest(model string) chat.ChatRequest {
	return chat.ChatRequest{
		Model:    model,
		Messages: []chat.Message{{Role: chat.RoleUser, Content: "hi"}},
	}
}

// collectStream runs Stream to completion, gathering every emitted delta.
// The returned error is left for the caller to assert on.
func collectStream(t *testing.T, p chat.Provider, req chat.ChatRequest) ([]chat.StreamDelta, error) {
	t.Helper()
	var deltas []chat.StreamDelta
	err := p.Stream(context.Background(), req, func(d chat.StreamDelta) error {
		deltas = append(deltas, d)
		return nil
	})
	return deltas, err
}
