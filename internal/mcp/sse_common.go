package mcp

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// finishDoResponse is the shared tail of a JSON-RPC POST for both the
// streamable and SSE transports: it captures the session id advertised by
// the server (via rememberSession, which locks the caller's transport),
// then drains notifications / 202 Accepted bodies and decodes the response
// body as SSE or plain JSON. Locking stays with the caller so the
// lock/unlock order of each transport is unchanged.
func finishDoResponse(resp *http.Response, reqID json.RawMessage, expectsResponse bool, rememberSession func(string)) (*ResponseMessage, error) {
	if sid := resp.Header.Get(headerMcpSessionID); sid != "" {
		rememberSession(sid)
	}

	// Notifications (and 202 Accepted) carry no JSON-RPC response body.
	if !expectsResponse || resp.StatusCode == http.StatusAccepted {
		_, _ = io.Copy(io.Discard, resp.Body)
		return nil, nil
	}

	ct := resp.Header.Get("Content-Type")
	if strings.Contains(ct, "text/event-stream") {
		return parseSSEResponse(resp.Body, reqID)
	}

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("mcp: read response: %w", err)
	}
	return parseResponse(data)
}
