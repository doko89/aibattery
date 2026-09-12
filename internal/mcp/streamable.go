package mcp

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
)

// Transport abstracts the wire transport used to exchange JSON-RPC messages
// with a single MCP server.
type Transport interface {
	// Do sends a request (or notification when expectsResponse is false) and
	// returns the correlated response. For notifications it returns nil.
	Do(ctx context.Context, req RequestMessage, expectsResponse bool) (*ResponseMessage, error)
	Close(ctx context.Context) error
}

// streamableTransport implements the HTTP Streamable transport: every message
// is POSTed as JSON to a fixed URL.
type streamableTransport struct {
	url         string
	bearerToken string
	client      *http.Client

	mu        sync.Mutex
	sessionID string
}

func (t *streamableTransport) Do(ctx context.Context, req RequestMessage, expectsResponse bool) (*ResponseMessage, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("mcp: encode request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, t.url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json, text/event-stream")
	httpReq.Header.Set("MCP-Protocol-Version", ProtocolVersion)
	if t.bearerToken != "" {
		httpReq.Header.Set("Authorization", "Bearer "+t.bearerToken)
	}
	t.mu.Lock()
	if t.sessionID != "" {
		httpReq.Header.Set(headerMcpSessionID, t.sessionID)
	}
	t.mu.Unlock()

	resp, err := t.client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	// Capture and echo the session id on subsequent requests.
	return finishDoResponse(resp, req.ID, expectsResponse, func(sid string) {
		t.mu.Lock()
		t.sessionID = sid
		t.mu.Unlock()
	})
}

func (t *streamableTransport) Close(context.Context) error { return nil }
