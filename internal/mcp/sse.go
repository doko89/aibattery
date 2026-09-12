package mcp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
)

// headerMcpSessionID is the session header advertised by MCP SSE servers.
const headerMcpSessionID = "Mcp-Session-Id"

// SSETransport implements the legacy SSE transport: it opens a GET stream to
// discover the POST endpoint (via an `event: endpoint` frame), then POSTs
// JSON-RPC messages to that endpoint.
type SSETransport struct {
	bootURL     string
	bearerToken string
	client      *http.Client

	mu        sync.Mutex
	postURL   string
	sessionID string
}

// Connect opens the SSE stream and resolves the POST endpoint URL plus any
// session id advertised by the server.
func (t *SSETransport) Connect(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, t.bootURL, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Accept", "text/event-stream")
	if t.bearerToken != "" {
		req.Header.Set("Authorization", "Bearer "+t.bearerToken)
	}

	resp, err := t.client.Do(req)
	if err != nil {
		return fmt.Errorf("mcp: open sse stream: %w", err)
	}
	defer resp.Body.Close()

	if sid := resp.Header.Get(headerMcpSessionID); sid != "" {
		t.mu.Lock()
		t.sessionID = sid
		t.mu.Unlock()
	}

	endpoint, err := readEndpointEvent(resp.Body, t.bootURL)
	if err != nil {
		return err
	}
	t.mu.Lock()
	t.postURL = endpoint
	t.mu.Unlock()
	return nil
}

// readEndpointEvent scans an SSE stream until an `event: endpoint` frame and
// resolves its data to an absolute URL.
func readEndpointEvent(r io.Reader, base string) (string, error) {
	scanner := bufio.NewScanner(r)
	var event, data string
	for scanner.Scan() {
		line := scanner.Text()
		switch {
		case strings.HasPrefix(line, "event:"):
			event = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
		case strings.HasPrefix(line, "data:"):
			data = strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		case line == "":
			if event == "endpoint" && data != "" {
				return resolveURL(base, data), nil
			}
			event, data = "", ""
		}
	}
	if err := scanner.Err(); err != nil {
		return "", fmt.Errorf("mcp: read sse stream: %w", err)
	}
	return "", fmt.Errorf("mcp: no endpoint event received")
}

// resolveURL resolves ref against base, returning ref unchanged if absolute.
func resolveURL(base, ref string) string {
	if strings.HasPrefix(ref, "http://") || strings.HasPrefix(ref, "https://") {
		return ref
	}
	bu, err := url.Parse(base)
	if err != nil {
		return ref
	}
	ru, err := url.Parse(ref)
	if err != nil {
		return ref
	}
	return bu.ResolveReference(ru).String()
}

func (t *SSETransport) Do(ctx context.Context, req RequestMessage, expectsResponse bool) (*ResponseMessage, error) {
	t.mu.Lock()
	postURL := t.postURL
	sessionID := t.sessionID
	t.mu.Unlock()
	if postURL == "" {
		return nil, fmt.Errorf("mcp: sse endpoint not resolved; call Connect first")
	}

	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("mcp: encode request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, postURL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json, text/event-stream")
	httpReq.Header.Set("MCP-Protocol-Version", ProtocolVersion)
	if t.bearerToken != "" {
		httpReq.Header.Set("Authorization", "Bearer "+t.bearerToken)
	}
	if sessionID != "" {
		httpReq.Header.Set(headerMcpSessionID, sessionID)
	}

	resp, err := t.client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	return finishDoResponse(resp, req.ID, expectsResponse, func(sid string) {
		t.mu.Lock()
		t.sessionID = sid
		t.mu.Unlock()
	})
}

func (t *SSETransport) Close(context.Context) error { return nil }
