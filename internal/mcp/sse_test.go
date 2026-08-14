package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

// sseTestServer serves an SSE endpoint event pointing at its own /post
// handler, which answers JSON-RPC POSTs.
func TestSSEConnectListCall(t *testing.T) {
	var postURL string

	mux := http.NewServeMux()

	mux.HandleFunc("/sse", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Mcp-Session-Id", "sse-sess-1")
		// Send the endpoint event then close the stream.
		fmt.Fprintf(w, "event: endpoint\ndata: %s\n\n", postURL)
	})

	mux.HandleFunc("/post", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		var req RequestMessage
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &req)

		switch req.Method {
		case "initialize":
			writeResult(w, req.ID, map[string]any{
				"protocolVersion": ProtocolVersion,
				"capabilities":    map[string]any{},
				"serverInfo":      map[string]any{"name": "sse-test", "version": "1.0.0"},
			})
		case "notifications/initialized":
			w.WriteHeader(http.StatusAccepted)
		case "tools/list":
			writeResult(w, req.ID, map[string]any{
				"tools": []map[string]any{
					{"name": "add", "description": "adds", "inputSchema": map[string]any{"type": "object"}},
				},
			})
		case "tools/call":
			writeResult(w, req.ID, map[string]any{
				"content": []map[string]any{{"type": "text", "text": "42"}},
				"isError": false,
			})
		default:
			writeError(w, req.ID, -32601, "method not found")
		}
	})

	ts := httptest.NewServer(mux)
	defer ts.Close()
	postURL = ts.URL + "/post"

	c, err := New(Config{URL: ts.URL + "/sse", Transport: "sse"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if err := c.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}

	tools, err := c.ListTools(context.Background())
	if err != nil {
		t.Fatalf("ListTools: %v", err)
	}
	if len(tools) != 1 || tools[0].Name != "add" {
		t.Fatalf("ListTools = %+v", tools)
	}

	text, isErr, err := c.CallTool(context.Background(), "add", map[string]any{"a": 1, "b": 2})
	if err != nil {
		t.Fatalf("CallTool: %v", err)
	}
	if isErr || text != "42" {
		t.Fatalf("CallTool = (%q, %v)", text, isErr)
	}
}

func TestResolveURL(t *testing.T) {
	cases := []struct{ base, ref, want string }{
		{"http://h/sse", "/post", "http://h/post"},
		{"http://h/sse", "http://other/x", "http://other/x"},
		{"http://h/sse", "post", "http://h/post"},
	}
	for _, c := range cases {
		if got := resolveURL(c.base, c.ref); got != c.want {
			t.Errorf("resolveURL(%q, %q) = %q, want %q", c.base, c.ref, got, c.want)
		}
	}
}