package mcp

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
)

// streamableTestServer is a minimal Streamable HTTP MCP server.
type streamableTestServer struct {
	requireBearer string
	sessionID     string
	nextID        atomic.Int64
}

func (s *streamableTestServer) handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if s.requireBearer != "" && r.Header.Get("Authorization") != "Bearer "+s.requireBearer {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		if s.sessionID != "" {
			w.Header().Set("Mcp-Session-Id", s.sessionID)
		}
		w.Header().Set("Content-Type", "application/json")

		var req RequestMessage
		body, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(body, &req); err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}

		switch req.Method {
		case "initialize":
			writeResult(w, req.ID, map[string]any{
				"protocolVersion": ProtocolVersion,
				"capabilities":    map[string]any{"tools": map[string]any{"listChanged": false}},
				"serverInfo":      map[string]any{"name": "test", "version": "1.0.0"},
			})
		case "notifications/initialized":
			w.WriteHeader(http.StatusAccepted)
		case "tools/list":
			writeResult(w, req.ID, map[string]any{
				"tools": []map[string]any{
					{
						"name":        "echo",
						"description": "echoes input",
						"inputSchema": map[string]any{"type": "object", "properties": map[string]any{"text": map[string]any{"type": "string"}}},
					},
				},
			})
		case "tools/call":
			var params struct {
				Name      string         `json:"name"`
				Arguments map[string]any `json:"arguments"`
			}
			_ = json.Unmarshal(req.Params, &params)
			if params.Name == "boom" {
				writeResult(w, req.ID, map[string]any{
					"content": []map[string]any{{"type": "text", "text": "failed"}},
					"isError": true,
				})
				return
			}
			if params.Name == "nope" {
				writeError(w, req.ID, -32602, "unknown tool")
				return
			}
			text, _ := params.Arguments["text"].(string)
			writeResult(w, req.ID, map[string]any{
				"content": []map[string]any{{"type": "text", "text": "hello " + text}},
				"isError": false,
			})
		default:
			writeError(w, req.ID, -32601, "method not found")
		}
	})
	return mux
}

func writeResult(w http.ResponseWriter, id json.RawMessage, result any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"jsonrpc": "2.0",
		"id":      json.RawMessage(id),
		"result":  result,
	})
}

func writeError(w http.ResponseWriter, id json.RawMessage, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{
		"jsonrpc": "2.0",
		"id":      json.RawMessage(id),
		"error":   map[string]any{"code": code, "message": msg},
	})
}

func TestStreamableConnectListCall(t *testing.T) {
	srv := &streamableTestServer{requireBearer: "sekret", sessionID: "sess-123"}
	ts := httptest.NewServer(srv.handler())
	defer ts.Close()

	c, err := New(Config{URL: ts.URL, Transport: "streamable", BearerToken: "sekret"})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if err := c.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}
	if got := c.ProtocolVersion(); got != ProtocolVersion {
		t.Fatalf("ProtocolVersion = %q, want %q", got, ProtocolVersion)
	}

	tools, err := c.ListTools(context.Background())
	if err != nil {
		t.Fatalf("ListTools: %v", err)
	}
	if len(tools) != 1 || tools[0].Name != "echo" {
		t.Fatalf("ListTools = %+v, want one 'echo' tool", tools)
	}
	if tools[0].InputSchema["type"] != "object" {
		t.Fatalf("InputSchema = %v", tools[0].InputSchema)
	}

	text, isErr, err := c.CallTool(context.Background(), "echo", map[string]any{"text": "world"})
	if err != nil {
		t.Fatalf("CallTool: %v", err)
	}
	if isErr || text != "hello world" {
		t.Fatalf("CallTool = (%q, %v), want (\"hello world\", false)", text, isErr)
	}

	// isError=true path
	_, isErr, err = c.CallTool(context.Background(), "boom", nil)
	if err != nil {
		t.Fatalf("CallTool boom: %v", err)
	}
	if !isErr {
		t.Fatal("expected isError=true for boom tool")
	}
}

func TestStreamableRejectsBadBearer(t *testing.T) {
	srv := &streamableTestServer{requireBearer: "sekret"}
	ts := httptest.NewServer(srv.handler())
	defer ts.Close()

	c, err := New(Config{URL: ts.URL})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if err := c.Connect(context.Background()); err == nil {
		t.Fatal("expected Connect to fail without bearer token")
	}
}

func TestJSONRPCError(t *testing.T) {
	srv := &streamableTestServer{}
	ts := httptest.NewServer(srv.handler())
	defer ts.Close()

	c, err := New(Config{URL: ts.URL})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if err := c.Connect(context.Background()); err != nil {
		t.Fatalf("Connect: %v", err)
	}

	// Force a method-not-found by calling a tool the server rejects at the
	// JSON-RPC layer via a raw transport request.
	req, _ := newRequest(99, "tools/call", map[string]any{"name": "nope", "arguments": map[string]any{}})
	resp, err := c.transport.Do(context.Background(), req, true)
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	if resp.Error == nil {
		t.Fatal("expected an RPCError")
	}
	if resp.Error.Code != -32602 {
		t.Fatalf("RPCError code = %d, want -32602", resp.Error.Code)
	}
	if !strings.Contains(resp.Error.Error(), "unknown tool") {
		t.Fatalf("RPCError message = %q", resp.Error.Message)
	}
}