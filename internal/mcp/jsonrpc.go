package mcp

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

// ProtocolVersion is the MCP protocol version this client speaks.
const ProtocolVersion = "2026-07-28"

// RequestMessage is a JSON-RPC 2.0 request or notification sent to the server.
// ID is omitted for notifications.
type RequestMessage struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

// ResponseMessage is a JSON-RPC 2.0 response received from the server.
type ResponseMessage struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *RPCError       `json:"error,omitempty"`
}

// RPCError is a JSON-RPC error object.
type RPCError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

func (e *RPCError) Error() string {
	return fmt.Sprintf("jsonrpc error %d: %s", e.Code, e.Message)
}

// newRequest builds a RequestMessage. A positive id assigns a numeric id; a
// zero id produces a notification (no id field). A nil params is omitted.
func newRequest(id int64, method string, params any) (RequestMessage, error) {
	req := RequestMessage{JSONRPC: "2.0", Method: method}
	if id > 0 {
		raw, err := json.Marshal(id)
		if err != nil {
			return req, err
		}
		req.ID = raw
	}
	if params != nil {
		raw, err := json.Marshal(params)
		if err != nil {
			return req, err
		}
		req.Params = raw
	}
	return req, nil
}

// matchID reports whether a response id corresponds to the sent request id.
// Both are compared as raw JSON so numeric ids match regardless of encoding.
func matchID(respID, sentID json.RawMessage) bool {
	return bytes.Equal(bytes.TrimSpace(respID), bytes.TrimSpace(sentID))
}

// parseResponse decodes a single JSON-RPC response body.
func parseResponse(data []byte) (*ResponseMessage, error) {
	var msg ResponseMessage
	if err := json.Unmarshal(data, &msg); err != nil {
		return nil, fmt.Errorf("mcp: decode response: %w", err)
	}
	return &msg, nil
}

// parseSSEResponse reads an SSE stream from r, parses each data frame as a
// ResponseMessage, and returns the first one whose id matches sentID.
func parseSSEResponse(r io.Reader, sentID json.RawMessage) (*ResponseMessage, error) {
	scanner := bufio.NewScanner(r)
	var data strings.Builder
	for scanner.Scan() {
		line := scanner.Text()
		switch {
		case strings.HasPrefix(line, "data:"):
			data.WriteString(strings.TrimPrefix(line, "data:"))
			data.WriteString("\n")
		case line == "":
			if data.Len() > 0 {
				msg, err := parseResponse([]byte(data.String()))
				if err == nil && matchID(msg.ID, sentID) {
					return msg, nil
				}
			}
			data.Reset()
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("mcp: read sse: %w", err)
	}
	return nil, fmt.Errorf("mcp: no sse response matching id %s", sentID)
}