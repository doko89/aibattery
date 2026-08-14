package api

import (
	"encoding/json"
	"errors"
	"net/http"

	"github.com/aibattery/router/internal/tools"
)

// toolListResponse is the OpenAI-style tool list payload for GET /v1/tools.
type toolListResponse struct {
	Object string       `json:"object"`
	Data   []toolObject `json:"data"`
}

// toolObject is a single OpenAI-style function tool entry.
type toolObject struct {
	Type     string         `json:"type"`
	Function toolFunction   `json:"function"`
}

// toolFunction carries the tool's name, description and JSON-Schema input.
type toolFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
}

// toolCallRequest is the wire body for POST /v1/tools/call.
type toolCallRequest struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

// toolCallResult is the MCP-style result payload returned on success.
type toolCallResult struct {
	Content []toolContent `json:"content"`
	IsError bool          `json:"isError"`
}

// toolContent is a single text content part of a tool call result.
type toolContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// handleTools serves GET /v1/tools, returning the aggregated tool set in the
// OpenAI function-calling shape.
func (s *Server) handleTools(w http.ResponseWriter, r *http.Request) {
	list := s.deps.Tools.List()
	data := make([]toolObject, 0, len(list))
	for _, t := range list {
		data = append(data, toolObject{
			Type: "function",
			Function: toolFunction{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.InputSchema,
			},
		})
	}
	writeJSON(w, http.StatusOK, toolListResponse{Object: "list", Data: data})
}

// handleToolCall serves POST /v1/tools/call, dispatching a named tool with its
// JSON arguments and returning an MCP-style CallResult.
func (s *Server) handleToolCall(w http.ResponseWriter, r *http.Request) {
	var req toolCallRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body: "+err.Error(), "invalid_request_error", "")
		return
	}
	if req.Name == "" {
		writeError(w, http.StatusBadRequest, "name is required", "invalid_request_error", "")
		return
	}

	text, err := s.deps.Tools.Call(r.Context(), req.Name, req.Arguments)
	if err != nil {
		if errors.Is(err, tools.ErrToolNotFound) {
			writeError(w, http.StatusNotFound, "tool not found: "+req.Name, "invalid_request_error", "tool_not_found")
			return
		}
		writeError(w, http.StatusBadGateway, err.Error(), "upstream_error", "")
		return
	}

	writeJSON(w, http.StatusOK, toolCallResult{
		Content: []toolContent{{Type: "text", Text: text}},
		IsError: false,
	})
}