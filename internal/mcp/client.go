package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/aibattery/router/internal/chat"
)

// Config configures a Client for a single MCP server.
type Config struct {
	Name         string       // logical name (config key); unused for wire but stored
	URL          string       // boot URL. streamable: POST here. sse: GET here for discovery.
	Transport    string       // "streamable" | "sse" (default "streamable")
	BearerToken  string       // optional; sent as Authorization: Bearer
	HTTPClient   *http.Client // optional; default http.DefaultClient w/ timeout 60s
}

// Client is a connected MCP client for ONE server.
type Client struct {
	cfg       Config
	transport Transport
	nextID    atomic.Int64

	mu              sync.Mutex
	protocolVersion string
}

// New validates cfg and builds a Client. It does not connect.
func New(cfg Config) (*Client, error) {
	if cfg.URL == "" {
		return nil, errors.New("mcp: url is required")
	}
	if cfg.Transport == "" {
		cfg.Transport = "streamable"
	}
	if cfg.Transport != "streamable" && cfg.Transport != "sse" {
		return nil, fmt.Errorf("mcp: unknown transport %q", cfg.Transport)
	}

	hc := cfg.HTTPClient
	if hc == nil {
		hc = &http.Client{Timeout: 60 * time.Second}
	}

	c := &Client{cfg: cfg}
	switch cfg.Transport {
	case "sse":
		c.transport = &SSETransport{bootURL: cfg.URL, bearerToken: cfg.BearerToken, client: hc}
	default:
		c.transport = &streamableTransport{url: cfg.URL, bearerToken: cfg.BearerToken, client: hc}
	}
	return c, nil
}

// Connect performs the initialize handshake. For the sse transport it first
// resolves the endpoint, then sends initialize and notifications/initialized.
func (c *Client) Connect(ctx context.Context) error {
	if sse, ok := c.transport.(*SSETransport); ok {
		if err := sse.Connect(ctx); err != nil {
			return err
		}
	}

	// initialize
	initParams := map[string]any{
		"protocolVersion": ProtocolVersion,
		"capabilities":    map[string]any{},
		"clientInfo": map[string]any{
			"name":    "aibattery-router",
			"version": "1.0.0",
		},
	}
	req, err := newRequest(c.nextID.Add(1), "initialize", initParams)
	if err != nil {
		return err
	}
	resp, err := c.transport.Do(ctx, req, true)
	if err != nil {
		return err
	}
	if resp.Error != nil {
		return resp.Error
	}

	var initResult struct {
		ProtocolVersion string `json:"protocolVersion"`
	}
	if err := json.Unmarshal(resp.Result, &initResult); err != nil {
		return fmt.Errorf("mcp: decode initialize result: %w", err)
	}
	c.mu.Lock()
	c.protocolVersion = initResult.ProtocolVersion
	c.mu.Unlock()

	// notifications/initialized — fire-and-forget, no JSON-RPC response.
	notif, err := newRequest(0, "notifications/initialized", nil)
	if err != nil {
		return err
	}
	_, err = c.transport.Do(ctx, notif, false)
	return err
}

// Close releases transport resources.
func (c *Client) Close(ctx context.Context) error {
	return c.transport.Close(ctx)
}

// ListTools returns the tools advertised by the server.
func (c *Client) ListTools(ctx context.Context) ([]chat.Tool, error) {
	req, err := newRequest(c.nextID.Add(1), "tools/list", nil)
	if err != nil {
		return nil, err
	}
	resp, err := c.transport.Do(ctx, req, true)
	if err != nil {
		return nil, err
	}
	if resp.Error != nil {
		return nil, resp.Error
	}

	var result struct {
		Tools []struct {
			Name        string         `json:"name"`
			Description string         `json:"description"`
			InputSchema map[string]any `json:"inputSchema"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(resp.Result, &result); err != nil {
		return nil, fmt.Errorf("mcp: decode tools/list result: %w", err)
	}

	tools := make([]chat.Tool, 0, len(result.Tools))
	for _, t := range result.Tools {
		tools = append(tools, chat.Tool{
			Name:        t.Name,
			Description: t.Description,
			InputSchema: t.InputSchema,
		})
	}
	return tools, nil
}

// CallTool invokes a server tool, returning the concatenated text content,
// whether the server marked it an error, and any transport/jsonrpc error.
func (c *Client) CallTool(ctx context.Context, name string, args map[string]any) (string, bool, error) {
	params := map[string]any{"name": name, "arguments": args}
	req, err := newRequest(c.nextID.Add(1), "tools/call", params)
	if err != nil {
		return "", false, err
	}
	resp, err := c.transport.Do(ctx, req, true)
	if err != nil {
		return "", false, err
	}
	if resp.Error != nil {
		return "", false, resp.Error
	}

	var result struct {
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
		IsError bool `json:"isError"`
	}
	if err := json.Unmarshal(resp.Result, &result); err != nil {
		return "", false, fmt.Errorf("mcp: decode tools/call result: %w", err)
	}

	var sb strings.Builder
	for _, part := range result.Content {
		if part.Type == "text" {
			sb.WriteString(part.Text)
		}
	}
	return sb.String(), result.IsError, nil
}

// ProtocolVersion returns the negotiated protocolVersion (post-Connect).
func (c *Client) ProtocolVersion() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.protocolVersion
}