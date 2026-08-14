// Package tools implements the TOOL GATEWAY: a concurrency-safe registry that
// aggregates tool definitions from remote MCP servers (proxied) and local Go
// executors, exposing a uniform JSON-Schema definition surface and a single
// call dispatcher.
package tools

import (
	"context"
	"encoding/json"
	"errors"
	"sort"
	"sync"

	"github.com/aibattery/router/internal/chat"
)

// LocalFunc implements a tool backed by a local Go function. It receives the
// raw JSON argument object and returns a text payload.
type LocalFunc func(ctx context.Context, args json.RawMessage) (string, error)

// RemoteFunc proxies a tool call to an upstream MCP server. name is the tool
// name as registered; args is the raw JSON argument object.
type RemoteFunc func(ctx context.Context, name string, args json.RawMessage) (string, error)

// ErrToolNotFound is returned by Call when no tool with the given name is
// registered.
var ErrToolNotFound = errors.New("tools: tool not found")

// ErrDuplicate is returned by RegisterLocal / RegisterRemote when a tool with
// the same name is already registered.
var ErrDuplicate = errors.New("tools: duplicate tool name")

// toolEntry binds a tool definition to its callable implementation.
type toolEntry struct {
	def  chat.Tool
	call func(ctx context.Context, args json.RawMessage) (string, error)
}

// Registry is a concurrency-safe collection of tool definitions and their
// callable implementations.
type Registry struct {
	mu    sync.RWMutex
	entry map[string]toolEntry
}

// New returns an empty Registry.
func New() *Registry {
	return &Registry{entry: make(map[string]toolEntry)}
}

// RegisterLocal adds a tool implemented by a local Go func. It returns
// ErrDuplicate on a name collision.
func (r *Registry) RegisterLocal(def chat.Tool, fn LocalFunc) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.entry[def.Name]; ok {
		return ErrDuplicate
	}
	r.entry[def.Name] = toolEntry{def: def, call: fn}
	return nil
}

// RegisterRemote adds a tool that is proxied to an upstream MCP server. It
// returns ErrDuplicate on a name collision. The remote func is bound to the
// tool's name at registration time.
func (r *Registry) RegisterRemote(def chat.Tool, fn RemoteFunc) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.entry[def.Name]; ok {
		return ErrDuplicate
	}
	name := def.Name
	r.entry[def.Name] = toolEntry{
		def: def,
		call: func(ctx context.Context, args json.RawMessage) (string, error) {
			return fn(ctx, name, args)
		},
	}
	return nil
}

// List returns the merged tool definitions (local + remote) sorted by Name. A
// fresh slice is returned on each call.
func (r *Registry) List() []chat.Tool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]chat.Tool, 0, len(r.entry))
	for _, e := range r.entry {
		out = append(out, e.def)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Name < out[j].Name })
	return out
}

// Call invokes a tool by name and returns its text result. An unknown name
// yields ErrToolNotFound. A nil or empty args is treated as `{}`.
func (r *Registry) Call(ctx context.Context, name string, args json.RawMessage) (text string, err error) {
	r.mu.RLock()
	e, ok := r.entry[name]
	r.mu.RUnlock()
	if !ok {
		return "", ErrToolNotFound
	}
	if len(args) == 0 {
		args = json.RawMessage(`{}`)
	}
	return e.call(ctx, args)
}

// Has reports whether name is registered.
func (r *Registry) Has(name string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	_, ok := r.entry[name]
	return ok
}