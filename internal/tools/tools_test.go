package tools

import (
	"context"
	"encoding/json"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/aibattery/router/internal/chat"
)

func fixtureLocal() chat.Tool {
	return chat.Tool{
		Name:        "echo_text",
		Description: "Echoes back the provided text.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"text": map[string]any{"type": "string"},
			},
			"required": []any{"text"},
		},
	}
}

func fixtureRemote() chat.Tool {
	return chat.Tool{
		Name:        "remote_upper",
		Description: "Uppercases text via a remote MCP server.",
		InputSchema: map[string]any{"type": "object"},
	}
}

// TestListSortedMerged verifies RegisterLocal + RegisterRemote then List
// returns the merged definitions sorted by Name.
func TestListSortedMerged(t *testing.T) {
	reg := New()
	if err := reg.RegisterRemote(fixtureRemote(), func(context.Context, string, json.RawMessage) (string, error) {
		return "", nil
	}); err != nil {
		t.Fatalf("RegisterRemote: %v", err)
	}
	if err := reg.RegisterLocal(fixtureLocal(), func(context.Context, json.RawMessage) (string, error) {
		return "", nil
	}); err != nil {
		t.Fatalf("RegisterLocal: %v", err)
	}

	defs := reg.List()
	if len(defs) != 2 {
		t.Fatalf("List len = %d, want 2", len(defs))
	}
	if defs[0].Name != "echo_text" || defs[1].Name != "remote_upper" {
		t.Fatalf("List not sorted by Name: %v, %v", defs[0].Name, defs[1].Name)
	}
	if defs[0].Description != fixtureLocal().Description {
		t.Errorf("echo_text description mismatch")
	}
	if defs[1].Description != fixtureRemote().Description {
		t.Errorf("remote_upper description mismatch")
	}
}

// TestCallDispatches verifies Call routes to the correct local and remote funcs.
func TestCallDispatches(t *testing.T) {
	reg := New()

	// Local: echo_text returns the input.
	if err := reg.RegisterLocal(fixtureLocal(), func(_ context.Context, args json.RawMessage) (string, error) {
		var in struct {
			Text string `json:"text"`
		}
		_ = json.Unmarshal(args, &in)
		return in.Text, nil
	}); err != nil {
		t.Fatalf("RegisterLocal: %v", err)
	}

	// Remote: stub asserts it receives name + args and returns canned text.
	var gotName string
	var gotArgs json.RawMessage
	if err := reg.RegisterRemote(fixtureRemote(), func(_ context.Context, name string, args json.RawMessage) (string, error) {
		gotName = name
		gotArgs = args
		return "CANNED", nil
	}); err != nil {
		t.Fatalf("RegisterRemote: %v", err)
	}

	// Local dispatch.
	text, err := reg.Call(context.Background(), "echo_text", json.RawMessage(`{"text":"hello"}`))
	if err != nil {
		t.Fatalf("Call echo_text: %v", err)
	}
	if text != "hello" {
		t.Errorf("echo_text = %q, want %q", text, "hello")
	}

	// Remote dispatch.
	text, err = reg.Call(context.Background(), "remote_upper", json.RawMessage(`{"x":1}`))
	if err != nil {
		t.Fatalf("Call remote_upper: %v", err)
	}
	if text != "CANNED" {
		t.Errorf("remote_upper = %q, want %q", text, "CANNED")
	}
	if gotName != "remote_upper" {
		t.Errorf("remote got name %q, want %q", gotName, "remote_upper")
	}
	if string(gotArgs) != `{"x":1}` {
		t.Errorf("remote got args %s, want %s", gotArgs, `{"x":1}`)
	}
}

// TestCallNilArgs treats nil/empty args as {}.
func TestCallNilArgs(t *testing.T) {
	reg := New()
	if err := reg.RegisterLocal(fixtureLocal(), func(_ context.Context, args json.RawMessage) (string, error) {
		if string(args) != "{}" {
			t.Errorf("args = %s, want {}", args)
		}
		return "ok", nil
	}); err != nil {
		t.Fatalf("RegisterLocal: %v", err)
	}
	if _, err := reg.Call(context.Background(), "echo_text", nil); err != nil {
		t.Fatalf("Call with nil args: %v", err)
	}
}

// TestCallUnknown verifies an unknown name yields ErrToolNotFound.
func TestCallUnknown(t *testing.T) {
	reg := New()
	if _, err := reg.Call(context.Background(), "nope", nil); !errors.Is(err, ErrToolNotFound) {
		t.Fatalf("Call unknown err = %v, want ErrToolNotFound", err)
	}
}

// TestDuplicateRegistration verifies name collisions yield ErrDuplicate.
func TestDuplicateRegistration(t *testing.T) {
	reg := New()
	fn := func(context.Context, json.RawMessage) (string, error) { return "", nil }
	if err := reg.RegisterLocal(fixtureLocal(), fn); err != nil {
		t.Fatalf("first RegisterLocal: %v", err)
	}
	if err := reg.RegisterLocal(fixtureLocal(), fn); !errors.Is(err, ErrDuplicate) {
		t.Fatalf("second RegisterLocal err = %v, want ErrDuplicate", err)
	}
	if err := reg.RegisterRemote(fixtureLocal(), func(context.Context, string, json.RawMessage) (string, error) {
		return "", nil
	}); !errors.Is(err, ErrDuplicate) {
		t.Fatalf("RegisterRemote collision err = %v, want ErrDuplicate", err)
	}
}

// TestHas covers Has() for registered and unregistered names.
func TestHas(t *testing.T) {
	reg := New()
	if reg.Has("echo_text") {
		t.Error("Has(echo_text) = true before registration")
	}
	if err := reg.RegisterLocal(fixtureLocal(), func(context.Context, json.RawMessage) (string, error) {
		return "", nil
	}); err != nil {
		t.Fatalf("RegisterLocal: %v", err)
	}
	if !reg.Has("echo_text") {
		t.Error("Has(echo_text) = false after registration")
	}
	if reg.Has("missing") {
		t.Error("Has(missing) = true")
	}
}

// TestConcurrentCall stresses Call from N goroutines to catch races under -race.
func TestConcurrentCall(t *testing.T) {
	reg := New()
	if err := reg.RegisterLocal(fixtureLocal(), func(_ context.Context, args json.RawMessage) (string, error) {
		var in struct {
			Text string `json:"text"`
		}
		_ = json.Unmarshal(args, &in)
		return in.Text, nil
	}); err != nil {
		t.Fatalf("RegisterLocal: %v", err)
	}

	const n = 64
	var wg sync.WaitGroup
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				if _, err := reg.Call(context.Background(), "echo_text", json.RawMessage(`{"text":"x"}`)); err != nil {
					t.Errorf("concurrent Call: %v", err)
					return
				}
				_ = reg.List()
				_ = reg.Has("echo_text")
			}
		}()
	}
	wg.Wait()
}

// TestRegisterBuiltin verifies the two built-in tools are wired and callable.
func TestRegisterBuiltin(t *testing.T) {
	reg := New()
	if err := RegisterBuiltin(reg); err != nil {
		t.Fatalf("RegisterBuiltin: %v", err)
	}
	if !reg.Has("echo_text") || !reg.Has("get_utc_time") {
		t.Fatalf("builtin tools not registered: %v", reg.List())
	}
	text, err := reg.Call(context.Background(), "echo_text", json.RawMessage(`{"text":"builtin"}`))
	if err != nil {
		t.Fatalf("Call echo_text: %v", err)
	}
	if text != "builtin" {
		t.Errorf("echo_text = %q, want %q", text, "builtin")
	}
	// Default "abc" when text absent.
	text, err = reg.Call(context.Background(), "echo_text", nil)
	if err != nil {
		t.Fatalf("Call echo_text nil: %v", err)
	}
	if text != "abc" {
		t.Errorf("echo_text default = %q, want %q", text, "abc")
	}
	// get_utc_time returns RFC3339.
	text, err = reg.Call(context.Background(), "get_utc_time", nil)
	if err != nil {
		t.Fatalf("Call get_utc_time: %v", err)
	}
	if _, err := time.Parse(time.RFC3339, text); err != nil {
		t.Errorf("get_utc_time = %q, not RFC3339: %v", text, err)
	}
	// Duplicate builtin registration -> ErrDuplicate.
	if err := RegisterBuiltin(reg); !errors.Is(err, ErrDuplicate) {
		t.Errorf("second RegisterBuiltin err = %v, want ErrDuplicate", err)
	}
}