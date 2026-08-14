package routing

import (
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/aibattery/router/internal/config"
)

// defaultCooldown is the default failure-cooldown window a model without an
// explicit cooldown uses.
const defaultCooldown = 30 * time.Second

// overrideNow swaps the package clock for the duration of a test.
func overrideNow(t *testing.T, fn func() time.Time) {
	t.Helper()
	orig := now
	now = fn
	t.Cleanup(func() { now = orig })
}

func cand(provider, model string) Candidate {
	return Candidate{ProviderName: provider, Model: model}
}

func TestFailoverBeginReturnsConfigOrder(t *testing.T) {
	sel := newSelector(config.ModelConfig{
		Name:     "m",
		Strategy: "failover",
		Candidates: []config.ModelCandidate{
			{Provider: "p1", Model: "m1"},
			{Provider: "p2", Model: "m2"},
			{Provider: "p3", Model: "m3"},
		},
	})

	got := sel.Begin()
	want := []Candidate{cand("p1", "m1"), cand("p2", "m2"), cand("p3", "m3")}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Begin() = %v, want %v", got, want)
	}
}

func TestFailoverCooldownExpiry(t *testing.T) {
	base := time.Now()
	overrideNow(t, func() time.Time { return base })

	sel := newSelector(config.ModelConfig{
		Name:     "m",
		Strategy: "failover",
		Candidates: []config.ModelCandidate{
			{Provider: "p1", Model: "m1"},
			{Provider: "p2", Model: "m2"},
		},
	})

	sel.RecordFailure(cand("p1", "m1"))

	got := sel.Begin()
	want := []Candidate{cand("p2", "m2")}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Begin() during cooldown = %v, want %v", got, want)
	}

	// Advance the clock past the cooldown window; p1 reappears.
	overrideNow(t, func() time.Time { return base.Add(defaultCooldown + time.Second) })

	got = sel.Begin()
	want = []Candidate{cand("p1", "m1"), cand("p2", "m2")}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Begin() after cooldown expiry = %v, want %v", got, want)
	}
}

func TestFailoverExhaustionReturnsEmpty(t *testing.T) {
	base := time.Now()
	overrideNow(t, func() time.Time { return base })

	sel := newSelector(config.ModelConfig{
		Name:     "m",
		Strategy: "failover",
		Candidates: []config.ModelCandidate{
			{Provider: "p1", Model: "m1"},
			{Provider: "p2", Model: "m2"},
		},
	})

	sel.RecordFailure(cand("p1", "m1"))
	sel.RecordFailure(cand("p2", "m2"))

	if got := sel.Begin(); len(got) != 0 {
		t.Fatalf("Begin() with all candidates in cooldown = %v, want empty", got)
	}
}

func TestRoundRobinRotation(t *testing.T) {
	sel := newSelector(config.ModelConfig{
		Name:     "m",
		Strategy: "round_robin",
		Candidates: []config.ModelCandidate{
			{Provider: "p1", Model: "m1"},
			{Provider: "p2", Model: "m2"},
			{Provider: "p3", Model: "m3"},
		},
	})

	first := sel.Begin()
	if first[0] != cand("p1", "m1") {
		t.Fatalf("first Begin() starts at %v, want p1/m1", first[0])
	}

	second := sel.Begin()
	if second[0] != cand("p2", "m2") {
		t.Fatalf("second Begin() starts at %v, want p2/m2 (rotation)", second[0])
	}

	// RecordSuccess advances the cursor past p2/m2, so the next call starts at p3/m3.
	sel.RecordSuccess(cand("p2", "m2"))

	third := sel.Begin()
	if third[0] != cand("p3", "m3") {
		t.Fatalf("third Begin() starts at %v, want p3/m3", third[0])
	}
}

func TestRoundRobinCooldownSkipsAndReappears(t *testing.T) {
	base := time.Now()
	overrideNow(t, func() time.Time { return base })

	sel := newSelector(config.ModelConfig{
		Name:     "m",
		Strategy: "round_robin",
		Candidates: []config.ModelCandidate{
			{Provider: "p1", Model: "m1"},
			{Provider: "p2", Model: "m2"},
			{Provider: "p3", Model: "m3"},
		},
	})

	sel.RecordFailure(cand("p2", "m2"))

	got := sel.Begin()
	want := []Candidate{cand("p3", "m3"), cand("p1", "m1")}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Begin() skipping cooldowned candidate = %v, want %v", got, want)
	}

	overrideNow(t, func() time.Time { return base.Add(defaultCooldown + time.Second) })

	got = sel.Begin()
	if len(got) != 3 {
		t.Fatalf("Begin() after cooldown expiry = %v, want all 3 candidates", got)
	}
}

func TestPerModelCooldownOverride(t *testing.T) {
	base := time.Now()
	overrideNow(t, func() time.Time { return base })

	sel := newSelector(config.ModelConfig{
		Name:     "m",
		Strategy: "failover",
		Cooldown: "2s",
		Candidates: []config.ModelCandidate{
			{Provider: "p1", Model: "m1"},
			{Provider: "p2", Model: "m2"},
		},
	})

	sel.RecordFailure(cand("p1", "m1"))

	got := sel.Begin()
	want := []Candidate{cand("p2", "m2")}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Begin() during 2s cooldown = %v, want %v", got, want)
	}

	// Advance past the 2s override; p1 reappears.
	overrideNow(t, func() time.Time { return base.Add(2*time.Second + time.Millisecond) })

	got = sel.Begin()
	want = []Candidate{cand("p1", "m1"), cand("p2", "m2")}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("Begin() after 2s cooldown expiry = %v, want %v", got, want)
	}

	// A model WITHOUT Cooldown uses the 30s default.
	base2 := time.Now()
	overrideNow(t, func() time.Time { return base2 })

	defSel := newSelector(config.ModelConfig{
		Name:     "m",
		Strategy: "failover",
		Candidates: []config.ModelCandidate{
			{Provider: "p1", Model: "m1"},
			{Provider: "p2", Model: "m2"},
		},
	})

	defSel.RecordFailure(cand("p1", "m1"))

	// Still skipped just before the 30s default window elapses.
	overrideNow(t, func() time.Time { return base2.Add(defaultCooldown - time.Second) })
	if got := defSel.Begin(); !reflect.DeepEqual(got, []Candidate{cand("p2", "m2")}) {
		t.Fatalf("Begin() before 30s default expiry = %v, want only p2", got)
	}

	// Reappears after the 30s default window.
	overrideNow(t, func() time.Time { return base2.Add(defaultCooldown + time.Second) })
	if got := defSel.Begin(); !reflect.DeepEqual(got, []Candidate{cand("p1", "m1"), cand("p2", "m2")}) {
		t.Fatalf("Begin() after 30s default expiry = %v, want both candidates", got)
	}
}

func TestRegistrySelectAndNames(t *testing.T) {
	reg, err := NewRegistry([]config.ModelConfig{
		{Name: "zebra", Strategy: "failover", Candidates: []config.ModelCandidate{{Provider: "p1", Model: "m1"}}},
		{Name: "alpha", Strategy: "round_robin", Candidates: []config.ModelCandidate{{Provider: "p1", Model: "m1"}}},
	})
	if err != nil {
		t.Fatalf("NewRegistry() error = %v", err)
	}

	if _, err := reg.Select("alpha"); err != nil {
		t.Fatalf("Select(alpha) error = %v", err)
	}

	if _, err := reg.Select("missing"); !errors.Is(err, ErrModelNotFound) {
		t.Fatalf("Select(missing) error = %v, want ErrModelNotFound", err)
	}

	names := reg.Names()
	want := []string{"alpha", "zebra"}
	if !reflect.DeepEqual(names, want) {
		t.Fatalf("Names() = %v, want %v", names, want)
	}
}

func TestNewRegistryRejectsEmptyCandidates(t *testing.T) {
	_, err := NewRegistry([]config.ModelConfig{
		{Name: "m", Strategy: "failover"},
	})
	if !errors.Is(err, ErrModelNotFound) {
		t.Fatalf("NewRegistry() error = %v, want ErrModelNotFound", err)
	}
}

func TestNewRegistryRejectsEmptyName(t *testing.T) {
	_, err := NewRegistry([]config.ModelConfig{
		{Name: "", Strategy: "failover", Candidates: []config.ModelCandidate{{Provider: "p1", Model: "m1"}}},
	})
	if !errors.Is(err, ErrModelNotFound) {
		t.Fatalf("NewRegistry() error = %v, want ErrModelNotFound", err)
	}
}