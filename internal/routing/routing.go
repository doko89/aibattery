// Package routing implements the MODEL-AGGREGATION bounded context: it maps a
// virtual model name to an ordered attempt-list of concrete {provider, model}
// candidates, honoring each model's strategy ("failover" or "round_robin")
// with failure-aware cooldown behavior.
//
// The package is deliberately decoupled from the canonical chat domain: it
// depends only on config types and the standard library.
package routing

import (
	"errors"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/aibattery/router/internal/config"
)

// Candidate is one concrete provider+model the router may forward a call to.
type Candidate struct {
	ProviderName string
	Model        string // concrete upstream model name forwarded to that provider
}

// ErrModelNotFound is returned when a virtual model name is absent from the
// registry, or when a model config is empty/invalid.
var ErrModelNotFound = errors.New("routing: model not found")

// now is the clock used for cooldown decisions. It is a package-level variable
// so tests can override it to simulate time passing.
var now = time.Now

// Selector yields, per call, an ordered snapshot of candidates to attempt.
type Selector interface {
	// Begin returns the ordered candidates for THIS call as a snapshot.
	// failover: all candidates in config order. round_robin: config order
	// ROTATED so the current cursor is first. Candidates currently in a
	// failure cooldown are skipped (not returned).
	Begin() []Candidate
	// RecordFailure informs the Selector that c just FAILED. It places c into
	// a cooldown window (so Begin skips it) and advances the round_robin
	// cursor past c. Concurrency-safe.
	RecordFailure(c Candidate)
	// RecordSuccess marks c as having succeeded; for round_robin the cursor
	// advances past c so the next call starts differently. No-op for failover.
	RecordSuccess(c Candidate)
}

// selector implements Selector for a single virtual model.
type selector struct {
	mu         sync.Mutex
	strategy   string
	cooldown   time.Duration
	candidates []Candidate
	cursor     int
	cooldowns  map[Candidate]time.Time
}

// newSelector builds a selector from a validated ModelConfig.
func newSelector(m config.ModelConfig) *selector {
	cands := make([]Candidate, 0, len(m.Candidates))
	for _, c := range m.Candidates {
		cands = append(cands, Candidate{ProviderName: c.Provider, Model: c.Model})
	}
	cd, err := m.CooldownDuration()
	if err != nil {
		cd = 30 * time.Second
	}
	return &selector{
		strategy:   m.Strategy,
		cooldown:   cd,
		candidates: cands,
		cooldowns:  make(map[Candidate]time.Time),
	}
}

// Begin returns a fresh snapshot of eligible candidates. For failover the
// config order is preserved; for round_robin the list is rotated so the
// current cursor is first, and the cursor advances to the index right after
// the first returned candidate so successive calls naturally rotate.
func (s *selector) Begin() []Candidate {
	s.mu.Lock()
	defer s.mu.Unlock()

	n := len(s.candidates)
	if n == 0 {
		return []Candidate{}
	}

	nowT := now()
	out := make([]Candidate, 0, n)
	firstIdx := -1
	for i := 0; i < n; i++ {
		idx := (s.cursor + i) % n
		c := s.candidates[idx]
		if until, ok := s.cooldowns[c]; ok && nowT.Before(until) {
			continue
		}
		if firstIdx == -1 {
			firstIdx = idx
		}
		out = append(out, c)
	}

	if s.strategy == "round_robin" && firstIdx != -1 {
		s.cursor = (firstIdx + 1) % n
	}
	return out
}

// RecordFailure places c into a cooldown window and, for round_robin, advances
// the cursor past c.
func (s *selector) RecordFailure(c Candidate) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.cooldowns[c] = now().Add(s.cooldown)
	if s.strategy == "round_robin" {
		if idx := s.indexOf(c); idx >= 0 {
			s.cursor = (idx + 1) % len(s.candidates)
		}
	}
}

// RecordSuccess advances the round_robin cursor past c. No-op for failover.
func (s *selector) RecordSuccess(c Candidate) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.strategy == "round_robin" {
		if idx := s.indexOf(c); idx >= 0 {
			s.cursor = (idx + 1) % len(s.candidates)
		}
	}
}

// indexOf returns the config-order index of c, or -1 if not present.
func (s *selector) indexOf(c Candidate) int {
	for i, cand := range s.candidates {
		if cand == c {
			return i
		}
	}
	return -1
}

// Registry maps virtual model names to Selectors.
type Registry struct {
	mu        sync.RWMutex
	selectors map[string]Selector
}

// NewRegistry builds a Registry from model configs. It returns an error
// wrapping ErrModelNotFound if a model has an empty name or zero candidates.
func NewRegistry(models []config.ModelConfig) (*Registry, error) {
	selectors := make(map[string]Selector, len(models))
	for _, m := range models {
		if m.Name == "" {
			return nil, fmt.Errorf("%w: model with empty name", ErrModelNotFound)
		}
		if len(m.Candidates) == 0 {
			return nil, fmt.Errorf("%w: model %q has no candidates", ErrModelNotFound, m.Name)
		}
		selectors[m.Name] = newSelector(m)
	}
	return &Registry{selectors: selectors}, nil
}

// Select returns the Selector for the named virtual model, or an error
// wrapping ErrModelNotFound when the name is absent.
func (r *Registry) Select(name string) (Selector, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	sel, ok := r.selectors[name]
	if !ok {
		return nil, fmt.Errorf("%w: %q", ErrModelNotFound, name)
	}
	return sel, nil
}

// Names returns the sorted list of registered virtual model names.
func (r *Registry) Names() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()

	names := make([]string, 0, len(r.selectors))
	for name := range r.selectors {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}