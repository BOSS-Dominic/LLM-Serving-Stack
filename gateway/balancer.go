package main

import (
	"errors"
	"fmt"
)

// ErrNoHealthyBackend is returned when every upstream is out of rotation or all
// remaining candidates have been excluded by the retry loop.
var ErrNoHealthyBackend = errors.New("no healthy backend available")

// Strategy selects how Acquire picks an upstream.
type Strategy string

const (
	// RoundRobin spreads requests evenly, weighted by Backend.Weight
	// (nginx-style smooth weighted round-robin).
	RoundRobin Strategy = "round_robin"
	// LeastConn picks the backend with the fewest in-flight requests per unit
	// of weight. For LLMs this converges much better than round-robin because
	// generation length varies by an order of magnitude between requests.
	LeastConn Strategy = "least_conn"
)

// ParseStrategy validates a strategy name from the config file.
func ParseStrategy(s string) (Strategy, error) {
	switch Strategy(s) {
	case RoundRobin, LeastConn:
		return Strategy(s), nil
	default:
		return "", fmt.Errorf("unknown strategy %q", s)
	}
}

// Acquire atomically selects a healthy backend and reserves one in-flight slot
// on it. "Select" and "reserve" happen under one lock, which is what makes
// least_conn correct under concurrency.
//
// exclude lets the retry loop avoid handing a failed request straight back to
// the backend that just failed it.
func (p *Pool) Acquire(strategy Strategy, exclude map[string]struct{}) (*Backend, error) {
	p.mu.RLock()
	healthy := make([]*Backend, 0, len(p.backends))
	for _, b := range p.backends {
		if b.Healthy.Load() {
			healthy = append(healthy, b)
		}
	}
	p.mu.RUnlock()

	if len(healthy) == 0 {
		return nil, ErrNoHealthyBackend
	}

	// Exclusion is a preference, not a hard filter. With a single backend (or
	// when every instance has failed once) retrying the same instance is still
	// better than answering 503, because the failure may have been transient.
	candidates := healthy
	if exclude != nil {
		filtered := make([]*Backend, 0, len(healthy))
		for _, b := range healthy {
			if _, skip := exclude[b.Name]; !skip {
				filtered = append(filtered, b)
			}
		}
		if len(filtered) > 0 {
			candidates = filtered
		}
	}

	p.selMu.Lock()
	defer p.selMu.Unlock()

	var pick *Backend
	switch strategy {
	case LeastConn:
		pick = pickLeastConn(candidates)
	default:
		pick = pickSmoothWeightedRoundRobin(candidates)
	}

	pick.Acquire()
	return pick, nil
}

// pickLeastConn returns the backend with the smallest (running+1)/weight.
// The comparison is cross-multiplied to stay in integer arithmetic.
func pickLeastConn(candidates []*Backend) *Backend {
	best := candidates[0]
	for _, b := range candidates[1:] {
		if lessLoaded(b, best) {
			best = b
		}
	}
	return best
}

// lessLoaded implements the least-connections comparison with two deliberate
// details:
//
//  1. Weight is folded in as (running+1)/weight, cross-multiplied.
//  2. Ties are broken by lifetime assignment count. Without this, two idle
//     backends compare equal, `candidate < best` is false for the second one,
//     and *every* request at low concurrency goes to the first backend. An
//     end-to-end run showed an 83/41 split before this tie-break existed.
func lessLoaded(a, b *Backend) bool {
	aw := int64(a.Weight)
	bw := int64(b.Weight)
	if aw <= 0 {
		aw = 1
	}
	if bw <= 0 {
		bw = 1
	}

	left := (a.Running.Load() + 1) * bw
	right := (b.Running.Load() + 1) * aw
	if left != right {
		return left < right
	}

	at, bt := a.TotalRequests.Load(), b.TotalRequests.Load()
	if at != bt {
		return at < bt
	}
	return a.Name < b.Name // deterministic final tie-break
}

// pickSmoothWeightedRoundRobin implements nginx's smooth weighted round-robin.
// Given weights 2:1 the pick order is A, A, B instead of A, A, B, A, A, B,
// which avoids bursts against the heavy backend. State lives on each Backend
// and is mutated only while Pool.selMu is held.
func pickSmoothWeightedRoundRobin(candidates []*Backend) *Backend {
	var best *Backend
	total := int64(0)

	for _, b := range candidates {
		w := int64(b.Weight)
		if w <= 0 {
			w = 1
		}
		total += w
		b.currentWeight += w
		if best == nil || b.currentWeight > best.currentWeight {
			best = b
		}
	}
	if best != nil {
		best.currentWeight -= total
	}
	return best
}
