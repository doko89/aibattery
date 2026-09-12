package api

import (
	"context"

	"github.com/aibattery/router/internal/chat"
	"github.com/aibattery/router/internal/routing"
)

// completeCandidateEnv groups the failover context for one non-streaming
// candidate attempt so helper signatures stay small.
type completeCandidateEnv struct {
	sel             routing.Selector
	cand            routing.Candidate
	clientToolNames map[string]bool
	session         string
	virtualModel    string
}

// finishCandidate runs the shared post-completion tail of the non-streaming
// failover loops (OpenAI `complete` and Anthropic `completeAnthropic`): cache
// reasoning, execute server-owned tools internally, and record the outcome.
// It returns the response to serve and true, or nil and false when the
// candidate failed and the loop must fail over to the next candidate.
func (s *Server) finishCandidate(ctx context.Context, p chat.Provider, cReq *chat.ChatRequest, resp chat.ChatResponse, env completeCandidateEnv) (*chat.ChatResponse, bool) {
	s.rememberReasoning(env.session, &resp)
	final, err := s.executeServerTools(ctx, p, cReq, &resp, env.clientToolNames, env.session)
	if err != nil {
		s.deps.Logger.Warn("provider completion failed during tool loop",
			"virtual_model", env.virtualModel, "provider", env.cand.ProviderName, "model", env.cand.Model, "error", err)
		env.sel.RecordFailure(env.cand)
		return nil, false
	}
	env.sel.RecordSuccess(env.cand)
	s.deps.Logger.Info("completion served",
		"virtual_model", env.virtualModel,
		"provider", env.cand.ProviderName,
		"model", env.cand.Model)
	return final, true
}
