package api

import (
	"net/http"
	"time"
)

// handleModels serves GET /v1/models, returning the aggregated virtual model
// names registered in the routing registry.
func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	names := s.deps.Registry.Names()
	now := time.Now().Unix()
	data := make([]modelEntry, 0, len(names))
	for _, n := range names {
		data = append(data, modelEntry{ID: n, Object: "model", Created: now, OwnedBy: "aibattery"})
	}
	writeJSON(w, http.StatusOK, modelListResponse{Object: "list", Data: data})
}