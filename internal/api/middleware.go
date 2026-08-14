package api

import (
	"context"
	"fmt"
	"log/slog"
	"math/rand"
	"net/http"
	"time"
)

type requestIDKey struct{}

// requestID injects a request ID into the context and echoes it back on the
// response header, generating one if the client did not supply it.
func requestID(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := r.Header.Get("X-Request-ID")
		if id == "" {
			id = fmt.Sprintf("%x", rand.Uint64())
		}
		w.Header().Set("X-Request-ID", id)
		ctx := context.WithValue(r.Context(), requestIDKey{}, id)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// logging logs one line per request with method, path, status and duration.
func logging(logger *slog.Logger) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			start := time.Now()
			sw := &statusWriter{ResponseWriter: w, status: http.StatusOK}
			next.ServeHTTP(sw, r)
			logger.Info("http request",
				"method", r.Method,
				"path", r.URL.Path,
				"status", sw.status,
				"duration", time.Since(start).String(),
				"request_id", r.Context().Value(requestIDKey{}),
			)
		})
	}
}

// recovery recovers from panics in downstream handlers and returns a 500.
func recovery(logger *slog.Logger) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			defer func() {
				if rec := recover(); rec != nil {
					logger.Error("panic recovered",
						"panic", rec,
						"request_id", r.Context().Value(requestIDKey{}),
					)
					writeError(w, http.StatusInternalServerError, "internal server error", "server_error", "")
				}
			}()
			next.ServeHTTP(w, r)
		})
	}
}

// statusWriter records the response status code for logging.
type statusWriter struct {
	http.ResponseWriter
	status int
}

func (w *statusWriter) WriteHeader(code int) {
	w.status = code
	w.ResponseWriter.WriteHeader(code)
}

// Flush forwards to the underlying writer when it supports flushing, so the
// statusWriter can wrap SSE handlers without breaking http.Flusher.
func (w *statusWriter) Flush() {
	if f, ok := w.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}