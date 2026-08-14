// Command server is the entrypoint of the AI router. It loads configuration,
// wires the application, and serves HTTP until it receives an interrupt or
// SIGTERM, at which point it shuts down gracefully.
package main

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/aibattery/router/internal/app"
	"github.com/aibattery/router/internal/config"
)

func main() {
	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))

	path := os.Getenv("ROUTER_CONFIG")
	if path == "" {
		path = "config.yaml"
	}

	cfg, err := config.Load(path)
	if err != nil {
		logger.Error("load config", "path", path, "error", err)
		os.Exit(1)
	}

	a, err := app.New(cfg)
	if err != nil {
		logger.Error("wire app", "error", err)
		os.Exit(1)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	runErr := make(chan error, 1)
	go func() {
		runErr <- a.Run()
	}()

	select {
	case err := <-runErr:
		if err != nil && !errors.Is(err, http.ErrServerClosed) {
			logger.Error("server failed", "error", err)
			os.Exit(1)
		}
	case <-ctx.Done():
		logger.Info("shutdown signal received")
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := a.Shutdown(shutdownCtx); err != nil {
			logger.Error("graceful shutdown failed", "error", err)
			os.Exit(1)
		}
		logger.Info("server stopped")
	}
}