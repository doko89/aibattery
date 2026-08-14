package chat

import "io"

// WriteStream adapts an io.Writer into a StreamFunc. It is a small convenience
// for adapters that receive streaming output as raw bytes (e.g. an SSE body
// reader) and want to forward each write as a StreamDelta.
type WriteStream struct {
	emit StreamFunc
}

// NewWriteStream returns a WriteStream that forwards every Write call to emit
// as a StreamDelta carrying the written bytes.
func NewWriteStream(emit StreamFunc) *WriteStream {
	return &WriteStream{emit: emit}
}

// Write implements io.Writer. It emits the given bytes as a single StreamDelta
// and returns the number of bytes written. If emit returns an error, Write
// returns that error with n set to 0.
func (w *WriteStream) Write(p []byte) (int, error) {
	if err := w.emit(StreamDelta{Delta: string(p)}); err != nil {
		return 0, err
	}
	return len(p), nil
}

var _ io.Writer = (*WriteStream)(nil)
