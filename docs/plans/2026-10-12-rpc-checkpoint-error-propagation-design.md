# Propagate checkpoint failures through the engine RPC thread

PyTorch's distributed checkpoint `CheckpointException` inherits `BaseException`. The
engine worker currently catches only `Exception`. A checkpoint failure can therefore
terminate the worker while leaving the request's `Future` pending. The HTTP request then
waits indefinitely, hiding the failure from the controller.

Catch failures at the serial worker's task execution boundary. Preserve ordinary
`Exception` objects. Wrap other `BaseException` instances in `RuntimeError`, with the
original exception as its cause, and complete the pending `Future` with that error.
Existing HTTP handlers can then return their normal JSON 500 response. Keep the worker
alive so queued requests can complete. The queue's shutdown sentinel remains the
worker's explicit shutdown mechanism; task-local `SystemExit` and `KeyboardInterrupt`
become RPC errors rather than silently killing the worker.

Catching the error only in the checkpoint engine would leave other task failures
unhandled. Passing the original `BaseException` through the `Future` would still bypass
the HTTP handlers, which also catch only `Exception`.

Regression coverage uses the real PyTorch checkpoint exception, ordinary errors, queued
work, and the Flask endpoint with the actual engine thread. Tests bound their waits to
expose hangs. This change reports storage failures; restoring storage writes and
validating a complete checkpoint are separate recovery gates.
