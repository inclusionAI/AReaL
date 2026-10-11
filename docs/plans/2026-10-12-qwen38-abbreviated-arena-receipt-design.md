# Arena abbreviated runner result compatibility

The running Flash Next trial rejected three zero-scored model failures as unknown in the
04:48–05:49 inspection interval. Their native Harness receipts report one failed turn,
`rust-core-runtime`, exit code 1, no delivery, a successful collection, and exactly
`model stopped without visible output or tool calls`. The task-result envelope contains
only a truncated `raw.error`, including the runner lifecycle; two tasks additionally
carry the `adopted harness` prefix.

AReaL identifies result families before fetching or validating native receipts. The
error-only wrapper currently has no recognized family, so these tasks never fetch their
native receipt. This prevents the existing zero-reward path from assessing them.

Two options are correcting Arena to always preserve the structured Harness envelope, or
recognizing this observed abbreviated format in AReaL and fetching the authoritative
receipt. The local compatibility change can be validated without changing the Arena
service. It must not synthesize missing metadata.

Recognize only an error-only dictionary with the agent exit-code-1 prefix and a runner
lifecycle line. Preserve family-conflict handling and the existing rules for other
result formats. Require score zero, successful native collection with a nonnegative
integer file count, the existing implementation/status/exit/turn and delivery checks,
and one allowlisted model failure. Explicit system errors in the available log remain a
veto. Context overflow alone cannot admit this format.

The proxy's sticky service-error veto, positive interaction count and successful
interaction export remain necessary. A zero-score classification does not prove group
admission or actor consumption. Tool-argument budget failures, public-input changes and
conflicting deliverables retain their current disposition.

Validate the observed envelopes, missing/malformed receipt fields, conflicting families,
infrastructure signals, collection failures, zero interactions and the actual
asynchronous receipt-fetch path. Replay all 14 privately saved cases offline with both
baseline and candidate classifiers. Run the existing Arena tests and full pre-commit
suite in the actual runtime environment.

This branch includes the latest MM/EOS fixes and the prepared required-recovery guard.
Do not deploy it by restarting the healthy training run. Online receipt fetch,
successful export and actual actor consumption remain unverified until a necessary
recovery or an independently authorized diagnostic run uses the change.
