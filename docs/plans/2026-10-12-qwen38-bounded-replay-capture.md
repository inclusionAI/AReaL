# Bounded Flash Next replay capture

The current rollout JSONL stores decoded text and version run lengths. It cannot
reconstruct original token IDs, masks, per-token behavior logp or processed vision
tensors for an exact replay, especially when decoding contains replacement characters.

Reuse the debug branch's existing lossless batch snapshot rather than adding another
proxy export or a second serialization format. Extra exports consume active sessions;
decoded-text reconstruction loses original bytes. Snapshotting every batch also adds
unbounded disk use and controller copies.

Keep capture opt-in through `QWEN_BATCH_SNAPSHOT_DIR`. Bound launcher capture to the
first call per preparation/advantage method by default, with a nonnegative optional
per-method call limit. Zero disables capture. Preserve the helper's unlimited default
for direct diagnostic callers. The existing atomic, non-overwriting publication and safe
tensor loading remain in place.

Verify exact IDs, masks, versions, behavior logp and processed pixels round-trip without
mutating remote wrappers; repeated image tensors retain aliases. Verify the cap stops
writes while all training calls continue, zero performs no wrapping, and invalid caps
fail before actor mutation. Run these CPU tests in both actual runtime images and the
full cached pre-commit suite.

This change stays on `chucai.dzq/debug-qwen38-flash-logp`. Do not alter or restart the
healthy production run to activate it. Captures still contain whole batches and have
memory, disk and synchronous I/O costs. They do not restore optimizer/RNG state, prove
weight-version alignment or produce an absent paired Megatron logp tensor. Those must be
verified separately before attributing logp differences to an engine or kernel.
