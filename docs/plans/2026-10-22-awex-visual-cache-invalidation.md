# Invalidate visual embeddings after AWEX cache flushes

## Problem and evidence

SGLang's multimodal embedding cache uses image hashes without a model weight version.
Its native scheduler cache flush clears KV state but leaves those embeddings intact.
Updating the visual encoder can therefore pair current language-model weights with
visual features computed using old weights.

A controlled Qwen 27B forward with a real later checkpoint confirmed that old features
survive weight updates and memory release/resume. Clearing only the image embedding
cache changed generation-token log probabilities under otherwise identical inputs and
weights. This establishes an inference consistency bug; it does not establish that the
bug alone caused a historical training collapse.

## Change

Extend the existing AReaL AWEX scheduler flush wrapper to clear the worker-local
multimodal embedding cache after the native flush succeeds unless all visual weights are
proven frozen. The wrapper already covers the scheduler, a separate weight updater, and
dispatcher callbacks. Every worker installs it; no new collective, configuration field,
environment setting, dependency, or SGLang source modification is needed.

For native Qwen VLMs, the training metadata publisher inspects the actual visual owner's
parameters, including the encoder and all projections. A nonempty, entirely frozen
module permits reuse across language-model weight versions. Missing ownership, unknown
metadata or any trainable visual parameter retains per-flush invalidation. Flash Next
uses its existing verified frozen-weight contract. No environment variable or new CLI
option is needed.

The first successful flush still clears embeddings once to establish the actor's
checkpoint, including recovery from a checkpoint different from the inference server's
initial load. Subsequent successful flushes preserve features only with the positive
freezing proof. Failed or busy flushes do not clear embeddings. The KV cache still
follows native flushing on every successful call.

Shared training metadata now lives in the dependency-light AWEX metadata module. Only
the Flash model path imports its specialized binding; the previous binding import path
remains available for compatibility.

## Validation

Regression tests exercise stale-feature invalidation, reuse after re-encoding, both
scheduler layouts, dispatcher registration, repeated installation, native flush
failure/exception handling, and an uninitialized cache. The plugin and pause/offload
regression suites passed all 65 tests.

A controlled full Qwen 27B probe used four real SGLang workers, original image hashes,
and a real later checkpoint. All four worker caches became empty after the fixed weight
update and re-encoded the same image keys with different feature fingerprints. On 16,097
generation positions, the fixed forward and the explicit fresh-encoding control were
bitwise equal; the stale/fixed comparison had mean absolute log-probability difference
0.025413 and maximum 3.124969. The probe used disk/tensor updates; AWEX update-hook
coverage was checked against the deployed reader source and regression tests. It was not
a training A/B or a reproduction of the historical collapse.

The fix covers servers using AReaL's AWEX plugin. Other server entry points need their
own update-hook coverage before they can claim the same guarantee.
