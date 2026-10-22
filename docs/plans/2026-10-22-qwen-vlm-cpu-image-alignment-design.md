# Qwen VL CPU image preprocessing alignment

## Behavior

Send original JPEG/PNG data to SGLang. Decode images with CPU/PIL, preserve SGLang's
native pixel budget and preliminary PIL resize, and retain native HF device dispatch
(GPU in the standard CUDA serving profile). The vision model continues to run on GPU.

Prepare training tensors in the rollout proxy using the same decoded pixels, resize
policy and effective HF bounds, using the CPU HF backend in the proxy. Megatron consumes
these already prepared `pixel_values` and `image_grid_thw`; it does not decode or resize
images again.

This supersedes the PNG transport strategy in the October 21 alignment plan.

## Alternatives

- Preparing a common PNG avoids GPU JPEG decoding but increases request bytes and
  introduces encoding work. Remove this transport conversion.
- Skipping SGLang's resize and removing its pixel limit changes inference input
  resolution. Preserve the native resolution policy instead.
- Keeping GPU decoding requires fixing Tensor-compatible token counting and reproducing
  the GPU pixel pipeline in the proxy. CPU/PIL is the selected path.

## Implementation

The existing model startup wrapper applies a guarded Qwen3-VL/Qwen3.5 source patch
before SGLang imports its worker classes. It disables GPU JPEG decoding and preserves
native HF device dispatch. If the previous CPU HF override was installed, it restores
the known original device branch. It does not change resize or pixel-limit code. Unknown
installed sources fail before any file is changed. Flash Next keeps its existing
separate preprocessing profile.

The proxy preserves SGLang's raw EXIF orientation semantics and converts decoded images
to RGB. It reproduces preliminary PIL resize and passes the effective pixel bounds to HF
per call. The shared processor configuration remains unchanged, so repeated requests
cannot divide its maximum pixel count again.

The existing per-image cache retains resized PIL images. Inference image strings remain
byte-for-byte identical to the request, including cached prompts and appended-image
histories. No PNG encoding is performed.

## Runtime contract

Proxy and SGLang must use the same processor implementation, model configuration,
pixel-limit setting, and resize filter. The selected path uses the native default
`SGLANG_PREPROCESS_USE_IMAGE_TENSOR=False`; explicitly requesting tensor-based loading
is outside this CPU/PIL contract. Server-side per-request image-size overrides are also
outside this contract.

The proxy and SGLang execute in the rollout runtime. The actor's HF version does not
itself change these image tensors, because it consumes the prepared tensors.

## Validation

Unit tests cover original request bytes, unchanged native resize and pixel bounds, CPU
proxy HF dispatch, native SGLang HF dispatch, EXIF handling, immutable shared
configuration, image-cache reuse, and rejection of unsupported source versions without
partial installation.

Use the actual rollout processor and native SGLang source for an offline CPU comparison
of small/large JPEGs, PNGs, and multi-image histories. Compare resized RGB pixels, final
`pixel_values`, grids, and prompt tokens, including generated history that does not
survive decode/retokenize unchanged. Measure preprocessing and request bytes separately
from model inference and training throughput.

A fixed-weight forward/logprob comparison and a controlled training run are separate
acceptance stages; preprocessing equality alone does not prove that reward collapse is
resolved.

## Verification results

The CPU unit suite passed with 288 tests, 3 skips, and 2 deselections. Full pre-commit
checks passed.

An offline replay using rollout Transformers 5.12.1 and the native SGLang resize
functions produced exactly equal RGB pixels, `pixel_values`, `image_grid_thw`, and
prompt IDs for six cases: over-budget JPEG, real history JPEG, multiple JPEGs, small
JPEG, EXIF-oriented JPEG, and RGBA PNG. A 2400 by 1800 image resized to 1152 by 864 on
both paths.

A real 25,800-token history replay preserved every original token with the selected
CPU/PIL path. The Tensor input path triggered the native fallback and returned 25,797
tokens. These are CPU contract replays, not a full server, forward/logprob, reward, or
training-throughput acceptance result.

## CPU/GPU boundary validation

With rollout Transformers 5.12.1, a scan of 164,616 original dimension pairs found
110,788 valid geometries, 52,202 native SGLang rejections, and 1,626 post-resize HF
aspect-ratio rejections. No valid geometry required another HF resize, and no
proxy/native preliminary resize geometry differed. This is a scalar geometry scan, not
164,616 image tensor tests.

Actual CPU/CUDA HF tensor comparisons covered 318 output sizes and 9 mixed-image and
grouping cases. All 327 comparisons were bitwise equal, including BF16, and tracing
observed no second resize. Full processor prompt IDs and grids matched for two
extreme-aspect cases and a mixed five-image batch.

An independent seven-case test that let HF itself resize unaligned inputs did produce
CPU/CUDA pixel differences, so the common preliminary PIL resize is an essential
condition. These tests support retaining GPU HF after that resize; they do not prove
equivalence under arbitrary processor versions or options.

A separate shared geometry boundary exists: an original 33 by 6566 image is within the
200:1 aspect limit, but native rounding produces 32 by 6560 (205:1), which HF rejects on
both CPU and GPU. CPU HF does not solve this boundary. This change preserves the native
geometry policy; fixing that boundary is separate.
