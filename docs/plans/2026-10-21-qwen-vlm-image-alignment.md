# Qwen-VL image alignment without SGLang source changes

Based on upstream `main` at `18f5271f3`. Targets Qwen3-VL and Qwen3.5 image requests
through the AReaL OpenAI proxy with SGLang 0.5.18. The processor class for Qwen3.8-27B
is `Qwen3VLProcessor`.

The proxy previously processed original images with HF, retained the resulting pixels
and expanded tokens for training, and sent the originals to SGLang. SGLang independently
limits the image budget, resizes with PIL or a bilinear tensor kernel, then calls HF.
Equal token counts do not imply equal pixels. The default Qwen3 image limit also reduces
a 16M-pixel model budget to 1M.

For example, a 1600 x 1200 source becomes 1600 x 1216 with 1900 image tokens in the
original training path, but 1152 x 864 with 972 image tokens in SGLang. The answer's
training log probability and rollout log probability then condition on different image
inputs and text positions. Their difference can reflect the changed input, even before
an optimizer update. Smaller images can have matching token counts while still differing
in pixels because the resize kernels differ.

The AReaL input fix prepares a common representation before generating image tokens:
resize once to SGLang's final dimensions, convert to RGB, and send a lossless PNG while
retaining tensors produced from the same image for training. SGLang's extra resize then
preserves the pixels, including for JPEG sources. This intentionally supplies SGLang
with the common PNG instead of requiring AReaL to reproduce its CUDA JPEG decoder.

Alignment is the default for `Qwen3VLProcessor` requests using a SGLang backend. No
opt-in environment variable is required. Backend detection supports configured
controllers, legacy engines, and v2 bridges. Other backends and processor families
retain their existing image processing. Direct `VisionRLVRWorkflow` calls are outside
this OpenAI/Arena entry-point fix.

The proxy and SGLang must inherit the same `SGLANG_VLM_QWEN_LIMIT_PIXEL` value (default
true). `SGLANG_RESIZE_RESAMPLE` is also respected. Use an unmodified HF processor and no
server image-size overrides. The fix preserves SGLang's current pixel budget; it does
not increase inference image resolution. Restart the proxy rather than changing these
existing SGLang settings with a populated cache.

Chat Completions, Responses, and cached/concat prompt preparation retain the canonical
wire images with the training tensors. Both legacy and v2 SGLang adapters send one
placeholder per image; v2 resubmission preserves this contract while retaining sampled
text tokens. Context limits still use the expanded training prompt length. RGB
conversion and PNG encoding run in the existing processor thread. PNG can increase
request size compared to JPEG; preprocessing and transport performance require
measurement.

CPU regression tests cover image size boundaries, RGB/RGBA/grayscale and JPEG inputs,
lossless transport, multiple images, cached prompts, both API request paths, and v2
request/resubmission contracts: 141 passed, one skipped. Actual runtime source isolation
compares both SGLang resize branches against the training resize/patchification path.
Four real-processor probes on an L20X use normal/large PNG, JPEG, and multiple JPEGs:
complete input IDs and grids match, float32 pixel differences are zero, and CPU/CUDA
processor outputs remain equal after BF16 conversion. These use the unmodified rollout
SIF, not a patched live service. Full-model logp, CP4 execution, and post-update weight
equality still require separate validation.

This fix does not invalidate SGLang's visual embedding cache after a vision weight
update. If vision weights train, separately disable that cache with
`SGLANG_VLM_CACHE_SIZE_MB=0` on the SGLang process, or implement version-aware
invalidation. Disabling the Qwen pixel limit alone only addresses image token counts; it
does not remove the independent decoding and resizing differences.
