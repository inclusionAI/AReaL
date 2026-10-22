# Preserve literal visual tokens in Qwen image conversations

Qwen can sample `image_pad`, `video_pad`, and visual delimiter IDs as text. A later
image request must preserve these sampled IDs while assigning visual features only to
real image spans. The HF processor previously expanded every matching string; the SGLang
transport similarly inferred images from every matching token run.

The OpenAI proxy protects visual strings originating in message/tool text with temporary
single-token markers on a request-local tokenizer copy. Typed image parts and template
source retain their normal behavior. The rendered prompt must restore exactly to the
original text. After HF processing, marker IDs restore one-to-one to the original IDs
with text modality. Shared processors and canonical history are unchanged. Existing
concat logic still preserves actual sampled prefixes and logps.

The proxy records inclusive real-image spans for the supported native Qwen3-VL and
Qwen3.5 families. Both SGLang adapters keep normal placeholder compaction for clean
requests. Ambiguous requests send full IDs, original image bytes, and explicit spans
through native `mm_sampling_kwargs`. Retry-appended tokens use the same check.

An AReaL subclass loads through SGLang's external multimodal processor interface,
automatically selected by the existing model bootstrap. It calls native image
preprocessing with an internal image-only prompt, then restores full model IDs and real
offsets. Native helpers validate grid lengths and reconstruct padding and mRoPE.
Per-request metadata is copied; image features retain native cache and IPC handling.
Unsupported mixed audio/video inputs fail explicitly on this path.

No sampled token is suppressed, deleted, replaced in model input, or retokenized. No
image re-encoding or pixel-tensor HTTP transport is introduced. Existing image
resize/device policy, visual freezing, and embedding-cache policy are retained. The new
adapter requires no additional SGLang source patch or user configuration.

Regression coverage includes literal visual tokens next to ordinary text, tools, custom
templates, unknown tokens, span validation, both inference adapters, and retry
transitions. Optional local-model integration checks real HF/SGLang IDs, features,
grids, offsets, mRoPE shape, and external processor registration.

This is a conditional correctness fix. Historical reward collapse has not been
attributed to generated visual special tokens; audited historical samples had no such
token hits. Processor tests also do not establish full training convergence.
