# GLM-5.3-Flash bring-up

This is an experimental text actor integration for `Glm5NextForConditionalGeneration`
(`glm5_next`, nested `glm5_next_text`). CPU architecture tests and focused single-GPU
kernel tests are available. Full-model GPU execution, distributed training, HF reload,
SGLang logps, routing replay, and live weight transfer still require validation before
production RL.

## Architecture and runtime

The source model has 45 layers: 34 KDA and 11 NoPE DSA layers. DSA appears at zero-based
indices 3, 7, ..., 43; layer 44 is KDA. The first three MLPs are dense and the remaining
42 are MoE. Each layer uses four mHC residual streams, 20 Sinkhorn iterations,
`hc_eps=1e-6`, and `rms_norm_eps=1e-5`. SwiGLU clamps at 10. The frozen DSA indexer uses
four-token key pools and always includes the causal unpooled tail.

Use `bridge_type: mbridge`, BF16 actor compute, CP=1, packed THD inputs, and
`use_mbridge_save: false`. KDA is replicated across TP; sequence-parallel gather/scatter
does not shard its weights. DSA uses native TP projections. Full activation recompute is
supported by the integration; the specialized mHC recompute arena, MTP, VPP, custom
pipeline layouts, critic training, and LoRA are outside this initial scope.

The repository's pinned stock Megatron-Core 0.19 lacks the required mHC interfaces. The
implementation targets the interfaces in
[`radixark/Megatron-LM@e8f57451`](https://github.com/radixark/Megatron-LM/tree/e8f57451):
native mHC layer slots, configurable block head contraction, FP32 parameter retention,
and SwiGLU clamping. The layer builder rejects runtimes missing these capabilities.
Dependency pins are unchanged. An explicitly selected runtime also needs Transformer
Engine, FLA's KDA/short-convolution/gated RMSNorm APIs, Triton, and TileLang. FlashMLA
is optional. Kernel sources are adapted from Apache-2.0
[`radixark/miles@ee36bb2`](https://github.com/radixark/miles/tree/ee36bb2), with
provenance in each file; Miles is not a runtime dependency.

## CPU checks

```bash
python examples/glm5_next/validate.py audit "$GLM53_MODEL"
pytest -q tests/test_glm5_next.py tests/test_glm5_next_dsa.py \
  tests/test_glm5_next_checkpoint.py tests/test_glm5_next_bridge.py
```

The audit reads raw JSON without loading model weights or importing AReaL/CUDA. Add
`--check-runtime` inside the intended training environment to check interfaces, import
the complete Megatron engine, and construct the bridge configuration. This does not
allocate model weights or execute kernels. Actual mbridge conversion tests skip when
mbridge or a compatible mHC runtime is unavailable; the CPU DSA test uses real Torch
projections and compares absorption against explicit dense MLA, including gradients.

The isolated CPU runtime uses the MCore revision linked above and mbridge
`310e8fb35ccf4fcd4419d32973e563a6d43ee5fb`, with Python 3.12, Torch 2.9.1+cu129,
Transformer Engine 2.14.1, FLA 0.4.2, Triton 3.5.1, TileLang 0.1.9, and Transformers
5.3.0. These versions describe the development test environment; full-model and
distributed execution still need the gates below.

The checkpoint's FP8 storage is distinct from BF16 actor compute. Its root
`quantization_config` is retained; legacy `model.layers.*` exclusions are expanded to
`model.language_model.layers.*`. Both fast save and online conversion honor these
exclusions, including BF16 `kv_b_proj`. KDA Q/K/V convolutions merge into a replicated
actor parameter and split on export. mHC scale remains one FP32 tensor with three
entries, avoiding a lossy streaming reconstruction.

The actor builds only the text tower. HF export copies the source's frozen
`model.visual.*` tensors into separate shards, retains the multimodal config, zeros MTP
layer counts, and repairs the index. The original local source checkpoint must remain
accessible; export refuses to overwrite it. The native mbridge save option is rejected
because it would retain FP8 metadata while writing dequantized text weights.

## Four-layer SFT smoke

Prepare a smaller checkpoint on CPU, including KDA, DSA, one MoE layer, and the original
vision tower. This copies selected tensors and consumes disk space; the destination must
be new. It does not alter the source checkpoint or reduce hidden/expert sizes.

```bash
python examples/glm5_next/validate.py prepare-smoke "$GLM53_MODEL" "$GLM53_SMOKE"
python examples/glm5_next/validate.py audit "$GLM53_SMOKE" --check-runtime
```

`sft_smoke.yaml` uses the existing deterministic profile dataset for three optimizer
steps and an HF save. Fill its mandatory paths, node/GPU counts, and allocation from the
actual execution environment; keep CP=1 and do not enable VPP. The topology and private
environment must be checked before allocating hardware. On an already allocated local
execution environment, the entry point is:

```bash
python -m areal.infra.launcher.local examples/profile/train_sft_profile.py \
  --config examples/glm5_next/sft_smoke.yaml \
  actor.path="$GLM53_SMOKE" \
  allocation_mode="$GLM53_ALLOCATION" \
  cluster.n_nodes="$GLM53_NODES" \
  cluster.n_gpus_per_node="$GLM53_GPUS_PER_NODE" \
  cluster.fileroot="$GLM53_OUTPUT" \
  cluster.name_resolve.nfs_record_root="$GLM53_NAME_ROOT"
```

Check finite loss/grad norms, step completion, checkpoint index/config consistency, and
reload. The synthetic repeated sample is a plumbing test; its loss is not a quality
metric. Validate TP=1 first, then TP>1/SP and EP, checking replicated KDA gradients and
FP32 mHC optimizer state across ranks.

## Numerical and RL gates

1. Run `pytest -q -m slow tests/test_glm5_next_kernels.py` on CUDA. This checks packed
   kpool causality and sparse MLA forward/backward against a dense oracle.

   The gradient oracle uses FP32 leaves. An independent dense BF16/autograd baseline
   bounds maximum and RMS error to twice the baseline error, with an additional 1%
   relative RMS limit. This accounts for cancellation in BF16 probability/value-gradient
   products while bounding both the worst element and aggregate numerical error. Two
   random seeds exercise the packed NoPE case.

1. Perform a no-update HF → Megatron → HF roundtrip. Check tensor names, shapes,
   quantization exclusions/scales, mHC scale precision, and untouched vision tensors.
   FP8 requantization is not byte-exact; compare dequantized tensors and logps.

1. Capture HF, Megatron, and SGLang logps for identical token IDs and scoring masks,
   with sampling disabled and equal model weights. Save each as NPZ arrays `token_ids`,
   `loss_mask`, and `logprobs`. Compare each pair with an explicitly chosen tolerance:

   ```bash
   python examples/glm5_next/validate.py compare-logps reference.npz candidate.npz \
     --atol 0.05
   ```

   The tool rejects mismatched token IDs/masks, non-finite scored logps, and empty
   comparisons. It reports mean, p99, and maximum absolute differences and exits nonzero
   on failure. The example tolerance must be calibrated to the reference runtime and
   quantization mode; it is not evidence of numerical agreement.

1. Validate native MoE routing replay on the 42 sparse layers. DSA indexer replay is not
   implemented. Compare routes and logps separately before enabling R3 in RL.

1. Validate an actual distributed weight update and receiver cache refresh. The sender
   uses the GLM bridge for names, convolution splits, and mHC tensors, then applies the
   checkpoint's FP8 policy. Receiver loading is still a GPU validation gate, especially
   for absorbed MLA weights.

1. Only then run the full-model SFT/RL sequence. No training job is submitted by these
   preparation or comparison commands.
