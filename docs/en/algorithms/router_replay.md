# Rollout Routing Replay (R3) for MoE Models

Last updated: Apr 29, 2026

## Overview

In asynchronous RL for Mixture-of-Experts (MoE) models, the policy that generates
rollouts (served by SGLang) and the policy that is being trained (driven by
Megatron-LM) may differ by one or more parameter versions.  Since the router is a
*learned* sparse gate, even small weight drift can send the same token to different
experts between inference and training, producing a **train/inference routing
mismatch** that corrupts importance-sampling ratios and destabilises optimisation.

**Rollout Routing Replay (R3)** eliminates this mismatch by:

1. Recording the per-token expert assignments emitted by the inference engine for
   every decoded token.
2. Re-using (*replaying*) those exact expert assignments during the training
   forward / backward pass in place of the routing computed from current weights.

R3 is inspired by the implementation in
[verl](https://github.com/volcengine/verl) and has been adapted for AReaL's
Megatron backend + SGLang bridge-mode inference service.

## Supported Configurations

| Dimension | Supported | Notes |
|---|---|---|
| Training backend | Megatron-LM (`MegatronEngine`) | FSDP engine is **not** supported. |
| Inference backend | SGLang 0.5.9 (bridge mode) | vLLM not supported. |
| Tensor Parallel (**TP**) | ✅ | Uses `scatter_to_sequence_parallel_region` to distribute packed router indices to SP ranks. |
| Expert Parallel (**EP**) | ✅ | Patched `MoEAlltoAllTokenDispatcher.preprocess` recomputes `num_out_tokens = routing_map.sum()` so that the dropless path stays correct when replay zeroes padding rows. |
| Pipeline Parallel (**PP**) | ✅ | `RouterReplayHelper.get_micro_batch_router_list` slices `RouterReplay.router_instances` according to the current PP rank's `(layer_offset, num_layers)`. |
| Virtual Pipeline Parallel (**VPP**) | ✅ | Same helper honours `virtual_pipeline_model_parallel_size` and iterates over VP stages. |
| Context Parallel (**CP**) | ⚠️ Experimental | `seq_align_to = tp_size * cp_size * 2` is applied when `cp_size > 1`; exercised only via unit tests, not covered by the provided E2E fixtures. |
| Data Parallel (**DP**) | ✅ | R3 runs independently per DP replica; no cross-DP communication is added. |
| Dense + MoE hybrid layers | ✅ | `is_moe_layer()` uses `moe_layer_freq` / `first_k_dense_replace` so dense layers are skipped from replay. |
| Role | Actor only | `config.actor.megatron.enable_router_replay` is set exclusively on the actor engine; Critic / Reference / Teacher engines are unaffected. |
| Capacity factor | `moe_expert_capacity_factor is None` (dropless) | Replay only overrides `num_out_tokens` on the dropless path, matching verl's guard. |
| FP8 / quantisation padding | ❌ | Replay is skipped when `moe_router_padding_for_fp8` or `moe_router_padding_for_quantization` is enabled to preserve FP8 dispatch correctness. |
| Vision / multimodal models | ❌ | No hooks in the VLM path. |

## How to Enable R3

R3 is driven by a single rollout flag; everything else is wired automatically by
`areal/trainer/rl_trainer.py`.

```yaml
rollout:
    # Request per-token routed expert indices from SGLang.
    return_routed_experts: true

actor:
    backend: "megatron:(attn:d1p1t4|ffn:d1p1t1e4)"   # TP=4, EP=4
    # actor.megatron.enable_router_replay is forced to True
    # automatically when rollout.return_routed_experts=true.

sglang:
    # R3 relies on per-token tokens being aligned with the routing
    # output. The trainer forces skip_tokenizer_init=True at startup;
    # declaring it here makes the intent explicit.
    skip_tokenizer_init: true
    enable_return_routed_experts: true
```

At trainer startup (`RLTrainer.__init__`):

1. `rollout.return_routed_experts=True` causes
   `config.actor.megatron.enable_router_replay` to be set to `True`.
2. `num_moe_layers` and `topk` are auto-resolved from the HuggingFace config
   (`num_experts_per_tok`, `num_hidden_layers`, `moe_layer_freq`,
   `first_k_dense_replace`) by `resolve_r3_moe_config()`.
3. `sglang.skip_tokenizer_init` is forced to `True` (warning printed if the user
   set it to `False`) to prevent tokenizer round-trip token shifts that would
   break per-token routing alignment.
4. The SGLang bridge entrypoint
   (`areal/experimental/inference_service/sglang/launch_server.py`) calls
   `apply_sglang_r3_patch()` so that `TokenizerManager._handle_batch_output`
   base64-encodes the `routed_experts` tensor before FastAPI serialisation.
5. On the training side, `MegatronEngine.initialize()` calls
   `apply_router_replay_patch()` (monkey-patches `TransformerConfig.__init__`,
   `TopKRouter.__init__`, `TopKRouter.routing` and
   `MoEAlltoAllTokenDispatcher.preprocess`) **before** model creation, and then
   wraps the engine with `patch_megatron_engine_for_r3()`.

## Pipeline Overview

```
┌────────────────────────────┐      ┌──────────────────────────────┐
│  SGLang inference server   │      │   MegatronEngine (actor)      │
│                            │      │                                │
│ generate_logprobs()        │      │ forward_backward_batch()       │
│  └─ routed_experts tensor  │───▶  │  ├─ REPLAY_FORWARD on           │
│     (base64 over HTTP)     │      │  │   each microbatch            │
└────────────────────────────┘      │  ├─ post-forward hook switches  │
                                    │  │   to REPLAY_BACKWARD          │
                                    │  └─ clear_router_replay()        │
                                    └──────────────────────────────┘
```

### Key data structures

| Object | Purpose |
|---|---|
| `RouterReplay` (per MoE layer) | Holds the replay indices (`target_topk_idx`), recording buffer (`recorded_topk_idx`), and current `RouterReplayAction`. |
| `RouterReplay.router_instances` | Class-level list, one entry per MoE layer *on the local rank*. Cleared each time `apply_router_replay_patch()` is called. |
| `RouterReplayAction` | Enum: `RECORD`, `REPLAY_FORWARD`, `REPLAY_BACKWARD`. |
| `RouterReplayHelper.get_micro_batch_router_list()` | Returns the subset of `router_instances` assigned to the current `(pp_rank, vp_stage)`. |
| `setup_per_microbatch_replay_forward()` | Called before each micro-batch forward: aligns rollout-format `routed_experts` to the training token layout, packs with `cu_seqlens`, scatters to SP ranks, and distributes to the per-layer `RouterReplay` instances. |

### Correctness notes

* **`num_out_tokens` override.** Megatron-Core 0.16.0's dropless branch of
  `MoEAlltoAllTokenDispatcher.preprocess` sets
  `num_out_tokens = routing_map.size(0) * moe_router_topk` as a static value.
  When R3 zeroes padding rows in `routing_map`, that static value overcounts,
  so the patched preprocess computes `num_out_tokens = int(routing_map.sum().item())`
  on the dropless path. The ~3,500 `.item()` syncs per training step are
  negligible compared to MoE compute.
* **Per-instance `__class__` swap.** The micro-batch iterator wraps
  `MicroBatchList` with a dynamic subclass assigned via `mb_list.__class__`,
  not by mutating the shared class, so concurrent engines (e.g. critic) are
  not affected.
* **Left-align from right-padded rollouts.** `_align_routed_experts_to_mask()`
  converts the rollout tensor from `(bs, batch_max_seqlen, L, K)` right-padded
  format to a training-oriented left-aligned layout using `cu_seqlens`.
* **Silent-drop removed.** When a micro-batch cannot be exactly split by
  `bs // n_mbs`, R3 raises instead of silently trimming rows.

## Minimal Example

See `examples/math/moonlight_16b_a3b_gsm8k_grpo_megatron.yaml` for the reference
Moonlight-16B-A3B configuration (PP=2, TP=4, EP=4, 8 GPUs).  Launch:

```bash
python3 examples/math/gsm8k_rl.py \
    --config examples/math/moonlight_16b_a3b_gsm8k_grpo_megatron.yaml \
    scheduler.type=local
```

On a single-node 8×H200 system the `*_h20.yaml` variant runs with PP=1,
TP=4, EP=4 and `max_tokens_per_mb=10240`.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `[R3] Number of replay tensors (...) does not match number of router instances (...)` | MoE layer count resolved from HF config differs from Megatron's per-rank layer count (usually due to `first_k_dense_replace` / `moe_layer_freq` mismatch or custom pipeline layout). | Verify `num_hidden_layers`, `first_k_dense_replace`, and `moe_layer_freq` in the model's `config.json` and that `pipeline_model_parallel_layout` (if set) matches the MoE layer count. |
| SGLang returns `routed_experts: {}` (empty dict) | Inference server was started without the R3 patch. | Ensure you are using the bridge entrypoint `areal.experimental.inference_service.sglang.launch_server`; it installs `apply_sglang_r3_patch()` automatically. |
| `moe_router_padding_for_fp8=True` + R3 | R3 is intentionally disabled on FP8 padding paths. | Either turn off FP8 router padding or disable `rollout.return_routed_experts`. |
| Critic does not pick up R3 | By design; only the actor is patched. | If a future use-case needs MoE critic replay, extend `rl_trainer._amend_xccl_weight_update_envvar` and `MegatronEngine._r3_enabled` plumbing. |

## References

* PR [#1207](https://github.com/inclusionAI/AReaL/pull/1207) — `[WIP]feat: add router replay for megatron engine`.
* verl router replay:
  [`volcengine/verl`](https://github.com/volcengine/verl) (`verl/workers/**/*router_replay*`).
* Megatron-Core MoE parallel folding:
  [NVIDIA/Megatron-LM MoE README](https://github.com/NVIDIA/Megatron-LM/tree/main/megatron/core/transformer/moe).
