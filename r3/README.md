# R3 Router Replay

R3 (Rollout Routing Replay) records MoE router choices during rollout and replays those
choices during Megatron training. It is for SGLang rollout + Megatron MoE actor runs
where rollout logprobs and training recompute logprobs can diverge because the two
engines choose different top-k experts.

The implementation uses Megatron-Core native `RouterReplay`. AReaL does not replace
Megatron router math. It transports the routing side channel, validates it, converts it
to Megatron token order, and drives native replay actions around each micro-batch.

## Implementation

End-to-end data flow:

1. **Rollout recording**: `rollout.return_routed_experts=true` makes AReaL send
   `return_routed_experts=True` to SGLang. SGLang returns base64 int32 routed expert ids
   in `meta_info["routed_experts"]`; AReaL decodes them into
   `[tokens, num_moe_layers * topk]`.
1. **Workflow preprocessing**: supported workflows create two trajectory tensors:
   `routed_experts` with shape `[batch, seq_len, num_moe_layers, topk]` and
   `r3_routing_valid` with shape `[batch]`. Legal missing tail rows are filled from the
   nearest previous valid row. Internal zero rows or unaligned samples are marked
   invalid instead of being replayed.
1. **Megatron replay**: before each Megatron micro-batch, AReaL packs routing into
   Megatron's padded packed-token order on CPU after selecting the current PP/VPP
   stage's MoE layers, applies CP/SP slicing on CPU, transfers only contiguous local
   slabs, calls native `set_target_indices()`, and switches native `RouterReplayAction`
   to `REPLAY_FORWARD`. For backward recompute it switches to `REPLAY_BACKWARD`, so
   activation checkpointing consumes the same router choices.

Key files:

| File                               | Responsibility                                                                                  |
| ---------------------------------- | ----------------------------------------------------------------------------------------------- |
| `areal/engine/r3/config.py`        | Resolve MoE layer layout and router top-k from model config.                                    |
| `areal/engine/r3/preprocess.py`    | Decode and normalize SGLang routed expert tensors.                                              |
| `areal/engine/r3/layout.py`        | Convert workflow routing tensors to Megatron native replay slabs.                               |
| `areal/engine/r3/discovery.py`     | Discover instance-local Megatron native `router_replay` objects.                                |
| `areal/engine/r3/orchestration.py` | Set/clear native replay actions and target indices.                                             |
| `areal/engine/r3/transport.py`     | Move R3 side-channel tensors through actor/engine boundaries.                                   |
| `areal/engine/r3/asserts.py`       | Raise `R3Error` with AReaL, Megatron-Core, and mbridge versions.                                |
| `areal/workflow/rlvr.py`           | Adds R3 tensors to RLVR trajectories when SGLang returns routing.                               |
| `areal/trainer/ppo/actor.py`       | Feeds R3 tensors into compute-logp and PPO update; logs mismatch metrics.                       |
| `areal/engine/megatron_engine.py`  | Initializes native replay, validates routing, drives per-micro-batch replay, and logs counters. |

If a micro-batch contains invalid routing, AReaL never passes all-zero fallback rows to
Megatron. In compute-logp (`forward_only=True`) it uses the live router. In training
updates it records live router choices for the forward pass and replays those recorded
choices for backward recompute, preserving forward/backward consistency.

## Usage

Enable R3 with the rollout-side switch:

```bash
python3 examples/math/gsm8k_rl.py --config <megatron_moe_grpo.yaml> \
  scheduler.type=local \
  rollout.backend=sglang:d1p1t4 \
  actor.backend='megatron:(attn:d1p1t4|ffn:d1p1t1e4)' \
  rollout.return_routed_experts=true \
  actor.megatron.enable_router_replay=true \
  actor.megatron.moe_router_fusion=false
```

Common YAML fragment:

```yaml
rollout:
  backend: "sglang:d1p1t4"
  return_routed_experts: true

actor:
  backend: "megatron:(attn:d1p1t4|ffn:d1p1t1e4)"
  megatron:
    enable_router_replay: true
    moe_router_fusion: false
```

`rollout.return_routed_experts=true` is the main user-facing switch. In the RL trainer
it also enables the SGLang server's internal routed-expert return flag and turns on
`actor.megatron.enable_router_replay`. Keeping the actor flag explicit in experiment
configs is recommended because saved configs then show the intended training path.

Only the actor replays rollout routing. Reference, teacher and critic scoring use their
own routers; their trajectory views omit the actor's routing metadata without copying
the shared token tensors. The actor retains its original routing for logprob
recomputation and PPO updates.

For concatenated multi-turn interactions, each continuation preserves the historical
parent routes except the parent's final sampled token. That token was not forwarded in
the parent, so its copied placeholder is replaced by the child's captured route.

Supported path:

- Megatron actor backend.
- SGLang rollout backend.
- MoE models with discoverable router top-k and either all-MoE layers or a dense prefix
  followed by a contiguous MoE suffix. Full-layer capture buffers may contain only
  zero-filled dense slots before the MoE suffix.
- Packed text RLVR and OpenAI proxy workflows.

Rejected path:

- vLLM rollout with `return_routed_experts=true`.
- Non-Megatron actor backends.
- Non-MoE actor models.
- Sparse/interleaved MoE layouts or dense layers after the first MoE layer. These need
  explicit global-layer metadata in rollout preprocessing and are rejected at R3
  configuration resolution rather than silently selecting the wrong captured layers.
- MTP layers, or `mcore-bridge` outside the experimental Qwen4Exp path below.
- Vision workflow, tree training, or padded-sequence Megatron training.
- Effective Megatron `moe_router_fusion=True`; native replay needs the unfused router
  path.

### Experimental Qwen Flash Next Path

The Qwen4Exp / Qwen3.8 Flash Next bridge path is restricted to an all-MoE decoder,
`WRAPPER_THD` packing and zigzag CP. Native `moe_enable_routing_replay` is passed to the
bridge before routers are constructed. MTP, tree training, padded/model-owned packing
and other bridge/VLM combinations remain rejected.

Use the pinned Flash Next Transformers, mcore-bridge and SGLang runtime from the model
recipe. OpenAI proxy exports must preserve processor-expanded prompt IDs, image tensors
and token masks; the image embedding replacement must not change the captured routing
token domain. Generic VisionRLVRWorkflow is not supported.

The validated SGLang `0.5.19.dev125+g119b5ffe4` rollout image still requires the QSA
compress-gather fix. Automatic model patching reads back the installed `qsa_indexer.py`
and requires its SHA256 to equal `sglang_qsa_compress_gather.PATCHED_SHA256` before the
selected SGLang/AWEX server entry point executes. The version string alone does not
prove the fix is active. Existing launch recipes may retain the three
`examples.swe.qwen38_flash_next.patch_sglang_*` commands; these compatibility entry
points delegate to the current guarded, idempotent patch modules.

Enable capture with `rollout.return_routed_experts=true`, replay with
`actor.megatron.enable_router_replay=true`, and disable `moe_router_fusion`. The
deployed SGLang fork's capture API must be checked separately from the legacy SGLang
compatibility patch.

This path requires a real-checkpoint GPU canary before production use: verify local
router discovery, exact forward/recompute target IDs, CP/SP token order, queue cleanup,
finite gradients and a successful optimizer update. Metadata checks and CPU tests alone
do not establish full-model support.

## Performance Validation

R3 does not require another actor forward when `reuse_train_logp` avoids logprob
recomputation. Capture, serialization, transport and replay still consume resources;
there is no validated zero-overhead or fixed slowdown percentage for this
implementation's long-context path.

The current wire representation is base64 int32. With 48 MoE layers and top-k 10, a 256
Ki-token trajectory carries approximately 480 MiB of raw routes, or 640 MiB as base64,
excluding framing. CPU rank-local slicing reduces replay H2D volume, but does not bound
the producer payload or RTensor request size. Binary compact routing, byte-budgeted
transport and asynchronous capture readback require separate validation against the
deployed SGLang version.

Before claiming a performance percentage, compare matched weights, inputs, lengths,
cache state and concurrency with capture off/on and replay off/on. Report warmup
separately, median/p95 timings, throughput and peak host/GPU memory, alongside replay
coverage and logprob mismatch. PP4/TP8/CP2 with full recomputation and 16K–256K contexts
has not been validated by the small-model smoke tests.

## Metrics

R3 adds two groups of metrics.

### Rollout/Training Logprob Mismatch

These metrics are emitted under `compute_logp/r3/*` during actor recompute-logp. They
compare SGLang rollout token logprobs with Megatron training recompute logprobs on valid
loss tokens.

These metrics are not emitted when actor recompute-logp is bypassed by
`reuse_train_logp`; the existing loss-stage logprob mismatch metrics remain available.

| Metric                                                          | Meaning                                                                                 |
| --------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `compute_logp/r3/enabled`                                       | `1.0` when the batch carried R3 side-channel tensors; `0.0` otherwise.                  |
| `compute_logp/r3/rollout_train_k3_kl/{avg,max,min}`             | Taylor KL estimate `exp(delta) - 1 - delta`, where `delta = train_logp - rollout_logp`. |
| `compute_logp/r3/rollout_train_logp_abs_diff/{avg,max,min}`     | Absolute token logprob difference.                                                      |
| `compute_logp/r3/rollout_train_logp_sq_diff/{avg,max,min}`      | Squared token logprob difference.                                                       |
| `compute_logp/r3/rollout_train_extreme_frac_tau2/{avg,max,min}` | Fraction of valid tokens with `abs(delta) > log(2)`.                                    |
| `compute_logp/r3/rollout_train_extreme_frac_tau5/{avg,max,min}` | Fraction of valid tokens with `abs(delta) > log(5)`.                                    |
| `compute_logp/r3/n_valid_tokens`                                | Denominator for token-level mismatch statistics.                                        |

### Replay Execution Counters

These metrics are emitted as `r3/*` from the Megatron engine. During PPO update they can
also appear with a trainer prefix such as `ppo_actor/update/r3/*`.

| Metric                           | Meaning                                                                                       |
| -------------------------------- | --------------------------------------------------------------------------------------------- |
| `r3/enabled`                     | `1.0` when Megatron native router replay was used for the call.                               |
| `r3/batches_with_side_channel`   | Batches that carried R3 tensors.                                                              |
| `r3/samples_total`               | Number of sequences in the side channel.                                                      |
| `r3/routing_valid_samples`       | Sequences whose rollout routing passed preprocessing validation.                              |
| `r3/routing_invalid_samples`     | Sequences marked invalid by preprocessing.                                                    |
| `r3/routing_valid_fraction`      | `routing_valid_samples / samples_total`.                                                      |
| `r3/logical_microbatches`        | Megatron logical micro-batches for this call.                                                 |
| `r3/router_stage_microbatches`   | Micro-batches on stages that contain local MoE routers.                                       |
| `r3/replayed_microbatches`       | Micro-batches where rollout routing was replayed.                                             |
| `r3/skipped_microbatches`        | Micro-batches where rollout routing was not safe to replay.                                   |
| `r3/invalid_samples`             | Invalid samples that caused replay skips.                                                     |
| `r3/mode_replay_microbatches`    | Micro-batches run in native replay mode.                                                      |
| `r3/mode_record_microbatches`    | Training micro-batches that used native `RECORD` because rollout routing was invalid.         |
| `r3/mode_live_microbatches`      | Forward-only micro-batches that fell back to the live router.                                 |
| `r3/mode_no_router_microbatches` | Micro-batches on model stages without local MoE routers.                                      |
| `r3/replay_fraction_active`      | `mode_replay_microbatches / router_stage_microbatches`; healthy R3 runs should be near `1.0`. |
| `r3/skip_fraction_active`        | `skipped_microbatches / router_stage_microbatches`; healthy R3 runs should be near `0.0`.     |
| `r3/tokens_real`                 | Real packed tokens seen by Megatron.                                                          |
| `r3/tokens_padded`               | Tokens after Megatron packing/padding.                                                        |
| `r3/tokens_padding`              | Padding tokens introduced by Megatron packing/alignment.                                      |
| `r3/replay_real_tokens`          | Real tokens in replayed micro-batches.                                                        |
| `r3/record_real_tokens`          | Real tokens in record-fallback micro-batches.                                                 |
| `r3/live_real_tokens`            | Real tokens in live-router fallback micro-batches.                                            |
| `r3/no_router_real_tokens`       | Real tokens on stages without local MoE routers.                                              |

Skip reasons are sparse counters emitted only when they happen, for example
`r3/skip_invalid_sample`, `r3/skip_zero_routing_for_real_token`, or
`r3/skip_missing_final_token_routing_without_replacement`.

## Sanity Checks

For a healthy R3-on run:

- `compute_logp/r3/enabled` is `1.0` on recompute-logp batches.
- `r3/replay_fraction_active` is close to `1.0`.
- `r3/routing_valid_fraction` is close to `1.0`.
- `r3/skipped_microbatches` and `r3/invalid_samples` stay at `0`, or are explained by
  rollout aborts.
- `rollout_train_k3_kl`, `rollout_train_logp_abs_diff`, and extreme-fraction metrics are
  materially lower than an otherwise identical R3-off run.

In the GSM8K Moonlight-16B-A3B bs32/group8 validation run, both R3-on and R3-off
completed `233/233` steps. Tail-20 mismatch metrics dropped with R3:

| Metric                                |    R3 off |        R3 on |
| ------------------------------------- | --------: | -----------: |
| `rollout_train_k3_kl/avg`             | `0.00958` |   `0.000373` |
| `rollout_train_logp_abs_diff/avg`     |  `0.0322` |    `0.00607` |
| `rollout_train_extreme_frac_tau2/avg` | `0.00908` |  `0.0000681` |
| `rollout_train_extreme_frac_tau5/avg` | `0.00133` | `0.00000397` |

## Debugging

`R3Error` includes package versions and shape/config context, for example:

```text
routed_experts has more token rows than the training sequence
(areal='...', megatron-core='...', mbridge='...', raw_shape=(...), seq_len=...)
```

Typical causes:

- SGLang and Megatron disagree on MoE layer count or router top-k.
- SGLang returns routing rows that no longer match the training sequence.
- A workflow emits `routed_experts` without the paired `r3_routing_valid` tensor.
- `moe_router_fusion=True` bypasses the native replay-capable router path.

Focused tests:

```bash
python -m pytest \
  tests/test_r3_preprocess.py \
  tests/test_r3_layout.py \
  tests/test_r3_orchestration.py \
  tests/test_r3_config_discovery.py \
  tests/test_r3_engine_validation.py \
  tests/test_remote_inf_engine_r3.py \
  tests/test_ppo_actor_r3.py
```
