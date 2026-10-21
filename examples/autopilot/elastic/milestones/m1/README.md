# M1: Data-triggered training GPU lifecycle

Historical H20 validation: 2026-10-17, AReaL v2.1.0 runtime `11cd5a38`, before directory
reorganization. Original measurements and layout checks are in the
[evidence summary](validation.json).

The [2026-10-18 current-runtime rerun](rerun-20261018.json) passed two fresh SWE
episodes from version 0 to 2, recovery/publication, and independent GPU/sandbox cleanup.
Linux regression: 190 passed, one CUDA skip; GPU runtime-state checks: four passed. The
local full-tree CLI-doc hook required `torchdata`; generation passed in the pinned Linux
runtime with byte-identical output.

The execution loop and resource scheduling rules are separated under `elastic/`, with
mechanisms in `elastic/runtime/`. `revision/` is a placeholder only; its design is
deferred. PPO/LM validation groups remain in `fixtures/`.

## Scope

On one H20 host, Qwen3-0.6B uses one persistent inference GPU and a fixed four-GPU FSDP
training group. A ready SWE batch wakes training; an update commits a full-state
checkpoint and publishes serving weights; clean idle time releases the training GPUs.
CPU services, queue ownership and inference survive worker generations.

The gate admits one episode at a time and requires version staleness 0. The next rollout
can overlap idle waiting and release; rollout and training updates are serialized. See
the [code map and run command](../../README.md) and
[sandbox setup](../../swe/README.md).

## Acceptance

| Check                                      | Evidence                                                                                                                                                      |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Data-driven wake and clean idle release    | Two fresh episodes of one real SWE task; versions 1 → 2 → 3; two restores and two verified releases                                                           |
| No unnecessary allocation or save          | Empty queue keeps training off; clean idle performs no additional checkpoint                                                                                  |
| Inference and CPU frontend survive release | Stable training endpoint/process identities; next live episode overlaps release; post-release tokens use the published version                                |
| Real optimizer and serving updates         | Nonzero gradients and parameter changes; HF embedding/head equality; loaded SGLang embedding/down-projection equality and proxy version checked               |
| Durable consumption and recovery           | Outbox equals offline queue payload; CPU fault tests cover replay and lost acknowledgements; separate six-batch LM experiment checks restored state/RNG/order |
| Cleanup and regression                     | Training GPU reuse verified; final GPU/sandbox cleanup passed; CPU regression 167 passed, one CUDA skip; full pre-commit passed                               |

The [portable evidence summary](validation.json) includes the original runtime commit,
source hashes and hashes of the original local reports. Historical paths/hashes refer to
that commit, not the reorganized working tree. Raw reports, models and checkpoints
remain outside Git. The LM state-equivalence result is separate from the SWE lifecycle
result; PPO continuous-versus-restored equivalence has not been established.

## Observed costs

| Operation                                         |   Seconds |
| ------------------------------------------------- | --------: |
| First wake, including CPU frontend initialization |      78.3 |
| Subsequent worker recreation and state restore    |      46.1 |
| Checkpoint, including serving weights             | 11.8–13.5 |
| Weight publication, including initial reload      |   2.5–3.2 |
| Release and independent GPU allocation check      | 11.5–11.7 |

The 5-second idle timeout is a validation setting; release checks follow it. Each
checkpoint is about 9.30 GB. No resource savings or optimal timeout is claimed.

## Boundaries and next work

- Training uses critic-free clipped policy gradient with fixed baseline 0.5. Both task
  rewards remain 0; every rank computes the same small batch. Default SWE/GRPO,
  distinct-data DP throughput and SWE quality improvements are unverified.
- Single producer, one lifecycle thread, local filesystem queue and fixed world size.
  Multi-producer/asynchronous backlog handling and arbitrary external traffic are
  unsupported. Training GPUs must be reserved for this job; no cross-job resource lease
  is provided.
- Checkpoint/update uncertainty stops the loop; publication failures retry the committed
  artifact. Coordinator SIGKILL recovery and the export-to-outbox loss window remain
  open.
- Local GPU release is verified; Slurm allocation return/reacquisition is pending.
- Next: discuss and implement a basic lifecycle/optimizer/simulator policy supporting
  1/2/4 training GPUs, then discuss the revision loop. Resident-training comparisons,
  native SWE training, asynchronous rollout and Slurm remain follow-up work.
