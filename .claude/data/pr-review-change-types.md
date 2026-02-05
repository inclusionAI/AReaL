# PR Review: Change Type Detection Reference

This file contains the change type detection tables for PR review. Referenced by:
`.claude/commands/pr-review.md`

______________________________________________________________________

## CRITICAL Level (Must use Opus)

| Change Type            | File Path Pattern                                             | Code Pattern                                                |
| ---------------------- | ------------------------------------------------------------- | ----------------------------------------------------------- |
| **ARCHON_CORE**        | `areal/experimental/models/archon/`                           | -                                                           |
| **ARCHON_PARALLEL**    | `parallel_dims.py`                                            | `ArchonParallelDims`, `_build_mesh`, `DeviceMesh`           |
| **ARCHON_MOE**         | `archon/moe/`                                                 | `router`, `grouped_experts`, `TokenReorderer`, `grouped_mm` |
| **ARCHON_PARALLELIZE** | `qwen*/infra/parallelize.py`                                  | `apply_moe_ep_tp`, `apply_tp`, `apply_cp`                   |
| **ARCHON_ENGINE**      | `areal/experimental/engine/archon_engine.py`                  | `ArchonEngine`                                              |
| **FSDP_CORE**          | `areal/utils/fsdp/`, `areal/engine/fsdp_engine.py`            | `FSDP`, `FullyShardedDataParallel`, `fully_shard`           |
| **MEGATRON_CORE**      | `areal/engine/megatron_engine.py`, `areal/utils/megatron*.py` | `MegatronEngine`                                            |
| **DCP_CHECKPOINT**     | -                                                             | `DCP`, `DistributedCheckpoint`, `dcp.save`, `dcp.load`      |

## HIGH Level (Recommend Opus)

| Change Type           | File Path Pattern | Code Pattern                                                                     |
| --------------------- | ----------------- | -------------------------------------------------------------------------------- |
| **DISTRIBUTED_COMM**  | -                 | `all_reduce`, `all_gather`, `reduce_scatter`, `all_to_all`, `dist.`              |
| **DTENSOR**           | -                 | `DTensor`, `DeviceMesh`, `Shard(`, `Replicate(`, `Partial(`, `distribute_tensor` |
| **MOE_LAYER**         | `moe/`            | `expert`, `token_dispatch`, `grouped_mm`, `MoE`                                  |
| **EP_ETP**            | -                 | `ExpertParallel`, `TensorParallel`, `ExpertTensorParallel`, `ep_size`, `etp`     |
| **TENSOR_PARALLEL**   | -                 | `ColwiseParallel`, `RowwiseParallel`, `parallelize_module`                       |
| **SEQUENCE_PARALLEL** | -                 | `SequenceParallel`, `context_parallel`, `Ulysses`, `cp_size`                     |
| **ASYNC_CONCURRENT**  | -                 | `async def`, `await`, `asyncio`, `threading.Lock`, `aiofiles`                    |
| **TRAINER_CORE**      | `areal/trainer/`  | `PPOTrainer`, `SFTTrainer`, `trainer.train`                                      |

## MEDIUM Level (Use Sonnet)

| Change Type             | File Path Pattern                                                                  | Code Pattern                                                             |
| ----------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| **TENSOR_OPS**          | -                                                                                  | `.view(`, `.reshape(`, `dtype=`, `.detach()`, `no_grad`, `.contiguous()` |
| **NUMERICAL**           | -                                                                                  | `log(`, `softmax`, `cross_entropy`, `eps=`, `.clamp(`, `nan`, `inf`      |
| **WORKFLOW_ENGINE**     | `areal/workflow/`, `areal/engine/`                                                 | `arun_episode`, `agenerate`, `RolloutWorkflow`                           |
| **API_CONFIG**          | `areal/api/`                                                                       | `@dataclass`, `__post_init__`, `field(`                                  |
| **COMPILE**             | -                                                                                  | `torch.compile`, `_dynamo`, `mark_dynamic`, `fullgraph`                  |
| **ACTIVATION_CKPT**     | `activation_checkpoint.py`                                                         | `activation_checkpoint`, `checkpoint_wrapper`, `selective_checkpoint`    |
| **CHECKPOINT_RECOVERY** | `areal/utils/saver.py`, `areal/utils/recover.py`, `areal/utils/fsdp/checkpoint.py` | `state_dict`, `load_state_dict`, `checkpoint`                            |
| **REWARD**              | `areal/reward/`                                                                    | `reward_fn`, `AsyncRewardWrapper`, `MathVerifyWorker`                    |
| **DATASET**             | `areal/dataset/`                                                                   | `get_*_dataset`, `DataLoader`, `IterableDataset`                         |
| **LAUNCHER_SCHEDULER**  | `areal/launcher/`, `areal/scheduler/`                                              | `LaunchConfig`, `Scheduler`, `RayLauncher`, `SlurmLauncher`              |
| **ATTENTION**           | `attention.py`, `varlen_attention.py`                                              | `flash_attn`, `sdpa`, `varlen`, `causal_mask`                            |

## LOW Level (Use Haiku)

| Change Type     | File Path Pattern            | Code Pattern |
| --------------- | ---------------------------- | ------------ |
| **TESTS**       | `tests/`, `*_test.py`        | -            |
| **DOCS**        | `docs/`, `*.md`              | -            |
| **CONFIG_ONLY** | `*.yaml`, `*.json`, `*.toml` | -            |

______________________________________________________________________

## Framework-Specific Risk Identification

### Archon Risks (When ARCHON\_\* types detected)

- **Device mesh dimension mismatch**: mesh dimension names don't correspond to placement
- **EP constraint violation**: `ep_size` must divide `num_experts`, and
  `dp_shard * cp * (tp if etp==1 else 1) % ep == 0`
- **ETP configuration error**: `etp` must be 1 or equal to `tp`
- **Token alignment error**: `grouped_mm` requires token count aligned to 8/16/32
- **All-to-All split/combine mismatch**: dispatch and combine split configs inconsistent
- **DTensor/Local tensor conversion missing**: need `.to_local()` or
  `DTensor.from_local()`
- **torch.compile dynamic shape marking missing**: missing `mark_dynamic` calls
- **AC application order error**: must be after TP/CP, before FSDP
- **Ulysses SP configuration**: CP uses Ulysses implementation, not Ring Attention
- **dp_shard_mod_ep mesh usage**: MoE experts must use `dp_shard_mod_ep` mesh for FSDP

### FSDP Risks (When FSDP\_\* types detected)

- **Shard/reshard timing error**: premature or delayed sharding operations
- **EP mesh interaction issue**: should use `dp_shard_mod_ep` not `dp_shard` for MoE
- **Gradient divide factor calculation**: incorrect relationship with world size
- **State dict save/load inconsistency**: mixing sharded vs full modes
- **Optimizer state handling**: aggregation and distribution of sharded state
- **DCP compatibility**: ensure DCP save/load works with FSDP2

### Megatron Risks (When MEGATRON\_\* types detected)

- **Pipeline stage splitting error**: unbalanced layer distribution
- **Micro-batch scheduling issues**: pipeline bubble handling
- **Weight sharding and sync**: tied weights handling
- **AC interaction**: checkpointing under pipeline parallelism

### DCP/Checkpoint Risks (When DCP_CHECKPOINT or CHECKPOINT_RECOVERY detected)

- **Distributed checkpoint consistency**: all ranks must participate in save/load
- **State dict key mismatch**: keys must match between save and load
- **Optimizer state compatibility**: ensure optimizer state is correctly
  sharded/gathered
- **Version compatibility**: old checkpoints should load in new code
- **Storage backend compatibility**: ensure storage backend (filesystem, S3, etc.) is
  compatible

______________________________________________________________________

## Risk Linkage Rules

| Detected Change             | Auto-Linked Review                                     |
| --------------------------- | ------------------------------------------------------ |
| EP changes                  | FSDP interaction check, dp_shard_mod_ep mesh check     |
| ETP changes                 | TP + EP combination check, mesh dimension check        |
| Megatron changes            | Pipeline + AC check                                    |
| Distributed comm changes    | Process group + sync check                             |
| SEQUENCE_PARALLEL changes   | TP combination + Attention mask check, Ulysses check   |
| CHECKPOINT_RECOVERY changes | FSDP state dict check, DCP compatibility check         |
| DCP_CHECKPOINT changes      | FSDP2 integration check, distributed consistency check |
| COMPILE changes             | Performance regression + FSDP/TP interaction check     |
| REWARD changes              | Workflow interaction check, AsyncRewardWrapper check   |
| LAUNCHER_SCHEDULER changes  | Resource config + parallel strategy match check        |
| TRAINER_CORE changes        | Engine lifecycle + workflow integration check          |
| ARCHON_ENGINE changes       | DCP checkpoint + parallel dims check                   |

______________________________________________________________________

## Core Framework Paths (Must Use Opus)

**Archon Core**:

- `areal/experimental/models/archon/` (entire directory)
- `areal/experimental/engine/archon_engine.py`
- `areal/experimental/engine/archon_checkpoint.py`

**FSDP Core**:

- `areal/utils/fsdp/`
- `areal/engine/fsdp_engine.py`

**Megatron Core**:

- `areal/engine/megatron_engine.py`
- `areal/utils/megatron.py`
- `areal/utils/megatron_checkpointer.py`

**Trainer Core**:

- `areal/trainer/`

**Training Engine Core** (excludes FSDP/Megatron which have their own categories):

- `areal/engine/` (except `fsdp_engine.py`, `megatron_engine.py`)
