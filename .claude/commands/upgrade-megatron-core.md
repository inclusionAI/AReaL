---
name: upgrade-megatron-core
description: Upgrade Megatron-Core version in AReaL. Audits all megatron.core and mbridge API usage, cross-references upstream source, and updates call sites.
argument-hint: <version>
---

## Usage

```
/upgrade-megatron-core $ARGUMENTS
```

**Arguments (`$VERSION`):** Target Megatron-Core version tag or commit hash, e.g.
`v0.12.0`, `core_r0.12.0`, or a commit SHA. If not given, get the required version from
AReaL's "pyproject.toml" and check whether the current code is fully compatible with the
specified version.

## Prerequisites — Source Code for Cross-Referencing

This command requires upstream source repos to cross-reference API signatures.

### Megatron-LM

```bash
MCORE_DIR="${REPO_ROOT}/Megatron-LM"
# Validate VERSION to prevent command injection
if [[ ! "$VERSION" =~ ^[a-zA-Z0-9._/-]+$ ]]; then
  echo "Error: Invalid version format: $VERSION"
  exit 1
fi
if [ ! -d "$MCORE_DIR" ]; then
  git clone --depth 1 --branch "${VERSION}" https://github.com/NVIDIA/Megatron-LM.git "$MCORE_DIR"
else
  cd "$MCORE_DIR" && git fetch origin && git checkout "${VERSION}" && cd -
fi
```

If cloning or checkout fails, report to the user immediately.

The relevant upstream source paths are:

- `Megatron-LM/megatron/core/parallel_state.py`
- `Megatron-LM/megatron/core/distributed/`
- `Megatron-LM/megatron/core/optimizer/`
- `Megatron-LM/megatron/core/optimizer_param_scheduler.py`
- `Megatron-LM/megatron/core/pipeline_parallel/`
- `Megatron-LM/megatron/core/transformer/`
- `Megatron-LM/megatron/core/models/gpt/`
- `Megatron-LM/megatron/core/fp8_utils.py`
- `Megatron-LM/megatron/core/dist_checkpointing/`
- `Megatron-LM/megatron/core/packed_seq_params.py`
- `Megatron-LM/megatron/core/utils.py`

### mbridge

mbridge wraps megatron.core for HF↔MCore weight conversion. AReaL depends on its
internal APIs for weight loading/saving, so mbridge must also be audited when upgrading
megatron-core.

```bash
MBRIDGE_DIR="${REPO_ROOT}/mbridge-src"
# Determine the compatible mbridge version from pyproject.toml
MBRIDGE_VER=$(grep 'mbridge' "${REPO_ROOT}/pyproject.toml" | grep -oP '\d+\.\d+\.\d+')
if [ ! -d "$MBRIDGE_DIR" ]; then
  git clone --branch "v${MBRIDGE_VER}" https://github.com/ISEEKYAN/mbridge.git "$MBRIDGE_DIR"
else
  cd "$MBRIDGE_DIR" && git fetch origin && git checkout "v${MBRIDGE_VER}" && cd -
fi
```

The relevant mbridge source paths are:

- `mbridge-src/mbridge/__init__.py` — top-level exports (`AutoBridge`)
- `mbridge-src/mbridge/core/bridge.py` — `Bridge` base class: `get_model()`,
  `load_weights()`, `save_weights()`, `export_weights()`, `set_extra_args()`, all
  `_weight_*` private methods
- `mbridge-src/mbridge/core/auto_bridge.py` — `AutoBridge.from_pretrained()`,
  `from_config()`
- `mbridge-src/mbridge/core/llm_bridge.py` — `LLMBridge`: `_build_base_config()`,
  `_get_gptmodel_args()`, `_get_transformer_layer_spec()`, `_model_provider()`
- `mbridge-src/mbridge/core/vlm_bridge.py` — `VLMBridge` (not directly used but may
  affect inheritance)
- `mbridge-src/mbridge/core/util.py` — `get_model()`, `unwrap_model()`,
  `broadcast_from_megatron_pp()`, `preprocess_packed_seqs()`,
  `postprocess_packed_seqs()`
- `mbridge-src/mbridge/core/parallel_states.py` — `ParallelStates` dataclass (wraps
  `mpu.*` getters)
- `mbridge-src/mbridge/utils/post_creation_callbacks.py` — `make_value_model()`,
  `freeze_moe_router()`

______________________________________________________________________

## Affected Files

### Primary (engine layer — most likely to break)

| File                                                     | Imports                                                                                                                                                                                                           |
| -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `areal/engine/megatron_engine.py`                        | `parallel_state`, `tensor_parallel`, `DDP`, `finalize_model_grads`, `OptimizerConfig`, `get_megatron_optimizer`, `OptimizerParamScheduler`, `get_forward_backward_func`, `TransformerConfig`, `get_model_config`  |
| `areal/engine/megatron_utils/megatron.py`                | `parallel_state`, `is_float8tensor`, `TransformerConfig`, `get_transformer_layer_offset`                                                                                                                          |
| `areal/engine/megatron_utils/checkpointer.py`            | `dist_checkpointing`, `mpu`, `tensor_parallel`, `ShardedObject`, `get_default_load_sharded_strategy`, `get_default_save_sharded_strategy`, `FullyParallelLoadStrategyWrapper`, `FullyParallelSaveStrategyWrapper` |
| `areal/engine/megatron_utils/packed_context_parallel.py` | `parallel_state`, `PackedSeqParams`                                                                                                                                                                               |
| `areal/engine/megatron_utils/pipeline_parallel.py`       | `TransformerConfig`, `PipelineParallelLayerLayout`                                                                                                                                                                |
| `areal/engine/megatron_utils/fp8/tensor_helper.py`       | `is_float8tensor`                                                                                                                                                                                                 |

### Secondary (model layer)

| File                                        | Imports                                                                                                                                                 |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `areal/models/mcore/registry.py`            | `tensor_parallel`, `DDP`, `MCoreDDPConfig`, `GPTModel`, `TransformerConfig`                                                                             |
| `areal/models/mcore/hf_load.py`             | `parallel_state`, `is_float8tensor`                                                                                                                     |
| `areal/models/mcore/hf_save.py`             | `parallel_state`, `is_float8tensor`                                                                                                                     |
| `areal/models/mcore/common.py`              | `TransformerConfig`                                                                                                                                     |
| `areal/models/mcore/qwen3.py`               | `get_gpt_decoder_block_spec`, `TransformerConfig`                                                                                                       |
| `areal/models/tree_attn/module_megatron.py` | `PackedSeqParams`, `TransformerConfig`, `SelfAttention`, `AttnMaskType`, `TransformerBlockSubmodules`, `TransformerLayer`, `TransformerLayerSubmodules` |

### Tertiary (infra + tests — lower risk)

| File                                                | Imports                                                  |
| --------------------------------------------------- | -------------------------------------------------------- |
| `areal/infra/workflow_executor.py`                  | `parallel_state` (conditional import inside method)      |
| `tests/test_estimate_num_params.py`                 | `parallel_state`, `tensor_parallel`                      |
| `tests/fp8/engine_utils.py`                         | `parallel_state`                                         |
| `tests/fp8/model_hooks.py`                          | `parallel_state`                                         |
| `tests/fp8/test_fp8_rmsnorm.py`                     | `get_fp8_context`, `is_float8tensor`, `get_model_config` |
| `tests/torchrun/run_megatron_engine_distributed.py` | `parallel_state`                                         |

### mbridge files (coupled with megatron.core)

| File                                        | mbridge Imports                                         |
| ------------------------------------------- | ------------------------------------------------------- |
| `areal/engine/megatron_engine.py`           | `mbridge.AutoBridge`                                    |
| `areal/models/mcore/registry.py`            | `mbridge.core.bridge.Bridge`                            |
| `areal/models/mcore/hf_load.py`             | `mbridge.core.bridge.Bridge`                            |
| `areal/models/mcore/hf_save.py`             | `mbridge.core.Bridge`, `mbridge.core.util.unwrap_model` |
| `areal/models/tree_attn/module_megatron.py` | `mbridge.core.LLMBridge`                                |
| `tests/test_estimate_num_params.py`         | `mbridge.AutoBridge`                                    |

______________________________________________________________________

## API Usage Catalog

For each function/class below, verify the call signature against the upstream source at
the target version. Focus on: **missing new required parameters**, **removed old
parameters**, **renamed parameters**, **changed return types**, and **changed method
signatures on returned objects**.

### 1. `megatron.core.parallel_state` (aliased as `mpu`)

**Source:** `Megatron-LM/megatron/core/parallel_state.py`

#### `mpu.initialize_model_parallel(...)`

Called in `megatron_engine.py:186`:

```python
mpu.initialize_model_parallel(
    tensor_model_parallel_size=...,
    pipeline_model_parallel_size=...,
    virtual_pipeline_model_parallel_size=...,
    use_sharp=False,
    order="tp-cp-ep-dp-pp",
    context_parallel_size=...,
    expert_model_parallel_size=...,
    expert_tensor_parallel_size=...,
    distributed_timeout_minutes=...,
)
```

**Check:** New required params? Renamed params? New parallelism dimensions? Removed
`use_sharp`? Changed `order` format?

#### `mpu.destroy_model_parallel()`

Called in `megatron_engine.py:426` and `tests/test_estimate_num_params.py:60`.
Straightforward — check for new required params.

#### Rank/world-size getters

All of the following are called without arguments unless noted:

- `mpu.get_data_parallel_rank()` /
  `mpu.get_data_parallel_rank(with_context_parallel=True)`
- `mpu.get_data_parallel_world_size()`
- `mpu.get_data_parallel_group()` /
  `mpu.get_data_parallel_group(with_context_parallel=True)`
- `mpu.get_tensor_model_parallel_rank()`
- `mpu.get_tensor_model_parallel_world_size()`
- `mpu.get_tensor_model_parallel_group()`
- `mpu.get_pipeline_model_parallel_rank()`
- `mpu.get_pipeline_model_parallel_world_size()`
- `mpu.get_pipeline_model_parallel_group()`
- `mpu.get_pipeline_model_parallel_last_rank()`
- `mpu.get_context_parallel_world_size()`
- `mpu.get_context_parallel_rank()`
- `mpu.get_context_parallel_group()`
- `mpu.get_expert_model_parallel_group()`
- `mpu.get_expert_model_parallel_world_size()`
- `mpu.get_expert_model_parallel_rank()`
- `mpu.get_expert_tensor_parallel_world_size()`
- `mpu.get_expert_tensor_parallel_group()`
- `mpu.get_expert_tensor_parallel_rank()`
- `mpu.get_expert_data_parallel_group()`
- `mpu.get_expert_data_parallel_rank()`
- `mpu.get_expert_tensor_model_pipeline_parallel_group()`
- `mpu.get_tensor_and_data_parallel_group(with_context_parallel=True)`
- `mpu.get_virtual_pipeline_model_parallel_rank()`
- `mpu.get_virtual_pipeline_model_parallel_world_size()`
- `mpu.set_virtual_pipeline_model_parallel_rank(vpp_rank)`
- `mpu.is_initialized()`

**Check:** Any renamed? Any removed? Any new required keyword-only args? Return type
changes?

#### `mpu.is_pipeline_last_stage(...)`

Called in two forms:

```python
mpu.is_pipeline_last_stage()
mpu.is_pipeline_last_stage(ignore_virtual=False, vp_stage=model_vp_stage)
```

**Check:** `ignore_virtual` / `vp_stage` params still exist?

#### `mpu.RankGenerator(...)`

Called in `megatron_engine.py:912`:

```python
mpu.RankGenerator(tp=..., ep=1, dp=..., pp=..., cp=..., order="tp-cp-ep-dp-pp", rank_offset=0)
```

**Check:** Constructor signature. New params?

#### `mpu.create_group(ranks, timeout=, pg_options=, group_desc=)`

Called in `megatron_engine.py:924`.

**Check:** Signature and kwargs.

#### `mpu.get_nccl_options(name, nccl_comm_cfgs)`

Called in `megatron_engine.py:927`:

```python
mpu.get_nccl_options("tp-cp-pp", {})
```

**Check:** Signature change.

______________________________________________________________________

### 2. `megatron.core.tensor_parallel`

**Source:** `Megatron-LM/megatron/core/tensor_parallel/`

#### `tensor_parallel.model_parallel_cuda_manual_seed(seed)`

Called in `megatron_engine.py:200`, `tests/test_estimate_num_params.py:34`.

**Check:** Signature.

#### `tensor_parallel.get_cuda_rng_tracker()`

Called in `checkpointer.py:172,313`. Returns object with `.get_states()` and
`.set_states(states)` methods.

**Check:** Return type still has `get_states()`/`set_states()` methods?

#### `tensor_parallel.gather_from_sequence_parallel_region(logits, tensor_parallel_output_grad=False)`

Called in `registry.py:49-51`.

**Check:** `tensor_parallel_output_grad` kwarg still exists?

______________________________________________________________________

### 3. `megatron.core.distributed`

**Source:** `Megatron-LM/megatron/core/distributed/`

#### `DistributedDataParallel` (DDP)

Used in `megatron_engine.py` and `registry.py`.

Constructor called in `registry.py:199`:

```python
DDP(config=tf_config, ddp_config=ddp_config, module=model, disable_bucketing=False)
```

Methods/attributes used:

- `model_chunk.no_sync` (property/context manager)
- `model_chunk.start_param_sync` (method)
- `.zero_grad_buffer()` (called in `megatron_engine.py:543`)
- `.module` attribute
- `.vp_stage` attribute (set manually)

**Check:** Constructor params. `.zero_grad_buffer()` still exists? `no_sync` /
`start_param_sync` interface?

#### `DistributedDataParallelConfig` (as `MCoreDDPConfig`)

Used in `registry.py:198`:

```python
MCoreDDPConfig(**dataclasses.asdict(mcore_config.ddp))
```

**Check:** Dataclass fields. Used fields: `use_distributed_optimizer`,
`overlap_grad_reduce`, `overlap_param_gather`, `align_param_gather`.

#### `finalize_model_grads`

Used in `megatron_engine.py:357`:

```python
model_config.finalize_model_grads_func = finalize_model_grads
```

**Check:** Signature of `finalize_model_grads`. Is it still assigned as a function
reference to `model_config.finalize_model_grads_func`?

______________________________________________________________________

### 4. `megatron.core.optimizer`

**Source:** `Megatron-LM/megatron/core/optimizer/`

#### `OptimizerConfig` (as `MCoreOptimizerConfig`)

Constructed in `megatron_engine.py:948`:

```python
MCoreOptimizerConfig(
    optimizer=..., lr=..., min_lr=..., weight_decay=...,
    bf16=..., fp16=...,
    adam_beta1=..., adam_beta2=..., adam_eps=...,
    use_distributed_optimizer=..., params_dtype=...,
    clip_grad=..., fp8_recipe=...,
)
```

Plus post-construction field assignments (lines 963-978):

- `overlap_param_gather_with_optimizer_step`
- `use_precision_aware_optimizer`
- `main_grads_dtype`
- `main_params_dtype`
- `exp_avg_dtype`
- `exp_avg_sq_dtype`

**Check:** All fields still exist? New required fields? Renamed fields? `fp8_recipe`
type changed?

#### `get_megatron_optimizer(config, model_chunks)`

Called in `megatron_engine.py:980`:

```python
self.optimizer = get_megatron_optimizer(mcore_opt_config, self.model)
```

**Check:** Signature change — does it still accept `(config, model_chunks)`? Any new
required params (e.g., `model_parallel_config`)? Return type interface: `.step()` should
return `(update_successful, grad_norm, num_zeros)`, `.param_groups`, `.zero_grad()`,
`.get_loss_scale()`, `.sharded_state_dict(state_dict)`, `.load_state_dict(state_dict)`.

______________________________________________________________________

### 5. `megatron.core.optimizer_param_scheduler`

**Source:** `Megatron-LM/megatron/core/optimizer_param_scheduler.py`

#### `OptimizerParamScheduler`

Constructed in `megatron_engine.py:987`:

```python
OptimizerParamScheduler(
    optimizer, init_lr=..., max_lr=..., min_lr=...,
    lr_warmup_steps=..., lr_decay_steps=..., lr_decay_style=...,
    start_wd=..., end_wd=..., wd_incr_steps=..., wd_incr_style="constant",
)
```

Methods used: `.step(1)`, `.state_dict()`, `.load_state_dict(state_dict)`.

**Check:** Constructor params — any renamed or removed? New required params? `.step()`
accepts integer increment?

______________________________________________________________________

### 6. `megatron.core.pipeline_parallel`

**Source:** `Megatron-LM/megatron/core/pipeline_parallel/`

#### `get_forward_backward_func()`

Called in `megatron_engine.py:621`. Returns a callable invoked as:

```python
forward_backward_func(
    forward_step_func=forward_step,
    data_iterator=data_iterator,
    model=...,
    num_microbatches=...,
    seq_length=...,
    micro_batch_size=...,
    forward_only=...,
)
```

**Check:** Return type callable signature. `forward_step_func` expected signature:
`(batch_iter, model) -> (output, loss_func)`. Any new required params like
`collect_non_loss_data`, `first_val_step`, `config`?

______________________________________________________________________

### 7. `megatron.core.transformer`

**Source:** `Megatron-LM/megatron/core/transformer/`

#### `TransformerConfig`

Used everywhere as configuration dataclass. Created via `bridge.config` or explicitly in
`common.py:check_and_construct_configs`.

Fields accessed in AReaL code:

- `hidden_size`, `num_attention_heads`, `num_query_groups`, `kv_channels`
- `ffn_hidden_size`, `num_layers`
- `num_moe_experts`, `moe_ffn_hidden_size`, `moe_layer_freq`
- `moe_shared_expert_intermediate_size`, `moe_router_enable_expert_bias`
- `expert_model_parallel_size`
- `sequence_parallel`, `context_parallel_size`
- `params_dtype`, `pipeline_dtype`, `bf16`, `fp16`
- `fp8`, `fp8_param`, `fp8_recipe`, and other `fp8_*` fields
- `gated_linear_unit`, `add_bias_linear`
- `deterministic_mode`, `cross_entropy_loss_fusion`, `bias_dropout_fusion`
- `no_sync_func`, `param_sync_func`, `finalize_model_grads_func`
- `variable_seq_lengths`, `masked_softmax_fusion`
- `pipeline_model_parallel_layout`
- `num_layers_in_first_pipeline_stage`, `num_layers_in_last_pipeline_stage`
- `account_for_embedding_in_pipeline_split`, `account_for_loss_in_pipeline_split`

**Check:** `check_and_construct_configs()` in `common.py` already handles removed fields
gracefully. But verify new required fields that may not have defaults.

#### `TransformerBlockSubmodules`, `TransformerLayer`, `TransformerLayerSubmodules`

Used in `module_megatron.py` for tree attention patching. Accessed via:

```python
spec.layer_specs  # list of layer specs
layer_spec.module  # should be TransformerLayer
layer_spec.submodules  # TransformerLayerSubmodules
submodules.self_attention  # attention spec
self_attn_spec.module  # should be SelfAttention
self_attn_spec.params["attn_mask_type"] = AttnMaskType.arbitrary
self_attn_spec.submodules.core_attention = PytorchFlexAttention
```

**Check:** `.layer_specs`, `.submodules`, `.self_attention`, `.params`,
`.submodules.core_attention` still exist on these objects?

#### `SelfAttention`

Used as a class reference check in `module_megatron.py:203`. Not instantiated directly.

**Check:** Still exists at `megatron.core.transformer.attention.SelfAttention`?

#### `AttnMaskType`

Used: `AttnMaskType.arbitrary` in `module_megatron.py:206`.

**Check:** `.arbitrary` enum value still exists?

#### `get_transformer_layer_offset(config, vp_stage=)`

Called in `megatron.py:612`:

```python
layer_offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
```

**Check:** Signature.

#### `PipelineParallelLayerLayout`

Constructed in `pipeline_parallel.py:62`:

```python
PipelineParallelLayerLayout(layout=layout, pipeline_model_parallel_size=pp_size)
```

**Check:** Constructor params.

______________________________________________________________________

### 8. `megatron.core.models.gpt`

**Source:** `Megatron-LM/megatron/core/models/gpt/`

#### `GPTModel`

Constructed in `registry.py:179`:

```python
GPTModel(
    config=tf_config, transformer_layer_spec=..., vocab_size=...,
    max_sequence_length=..., pre_process=True, post_process=True,
    share_embeddings_and_output_weights=False,
    position_embedding_type="rope", rotary_base=...,
)
```

Attributes/methods used: `.output_layer`, `.vocab_size`, `.sharded_state_dict()`,
`.config`, `.module`, `.named_parameters()`, `.load_state_dict()`, `.state_dict()`.

**Check:** Constructor signature. `position_embedding_type` values. New required params?

#### `get_gpt_decoder_block_spec(config, use_transformer_engine=True)`

Called in `qwen3.py:32`:

```python
get_gpt_decoder_block_spec(tfconfig, use_transformer_engine=use_te)
```

**Check:** Signature. Was it renamed? Does it accept the same args?

______________________________________________________________________

### 9. `megatron.core.fp8_utils`

**Source:** `Megatron-LM/megatron/core/fp8_utils.py`

#### `is_float8tensor(param)`

Called in `megatron.py:83`, `tensor_helper.py:4`, `hf_load.py:12`, `hf_save.py:13`,
`test_fp8_rmsnorm.py:15`.

**Check:** Still exists? Signature unchanged?

#### `get_fp8_context()`

Called in `test_fp8_rmsnorm.py:15`.

**Check:** Signature.

______________________________________________________________________

### 10. `megatron.core.dist_checkpointing`

**Source:** `Megatron-LM/megatron/core/dist_checkpointing/`

#### `dist_checkpointing.save(...)`

Called in `checkpointer.py:60`:

```python
dist_checkpointing.save(
    sharded_state_dict, ckpt_path,
    sharded_strategy=save_strategy,
    async_sharded_save=async_save,
    validate_access_integrity=validate_sharding_integrity,
)
```

**Check:** `async_sharded_save` renamed? `validate_access_integrity` still exists?

#### `dist_checkpointing.load(...)`

Called in `checkpointer.py:79`:

```python
dist_checkpointing.load(sharded_state_dict, ckpt_dir, sharded_strategy=load_strategy)
```

**Check:** Signature.

#### `ShardedObject(key, data, global_shape, global_offset, replica_id=)`

Called in `checkpointer.py:198`.

**Check:** Constructor params.

#### Serialization strategies

```python
get_default_load_sharded_strategy(ckpt_dir)
get_default_save_sharded_strategy("torch_dist")
FullyParallelLoadStrategyWrapper(load_strategy, group)
FullyParallelSaveStrategyWrapper(save_strategy, group)
```

**Check:** All four still exist? Signatures unchanged?

______________________________________________________________________

### 11. `megatron.core.packed_seq_params`

**Source:** `Megatron-LM/megatron/core/packed_seq_params.py`

#### `PackedSeqParams`

Constructed in `packed_context_parallel.py:34`:

```python
PackedSeqParams(
    qkv_format="thd",
    cu_seqlens_q=cu_seqlens, max_seqlen_q=max_seqlen,
    cu_seqlens_kv=cu_seqlens, max_seqlen_kv=max_seqlen,
    cu_seqlens_q_padded=cu_seqlens, cu_seqlens_kv_padded=cu_seqlens,
)
```

**Check:** Constructor fields.

______________________________________________________________________

### 12. `megatron.core.utils`

**Source:** `Megatron-LM/megatron/core/utils.py`

#### `get_model_config(model)`

Called in `megatron_engine.py:326`, `test_fp8_rmsnorm.py:16`.

**Check:** Signature. Return type. What fields are expected on the returned config
object (`.no_sync_func`, `.param_sync_func`, `.finalize_model_grads_func`,
`.deterministic_mode`, `.cross_entropy_loss_fusion`, `.bias_dropout_fusion`)?

______________________________________________________________________

## Upgrade Workflow

### Step 0: Prepare Megatron-LM source

Clone or checkout the target version as described in Prerequisites above.

### Step 1: Audit `megatron.core` API signatures

For EACH entry in the API Usage Catalog above:

1. Open the upstream source file at the target version.
1. Compare the function/class signature against the current AReaL invocation.
1. Flag any of:
   - **Removed parameters** still passed by AReaL → must remove from call site
   - **Renamed parameters** → must rename in call site
   - **New required parameters** (no default) → must add to call site
   - **New optional parameters** with useful defaults → document but skip
   - **Changed return types** → must update consumers
   - **Removed functions/classes** → must find replacement
   - **Changed method signatures** on returned objects → must update call sites
1. Record findings per-file.

### Step 2: Audit `mbridge` compatibility

mbridge wraps megatron.core and may also need updates. Cross-reference the cloned
`mbridge-src/` repo to verify each API AReaL depends on.

#### 2a. Public API (used directly by AReaL)

| AReaL Call Site                     | mbridge API                                                                           | Source File to Check                                                                                      |
| ----------------------------------- | ------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `megatron_engine.py:242`            | `AutoBridge.from_pretrained(path)`                                                    | `mbridge-src/mbridge/core/auto_bridge.py` — `from_pretrained()` resolves model type via `_MODEL_REGISTRY` |
| `registry.py:139`                   | `bridge.get_model(wrap_with_ddp=..., ddp_config=..., use_torch_fsdp2=..., ...)`       | `mbridge-src/mbridge/core/bridge.py` — `get_model()` passes kwargs to `get_model()` util                  |
| `megatron_engine.py`                | `bridge.load_weights(model, path)`                                                    | `mbridge-src/mbridge/core/bridge.py` — `load_weights()`                                                   |
| `megatron_engine.py`                | `bridge.save_weights(models, path, memory_efficient=..., distributed_filesystem=...)` | `mbridge-src/mbridge/core/bridge.py` — `save_weights()` and `_save_weights_fast()`                        |
| `megatron_engine.py`                | `bridge.export_weights(models)`                                                       | `mbridge-src/mbridge/core/bridge.py` — `export_weights()` generator                                       |
| `megatron_engine.py`                | `bridge.set_extra_args(**kwargs)`                                                     | `mbridge-src/mbridge/core/bridge.py` — rebuilds `self.config`                                             |
| `registry.py`, `megatron_engine.py` | `bridge.config` (returns `TransformerConfig`)                                         | `mbridge-src/mbridge/core/llm_bridge.py` — `_build_base_config()` constructs the config                   |
| `registry.py`, `hf_save.py`         | `bridge.hf_config`                                                                    | Stored on `Bridge.__init__()` from HF `AutoConfig`                                                        |

**Check:** Do `get_model()` kwargs still match `bridge.py:get_model()` signature? Does
`_build_base_config()` in `llm_bridge.py` pass any new required fields to
`TransformerConfig`? Does `set_extra_args()` still call `_build_config()`?

#### 2b. Private/internal API (used by AReaL's custom weight loaders)

| AReaL Call Site  | mbridge Private API                                                           | Source File to Check                                                                            |
| ---------------- | ----------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `hf_load.py:396` | `bridge._weight_name_mapping_mcore_local_to_global(model)`                    | `mbridge-src/mbridge/core/bridge.py` — maps VPP+EP local names to global                        |
| `hf_load.py:399` | `bridge._weight_name_mapping_mcore_to_hf(global_name)`                        | `mbridge-src/mbridge/core/bridge.py` — dispatches to `_weight_name_mapping_attention/mlp/other` |
| `hf_save.py:376` | `bridge._weight_to_hf_format(global_name, infer_params)`                      | `mbridge-src/mbridge/core/bridge.py` — splits QKV/gate-up, returns `(names, tensors)`           |
| `hf_save.py:368` | `bridge._weight_merge_across_tp(name, params, param)`                         | `mbridge-src/mbridge/core/bridge.py` — merges TP-split tensors                                  |
| `hf_load.py:365` | `bridge._get_actual_hf_path(weights_path)`                                    | `mbridge-src/mbridge/core/bridge.py` or subclass — resolves HF cache paths                      |
| `hf_save.py:197` | `bridge._weight_name_mapping_mcore_local_to_global(model, consider_ep=False)` | Same as above but with `consider_ep` kwarg                                                      |
| `hf_save.py:452` | `bridge.config.num_moe_experts`                                               | Field on `TransformerConfig` built by `_build_base_config()`                                    |
| `hf_save.py:536` | `bridge.hf_config.save_pretrained(weights_path)`                              | Standard HF `PretrainedConfig` method                                                           |
| `hf_save.py:191` | `unwrap_model(model)` from `mbridge.core.util`                                | `mbridge-src/mbridge/core/util.py` — unwraps DDP/Float16Module/FSDP wrappers                    |

**Check:** Do any of these private methods have changed signatures? Has the weight name
mapping logic changed (e.g., new `_DIRECT_MAPPING`, `_ATTENTION_MAPPING`, or
`_MLP_MAPPING` entries)? Does `unwrap_model()` still handle the same wrapper classes?

#### 2c. `LLMBridge._get_transformer_layer_spec()` (monkey-patched by tree attention)

| AReaL Call Site              | mbridge API                                             | Source File to Check                                                            |
| ---------------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `module_megatron.py:193,211` | `LLMBridge._get_transformer_layer_spec(self, vp_stage)` | `mbridge-src/mbridge/core/llm_bridge.py` — calls `get_gpt_decoder_block_spec()` |

AReaL monkey-patches this method to inject `PytorchFlexAttention` as `core_attention`.
The patch accesses `spec.layer_specs[i].submodules.self_attention.params` and
`spec.layer_specs[i].submodules.self_attention.submodules.core_attention`.

**Check:** Does `_get_transformer_layer_spec()` still return a
`TransformerBlockSubmodules` with `.layer_specs` list? Does each layer spec still have
`.submodules.self_attention.params` dict and `.submodules.core_attention` attribute? Has
the `vp_stage` parameter been added/removed?

#### 2d. mbridge version compatibility

If mbridge also needs a version bump to work with the new megatron-core, note the
compatible mbridge version. Check `mbridge-src/pyproject.toml` for its megatron-core
version pin.

### Step 3: Update `pyproject.toml`

Update the `megatron-core` (and optionally `mbridge`) version pin in `pyproject.toml`.
Run `uv lock` to verify dependency resolution.

### Step 4: Apply code changes

For each flagged incompatibility from Steps 1-2:

1. Update the call site in the affected file.
1. Preserve existing behavior — do NOT refactor beyond what's required.
1. If a function was removed, check the upstream migration guide or changelog.
1. If a mbridge API changed (Step 2b/2c), update AReaL's usage to match the new mbridge
   signatures. Common cases:
   - `bridge.get_model()` gained/lost kwargs → update `registry.py:139`
   - `bridge._weight_*` private methods renamed or re-signed → update `hf_load.py` and
     `hf_save.py` callers
   - `LLMBridge._get_transformer_layer_spec()` return structure changed → update the
     monkey-patch in `module_megatron.py`
   - `unwrap_model()` wrapper class list changed → verify unwrapping still works

Priority order for applying changes:

1. `areal/engine/megatron_engine.py` (highest risk, most API surface)
1. `areal/engine/megatron_utils/megatron.py`
1. `areal/engine/megatron_utils/checkpointer.py`
1. `areal/engine/megatron_utils/packed_context_parallel.py`
1. `areal/engine/megatron_utils/pipeline_parallel.py`
1. `areal/engine/megatron_utils/fp8/tensor_helper.py`
1. `areal/models/mcore/registry.py`
1. `areal/models/mcore/common.py`
1. `areal/models/mcore/qwen3.py`
1. `areal/models/mcore/hf_load.py` (mbridge private API consumer)
1. `areal/models/mcore/hf_save.py` (mbridge private API consumer)
1. `areal/models/tree_attn/module_megatron.py` (mbridge monkey-patch)
1. Test files

### Step 5: Verify `TransformerConfig` field compatibility

`areal/models/mcore/common.py` uses `check_and_construct_configs()` which automatically
removes unsupported fields. However:

1. Check that no **new required fields** (without defaults) were added to
   `TransformerConfig`.
1. Verify `hf_to_mcore_base_args()` in `common.py` — the hardcoded field names
   (`num_layers`, `hidden_size`, etc.) still match.
1. Check if `FP8`-related fields on `TransformerConfig` changed (used in
   `megatron_engine.py:_check_and_apply_fp8_config()`).

### Step 6: Run pre-commit and tests

```bash
pre-commit run --all-files
uv run pytest tests/test_estimate_num_params.py -v
```

If GPU is available:

```bash
uv run pytest tests/fp8/ -v
```

### Step 7: Report changes

Output a summary in this format:

```
## Upgrade Summary: megatron-core ${OLD_VERSION} → ${NEW_VERSION}

### Breaking Changes Found
- [file:line] description of change needed

### mbridge Compatibility
- mbridge version: ${MBRIDGE_VERSION} (compatible / needs bump to X.Y.Z)
- mbridge API changes affecting AReaL: (list or "none")

### API Additions (new optional params, informational)
- [upstream_file] description

### Files Modified
- path/to/file.py: description of change

### Tests
- ✅ pre-commit passed
- ✅ test_estimate_num_params passed
- ⬚ FP8 tests (requires GPU)
```

If there are unresolvable breaking changes, STOP and ask the user before proceeding.
