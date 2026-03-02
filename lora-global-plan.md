# LoRA Integration Plan for Archon Engine

## Progress Tracker

| Phase       | Status           | Description                     | Tests         |
| ----------- | ---------------- | ------------------------------- | ------------- |
| **Phase 1** | ✅ **COMPLETED** | Core LoRA Infrastructure        | 18/18 passing |
| **Phase 2** | 🔄 **NEXT**      | Checkpointing & PEFT Conversion | Not started   |
| **Phase 3** | ⏳ Pending       | Model Integration (Qwen2)       | Not started   |
| **Phase 4** | ⏳ Pending       | Engine Integration              | Not started   |
| **Phase 5** | ⏳ Pending       | Parallelization Support         | Not started   |

**Current Focus:** Phase 2 - Implementing PEFT format conversion and checkpoint I/O

______________________________________________________________________

## Context

This plan addresses integrating LoRA (Low-Rank Adaptation) training into the Archon
engine to enable parameter-efficient fine-tuning of large language models. The
motivation is to support LoRA training workflows while maintaining full compatibility
with the PEFT/HuggingFace ecosystem for checkpoint interoperability.

**Key Requirements:**

1. Save/load checkpoints in PEFT format for HuggingFace ecosystem compatibility
1. Avoid patching PEFT library - implement LoRA directly in areal repo using patterns
   from [torchtune](https://github.com/meta-pytorch/torchtune)
1. Work seamlessly with Archon's existing parallelization strategies (TP/CP/PP/FSDP)
1. Follow AReaL's design philosophy - extend existing patterns (StateDictAdapter)
1. For any breaking changes (like configuration merging), do not remain backward
   compatibility
1. LoRA must work with all parallelism approaches including FSDP/TP/PP/EP/CP

**Design Philosophy:** Following torchtune's proven approach, we implement custom
LoRALinear modules that are created **at model construction time** (not post-hoc
injection like PEFT). This integrates cleanly with Archon's meta device initialization →
parallelization → materialization flow.

**Key Design Decision - Model-Specific Builders:** After analyzing torchtune's
implementation, we adopt their **model-specific component builder** pattern rather than
generic post-hoc injection:

- **torchtune approach**: Each model (Llama, Qwen2, Gemma2) has dedicated
  `lora_{model}_self_attention()` and `lora_{model}_mlp()` functions that conditionally
  create LoRALinear or nn.Linear based on config
- **Benefit**: LoRA layers exist from model construction, enabling seamless integration
  with meta device init and parallelization
- **Model-agnosticism**: Achieved via shared LoRALinear module + model-specific builders
  (similar to how Archon's `parallelize_fn` is model-specific)

This approach is superior to generic `apply_lora_to_model()` post-hoc injection because:

1. Works naturally with meta device → parallelize → materialize flow
1. Handles model-specific quirks (different bias settings, qnorm/knorm, etc.)
1. No complex module surgery after parallelization
1. Follows Archon's existing pattern (model-specific `parallelize_fn`, `pipelining_fn`)

______________________________________________________________________

## Phase 1: Core LoRA Infrastructure - ✅ COMPLETED

**Status:** All components implemented, tested, and verified against PEFT library.

**Completed Tasks:**

- ✅ LoRALinear module with PEFT-compatible naming (lowercase lora_a/lora_b internally)
- ✅ AdapterModule protocol and utilities
- ✅ Comprehensive unit tests (18 tests, all passing)
- ✅ PEFT compatibility tests (exact numerical match)
- ✅ Pre-commit hooks passing (ruff format + lint)

**Test Results:**

- Forward pass vs PEFT: **0.00e+00** difference (exact match!)
- Gradient flow vs PEFT: **0.00e+00** difference (exact match!)
- All 18 tests passing on GPU

**Files Created:**

1. `areal/experimental/models/archon/lora/__init__.py`
1. `areal/experimental/models/archon/lora/lora_linear.py` (169 lines)
1. `areal/experimental/models/archon/lora/adapter.py` (113 lines)
1. `areal/tests/experimental/archon/test_lora_linear.py` (548 lines, 18 tests)

**Key Achievements:**

- Numerically equivalent to PEFT's `tuners.lora.Linear`
- Zero-initialization of `lora_b` matches PEFT convention
- Scaling factor (`alpha/rank`) matches PEFT exactly
- Gradient flow verified with non-zero `lora_b` weights

______________________________________________________________________

## Phase 2: Checkpointing & PEFT Conversion - 🔄 NEXT

**Status:** Implementation plan ready. Extending StateDictAdapter pattern.

**Key Design Decision:** Following AReaL's design philosophy, we extend the existing
`BaseStateDictAdapter` class to handle LoRA key conversion in a model-specific way,
rather than creating a separate converter module.

**Implementation Tasks:**

1. **Extend `Qwen2StateDictAdapter`** - ✅ DONE

   - Add 16 LoRA key mappings to `from_hf_map`
   - Handles case conversion: `lora_a` (Archon) ↔ `lora_A` (PEFT)
   - Add `to_peft_module_map` for config generation

1. **Create `convert.py`** (~100 lines)

   - `create_peft_adapter_config()` - Generate adapter_config.json
   - `merge_lora_weights_inplace()` - Merge LoRA into base weights (Phase 3+)

1. **Create `archon_lora_checkpoint.py`** (~200 lines)

   - `save_lora_adapter()` - Save adapter in PEFT format (uses
     state_dict_adapter.to_hf())
   - `load_lora_adapter()` - Load adapter from PEFT format (uses
     state_dict_adapter.from_hf())
   - `is_lora_adapter_checkpoint()` - Detect PEFT adapter checkpoints

1. **Extend `ArchonEngine`**

   - Add `lora_config` field
   - Extend `save()` with LoRA branches (adapter_only, full+adapter)
   - Extend `load()` with adapter detection

1. **Create comprehensive tests** (~300 lines)

   - State dict adapter key conversion tests
   - Adapter config generation tests
   - Checkpoint I/O round-trip tests
   - PEFT compatibility tests (optional)

**Acceptance Criteria:**

- Checkpoints saved by Archon load correctly in PEFT
- Checkpoints saved by PEFT load correctly in Archon
- Base model weights never modified during adapter save/load
- All 16 LoRA key patterns convert correctly
- Round-trip conversion preserves keys exactly

______________________________________________________________________

## Phase 3: Model Integration (Qwen2)

**Status:** Pending Phase 2 completion

**Goal:** Create model-specific LoRA builders for Qwen2 that conditionally create
LoRALinear layers at model construction time.

**Implementation Tasks:**

1. **Create `qwen2/model/lora_model.py`**

   - `lora_attention()` - Create Attention with conditional LoRA layers
   - `lora_feed_forward()` - Create FeedForward with conditional LoRA layers
   - `lora_transformer_block()` - Assemble block with LoRA components

1. **Add top-level builder in `qwen2/model/model.py`**

   - `qwen2_lora()` - Build complete Qwen2 model with LoRA enabled
   - Used by ArchonEngine when `lora_config` is set

1. **Modify `ArchonEngine._create_model_structure()`**

   - Call `qwen2_lora()` when `lora_config` is not None
   - Call standard `Qwen2Model()` otherwise

1. **Update optimizer creation**

   - Filter trainable params to only LoRA adapters
   - Freeze base model weights

**Integration Tests:**

1. Model-wise end-to-end single-GPU forward equivalence (vs PEFT)
1. Model-wise end-to-end single-GPU backward equivalence
1. Training step test (only LoRA weights change)
1. FSDP integration test (requires 2 GPUs)
1. FSDP checkpoint test

**Acceptance Criteria:**

- Single-GPU training matches PEFT numerically
- Only LoRA params have `requires_grad=True`
- Training step updates only LoRA weights
- Model creation works on meta device
- FSDP wrapping works correctly

______________________________________________________________________

## Phase 4: Engine Integration

**Status:** Pending Phase 3 completion

**Goal:** Full integration of LoRA into ArchonEngine training workflow.

**Key Changes:**

1. **Unified LoRAConfig** (`areal/api/cli_args.py`)

   - Merge existing FSDP LoRA config with new unified version
   - **Breaking change** per requirement #5
   - Add to `TrainEngineConfig` as `lora: LoRAConfig | None`

1. **Complete save/load integration**

   - Implement `merge_on_save` option (merge LoRA into base)
   - Handle distributed checkpointing
   - Support adapter swapping workflows

1. **End-to-end example**

   - Create `examples/math/gsm8k_sft_lora_archon.yaml`
   - Run full training workflow with LoRA

**Acceptance Criteria:**

- Config migration works (FSDP engine updated using the new `LoRAConfig` class)
- Example training runs successfully
- Adapter loads in HuggingFace/PEFT

______________________________________________________________________

## Phase 5: Parallelization Support

**Status:** Pending Phase 4 completion

**Goal:** Extend TP/PP/CP/EP parallelization to support LoRA layers.

**Implementation Tasks:**

1. **Extend TP parallelization** (`qwen2/infra/parallelize.py`)

   - ColwiseParallel base layers (wq/wk/wv/w1/w3):
     - `lora_a`: Replicate
     - `lora_b`: ColwiseParallel
   - RowwiseParallel base layers (wo/w2):
     - `lora_a`: RowwiseParallel
     - `lora_b`: Replicate

1. **PP (Pipeline Parallel) integration**

   - Verify LoRA layers correctly distributed across stages
   - LoRA lives within TransformerBlocks (already PP-partitioned)

1. **CP (Context Parallel) integration**

   - Verify sequence splitting works with LoRA

1. **EP (Expert Parallel) for MoE** (Future - Qwen3)

   - Design expert-specific LoRA patterns

**Integration Tests:**

1. TP test (requires 2 GPUs) - Verify parallelization correctness
1. PP test (requires 2 GPUs) - Verify gradient flow
1. CP test (requires 2 GPUs) - Verify sequence splitting
1. Distributed checkpoint tests

**Acceptance Criteria:**

- All parallelism strategies work with LoRA
- Numerical correctness (multi-GPU matches single-GPU)
- Checkpointing works with all parallelism modes

______________________________________________________________________

## Critical Files Reference

### Phase 1 (Completed)

- `areal/experimental/models/archon/lora/lora_linear.py`
- `areal/experimental/models/archon/lora/adapter.py`
- `areal/tests/experimental/archon/test_lora_linear.py`

### Phase 2 (Next)

- `areal/experimental/models/archon/lora/convert.py` (NEW)
- `areal/experimental/engine/archon_lora_checkpoint.py` (NEW)
- `areal/experimental/models/archon/qwen2/model/state_dict_adapter.py` (MODIFY - ✅ DONE)
- `areal/experimental/engine/archon_engine.py` (MODIFY)
- `areal/tests/experimental/engine/test_archon_lora_checkpoint.py` (NEW)

### Phase 3 (Future)

- `areal/experimental/models/archon/qwen2/model/lora_model.py` (NEW)
- `areal/experimental/models/archon/qwen2/model/model.py` (MODIFY)
- `areal/experimental/engine/archon_engine.py` (MODIFY)

### Phase 4 (Future)

- `areal/api/cli_args.py` (MODIFY - breaking change)
- `examples/math/gsm8k_sft_lora_archon.yaml` (NEW)

### Phase 5 (Future)

- `areal/experimental/models/archon/qwen2/infra/parallelize.py` (MODIFY)
- Parallelization tests (NEW)

______________________________________________________________________

## Design Trade-offs

### Chosen: At-Construction Injection

**Pros:** Clean integration with meta device init, proper FSDP sharding **Cons:**
Requires model recreation to switch LoRA config **Alternative Rejected:**
Post-construction injection (complex with DTensor)

### Chosen: Custom LoRALinear

**Pros:** Full control, FSDP2-compatible, no PEFT dependency for training **Cons:** Must
implement LoRA logic ourselves **Alternative Rejected:** Use PEFT directly (doesn't work
with meta device)

### Chosen: Extend StateDictAdapter

**Pros:** Follows AReaL patterns, centralized conversion logic, model-aware **Cons:**
None significant **Alternative Rejected:** Separate converter module (violates design
philosophy)

### Chosen: Separate Adapter Checkpoints

**Pros:** PEFT ecosystem compatibility, adapter swapping **Cons:** Slightly more complex
checkpoint logic **Alternative Supported:** Merged checkpoints also available via
`merge_on_save=true`

______________________________________________________________________

## Risks & Mitigations

**Risk 1:** TP parallelization incorrectness for LoRA layers

- **Mitigation:** Follow torchtune patterns, add numerical tests comparing single-GPU vs
  multi-GPU

**Risk 2:** PEFT format incompatibility

- **Mitigation:** Test loading saved adapters with actual PEFT library

**Risk 3:** PP (Pipeline Parallelism) with LoRA

- **Mitigation:** LoRA layers live within TransformerBlocks (already PP-partitioned),
  should work naturally

**Risk 4:** Breaking config changes affecting FSDP

- **Mitigation:** Update FSDP engine in same PR, run existing tests

______________________________________________________________________

## References

- **Torchtune LoRA:** `torchtune/torchtune/modules/peft/`
- **PEFT Library:** `peft/src/peft/`
- **Archon Engine:** `areal/experimental/engine/archon_engine.py`
- **Qwen2 Model:** `areal/experimental/models/archon/qwen2/`
- **Phase 2 Plan:** `/Users/fw/.claude/plans/shiny-bubbling-perlis.md`

______________________________________________________________________

## Success Metrics

- [ ] Phase 1: Core infrastructure with PEFT numerical equivalence ✅ DONE
- [ ] Phase 2: PEFT-compatible checkpoint I/O
- [ ] Phase 3: Single-GPU LoRA training matches PEFT
- [ ] Phase 4: End-to-end training example works
- [ ] Phase 5: All parallelism modes support LoRA
- [ ] Checkpoints load in HuggingFace/PEFT ecosystem
- [ ] Training metrics match baseline (non-LoRA)
