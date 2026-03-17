---
name: adapt-bailing-moe
description: Guide for adapting BailingMoeV2.5 (Lightning Attention + MLA + MoE) into AReaL's Megatron framework. Use when user wants to adapt, debug, or validate BailingMoe models.
---

# Adapt BailingMoe V2.5

Adapt BailingMoeV2.5 heterogeneous architecture (Lightning Attention + MLA + MoE) into AReaL's Megatron training framework, with full debugging and cross-engine validation methodology.

## When to Use

This skill is triggered when:

- User asks to adapt a new BailingMoe variant into AReaL
- User encounters high initial loss with BailingMoe models
- User wants to validate AReaL's BailingMoe implementation against another engine (e.g., HybridEngine)
- User needs to debug MLA or Lightning Attention layer divergence
- User asks about BailingMoe architecture details or weight mapping

## Architecture Overview

BailingMoeV2.5 is a heterogeneous Mixture-of-Experts model with:

- **Lightning Attention** (linear attention with learned decay): Most layers, uses `fla` library's `chunk_simple_gla` kernel
- **MLA** (Multi-Latent Attention, softmax attention with low-rank KV compression): Every `layer_group_size`-th layer
- **MoE MLP**: All layers except the first `first_k_dense_replace` layers use MoE with sigmoid routing + grouped TopK
- **Layer pattern** (layer_group_size=5): layers 0,1,2,3 = Lightning, layer 4 = MLA; layers 5,6,7,8 = Lightning, layer 9 = MLA; etc.

## Implementation Files

| File | Purpose |
|------|---------|
| `areal/models/mcore/lightning_attention.py` | Lightning Attention module (LightningCoreAttention, GroupRMSNorm, LightningSelfAttention) |
| `areal/models/mcore/bailing_moe.py` | HF config -> MLATransformerConfig conversion + heterogeneous layer spec construction |
| `areal/models/mcore/bailing_moe_bridge.py` | mbridge Bridge for HF <-> mcore weight mapping |
| `areal/models/mcore/registry.py` | Model architecture registration |
| `areal/models/mcore/hf_load.py` | HF weight loading with TP slicing (fused QKV format conversion) |
| `areal/engine/megatron_utils/megatron.py` | NCCL weight conversion (mcore -> HF for SGLang inference) |
| `areal/engine/megatron_engine.py` | Bridge import + trust_remote_code |
| `areal/engine/core/model.py` | VALID_MOE_MODELS list |
| `examples/bailing_moe_sft.yaml` | SFT training config example |

## Step-by-Step Adaptation Guide

### Phase 1: Config Conversion

Create `hf_to_mcore_config_bailing_moe(hf_config, dtype)` in `bailing_moe.py`:

- Map HF config fields to `MLATransformerConfig`
- Build `moe_layer_freq` as a per-layer list (0=dense, 1=MoE)
- Set MLA parameters: `q_lora_rank`, `kv_lora_rank`, `qk_head_dim`, `v_head_dim`, `qk_pos_emb_head_dim`
- Set MoE parameters: `num_moe_experts`, `moe_router_topk`, `moe_router_score_function="sigmoid"`, shared experts
- Use `hf_to_mcore_base_args()` from `common.py` for base settings

### Phase 2: Heterogeneous Layer Specs

Create `make_mcore_layer_specs_bailing_moe()` in `bailing_moe.py`:

- Build 4 layer spec variants: Lightning+Dense, Lightning+MoE, MLA+Dense, MLA+MoE
- Use `is_lightning_layer(layer_number, layer_group_size)` to determine layer type
- For MLA specs: use megatron-core's `get_gpt_layer_with_transformer_engine_spec`
- For Lightning specs: replace `self_attention` with custom `LightningSelfAttention`
- **Critical**: Restore real `input_layernorm` (TENorm) on Lightning specs because the custom module uses plain `TEColumnParallelLinear` without fused layernorm
- **Critical**: Handle PP slicing - use `get_num_layers_to_build()` and `get_transformer_layer_offset()` to slice `layer_specs` before constructing `TransformerBlockSubmodules`

### Phase 3: Lightning Attention Module

Implement in `lightning_attention.py`:

**LightningCoreAttention**:
- Wraps `fla.ops.simple_gla.chunk_simple_gla` Triton kernel
- Pre-computes `g_gamma` using ALiBi geometric slopes scaled by layer position
- Handles layout conversion: megatron `[S,B,H,D]` <-> fla `[B,T,H,D]`
- TP-aware: compute g_gamma with global head count, then slice to local heads

**LightningSelfAttention**:
- Fused QKV projection (megatron interleaved `[H,3,D]` format)
- Q/K RMSNorm
- RoPE (own RotaryEmbedding with `attn_head_dim * partial_rotary_factor` dim)
- Lightning Attention kernel
- Gate: `gate_norm(attn_output) * sigmoid(gate)` (NOT `attn_output * sigmoid(gate_norm(gate))`)
- Output projection

**GroupRMSNorm**:
- Group-wise RMSNorm for gate normalization
- Under TP, uses local head count with `tensor_model_parallel=True` weight metadata

### Phase 4: Weight Mapping

**mbridge Bridge** (`bailing_moe_bridge.py`):
- Lightning QKV: `linear_qkv` <-> `query_key_value` (with `[H,3,D]` <-> `[3,H,D]` conversion)
- MLA Q: dynamic dispatch between Q_DIRECT (`q_proj`) and Q_LORA (`q_a_proj` + `q_a_layernorm` + `q_b_proj`) based on `q_lora_rank`
- MLA KV: `kv_a_proj_with_mqa` (down) + `kv_a_layernorm` + `kv_b_proj` (up)
- MoE: router weight, expert_bias, per-expert weights (gate_proj/up_proj/down_proj), shared experts

**HF Loading with TP** (`hf_load.py`):
- `_convert_fused_qkv_weight()`: converts `[3,H,D,hidden]` -> `[H,3,D,hidden]` + TP head slicing
- Lightning layers have 1 fused QKV tensor; MLA layers have 3 separate Q/K/V tensors

**NCCL Conversion** (`megatron.py`):
- `convert_bailingmoe_to_hf()`: reverse conversion for pushing weights to SGLang
- Lightning QKV: `[H,3,D]` -> `[3,H,D]` rearrangement
- Register in `_CONVERSION_FN_REGISTRY` for model types: `bailing_moe_v2`, `bailing_moe_linear`, `bailing_hybrid`

### Phase 5: Registration

- `registry.py`: Add architecture names to `make_hf_and_mcore_config()` and `make_mcore_layer_specs()`
- `model.py`: Add to `VALID_MOE_MODELS`
- `megatron_engine.py`: Import bridge module for `@register_model` decorators

## Critical Bugs and Fixes (Historical Reference)

### Bug 1: g_gamma slope formula - loss 7.8 -> 0.567

**Root cause**: `layer_idx` offset error in Lightning Attention decay computation.

HF reference formula (`modeling_bailing_moe_v2_5.py:754`):
```python
slope = -build_slope_tensor(H) * (1 - (layer_idx - 1) / (N - 1) + 1e-5)
```

Initial AReaL implementation (wrong):
```python
layer_scale = 1.0 - self.layer_idx / max(self.num_layers - 1, 1) + 1e-5
```

The missing `- 1` in `(self.layer_idx - 1)` caused:
- Layer 0: slope off by ~5.3%
- Layer 18: slope off by ~50%
- **Layer 19: decay becomes exactly zero** (no decay at all, turning Lightning Attention into vanilla linear attention without decay)

**Fix**:
```python
layer_scale = 1.0 - (self.layer_idx - 1) / max(self.num_layers - 1, 1) + 1e-5
```

### Bug 2: g vs g_gamma parameter - loss 7.8 -> 14.42

Switching from `g_gamma` to `g` parameter in `chunk_simple_gla` made loss worse because:
- `g` path computes backward gradients via `chunk_local_cumsum(dg, reverse=True)` on what should be a constant buffer
- This gradient noise corrupts upstream parameter updates

**Fix**: Always use `g_gamma` (constant per-head scalar), never `g` (per-token tensor).

### Bug 3: chunk_lightning_attn vs chunk_simple_gla

Initial implementation used `fla.ops.lightning_attn.chunk_lightning_attn` which has its own built-in linear slopes formula: `-(8/H) * (1 - idx/N) * arange(H)`. This is NOT the ALiBi geometric slopes the model was trained with.

**Fix**: Switch to `fla.ops.simple_gla.chunk_simple_gla` with externally computed ALiBi g_gamma.

### Bug 4: gate_norm applied to wrong tensor

Initial implementation: `gate_norm(gate)` then `attn_output * sigmoid(normed_gate)` (wrong)
Reference implementation: `gate_norm(attn_output) * sigmoid(gate)` (correct)

**Fix**: Apply gate_norm to attention output, not to gate. Gate only goes through sigmoid.

### Bug 5: PP layer spec not sliced

`TransformerBlock._build_layers()` builds ALL specs in `layer_specs` without slicing. When PP>1, every rank was building all layers.

**Fix**: Pre-slice `layer_specs` using `get_num_layers_to_build()` and `get_transformer_layer_offset()` before constructing `TransformerBlockSubmodules`.

### Bug 6: QKV format conversion missing in hf_load

HF stores Lightning QKV as `[Q_all, K_all, V_all]` concatenated. Megatron expects `[q0,k0,v0, q1,k1,v1, ...]` interleaved. Without conversion, weights load successfully (same total shape) but are semantically wrong.

**Fix**: Add `_convert_fused_qkv_weight()` to transform `[3,H,D]` -> `[H,3,D]` during loading.

### Bug 7: TP bugs in Lightning Attention (all forward reshapes)

Initial implementation used global `num_attention_heads` everywhere. Under TP>1:
- Forward reshapes produced wrong shapes
- g_gamma was not sliced to local heads
- GroupRMSNorm weight was sized for all heads

**Fix**: Use `num_heads_per_partition` (local) for all forward reshapes, TP-slice g_gamma, mark GroupRMSNorm weight as `tensor_model_parallel=True`.

## Cross-Engine Validation Methodology

When validating AReaL's implementation against another engine (e.g., HybridEngine), follow this systematic approach.

### Step 1: Generate Shared Test Data

Create a script to generate identical input for both engines:

```python
# scripts/generate_test_data.py
# Two modes:
# - Real text (recommended): tokenize actual Chinese/English text, loss in 2-5 range
# - Random tokens (smoke test): random token IDs, loss ~13
# Output: test_input.pt with input_ids, labels, loss_mask, position_ids, attention_mask
```

**Important**: Use real text for meaningful comparison. Random tokens produce loss ~13, where exp amplification makes PPL differences look huge even for small loss differences.

### Step 2: Run Both Engines

Run both engines with identical settings:
- Same model weights (HF safetensors)
- Same parallelism: EP=8, TP=1, PP=1 (simplest config to minimize parallelism-induced differences)
- Same precision: bf16
- Same test input

Capture per-layer outputs via forward hooks:
```python
# Register hooks on every TransformerLayer and self_attention
# Capture: hidden_states, attention_output per layer
# Also capture: loss, logits, embedding output, final layernorm output
```

### Step 3: Compare Results

Compare using multiple metrics:
- **max_abs_diff**: Maximum absolute difference
- **mean_abs_diff**: Average absolute difference
- **rel_L2**: Relative L2 norm of difference
- **cosine_similarity**: Direction alignment (most informative)

Thresholds for bf16:
- **cosine > 0.999**: OK - within bf16 numerical precision
- **cosine 0.99-0.999**: WARN - investigate
- **cosine < 0.99**: FAIL - likely a code bug

### Step 4: Deep Dive on Divergent Layers

If specific layers diverge, register hooks on sub-modules to isolate the exact computation step:

For MLA layers, capture in order:
1. `attn_input` (hidden_states entering self_attention)
2. `q_proj_out` (after linear_q_proj)
3. `kv_down_out` (after linear_kv_down_proj)
4. `kv_up_out` (after linear_kv_up_proj, includes fused layernorm)
5. `core_attn_query` (post-RoPE query)
6. `core_attn_key` (post-RoPE key)
7. `core_attn_value` (value)
8. `core_attn_out` (attention output)
9. `proj_out` (after linear_proj)

For Lightning Attention layers, capture:
1. `attn_input`
2. `qkv_out` (after linear_qkv)
3. Q/K after RMSNorm
4. Q/K after RoPE
5. `core_attn_out` (after chunk_simple_gla)
6. `gate_out` (after gate projection)
7. `gated_output` (after gate_norm * sigmoid gating)
8. `proj_out`

The first step where cosine drops below 0.999 identifies the bug location.

### Step 5: Verify Weight Loading

Create a weight verification script that:
1. Records random-init weight checksums (mean, std, first values) before loading
2. Loads HF weights
3. Verifies checksums changed (weights actually loaded)
4. Checks for all-zero weights (might indicate unloaded parameters)
5. Compares weight statistics between engines

## Validation Results Summary (Mini V2.5 Model)

### End-to-End

| Round | Loss Diff | Root Cause |
|-------|-----------|------------|
| Round 1 | 69.9% (13.85 vs 23.53) | HybridEngine used fla's default linear slopes instead of ALiBi geometric slopes |
| Round 2 | 0.68% (13.85 vs 13.75) | TE version difference (2.7 vs 2.11) in MLA attention kernel |
| Round 3 | Real text: AReaL loss=1.512, PPL=4.54 | Normal range |

### Per-Layer Analysis

- **Lightning Attention layers**: cosine > 0.999 (perfect match)
- **MLA layers**: cosine ~0.92-0.95 (TE version difference, not code bug)

### MLA Deep Dive (Layer 4)

| Computation Step | Cosine | Status |
|-----------------|--------|--------|
| attn_input | 0.9997 | OK |
| q_proj_out | 0.9997 | OK |
| kv_down_out | 0.9997 | OK |
| kv_up_out (fused layernorm) | 0.9995 | OK |
| core_attn_query (post-RoPE) | 0.9997 | OK |
| core_attn_key (post-RoPE) | 0.9999 | OK |
| core_attn_value | 0.9993 | OK |
| **core_attn_out** | **0.9216** | **FAIL** |
| proj_out | 0.9263 | FAIL |

Root cause: TransformerEngine version difference (HybridEngine TE 2.7.0 vs AReaL TE 2.11.0). All computation before `TEDotProductAttention.forward()` matches perfectly, proving weight loading, projections, and RoPE are correct.

## Diagnostic Scripts Reference

| Script | Purpose | Location |
|--------|---------|----------|
| `generate_test_data.py` | Generate shared test input (real text or random tokens) | `comparison/scripts/` |
| `compare_areal.py` | AReaL-side forward pass with hooks | `comparison/scripts/` |
| `compare_hybrid_engine_hf.py` | HybridEngine-side forward with HF weight loading | `comparison/scripts/` |
| `compare_results.py` | Offline comparison of both engines' outputs | `comparison/scripts/` |
| `debug_mla_areal.py` | AReaL MLA sub-module intermediate capture | `comparison/scripts/` |
| `debug_mla_hybrid.py` | HybridEngine MLA sub-module intermediate capture | `comparison/scripts/` |
| `compare_mla_debug.py` | Offline MLA intermediate comparison | `comparison/scripts/` |
| `test_te_attn_directly.py` | Standalone TE DotProductAttention test | `comparison/scripts/` |
| `verify_weights.py` | Weight loading verification | `comparison/scripts/` |

## Common Mistakes

- Using `chunk_lightning_attn` instead of `chunk_simple_gla` (fla's built-in slopes override your ALiBi slopes)
- Passing decay via `g` parameter instead of `g_gamma` (causes gradient noise on constant buffer)
- Using `layer_idx` directly instead of `(layer_idx - 1)` in slope formula (last layer gets zero decay)
- Applying `gate_norm` to the gate signal instead of the attention output
- Using global `num_attention_heads` instead of local `num_heads_per_partition` under TP
- Not slicing `layer_specs` for PP (every rank builds all layers)
- Missing `[3,H,D]` <-> `[H,3,D]` QKV format conversion between HF and megatron
- Forgetting to set `input_layernorm` on Lightning specs (needed because Lightning uses plain TEColumnParallelLinear, not fused layernorm variant)
- Not registering bridge import in `megatron_engine.py` (mbridge won't find the model type)
- Attributing MLA cosine ~0.92 to code bugs when it's actually TE version differences

## Reference Implementations

| Component | AReaL File | HF Reference |
|-----------|-----------|--------------|
| Lightning Attention | `areal/models/mcore/lightning_attention.py` | `antllm/models/bailing_moe/modeling_bailing_moe_v2_5.py:705+` |
| ALiBi slopes | `_build_alibi_slopes()` in lightning_attention.py | `build_slope_tensor()` in modeling_bailing_moe_v2_5.py |
| MLA | megatron-core `multi_latent_attention.py` | `BailingMoeV2_5MultiLatentAttention` in modeling_bailing_moe_v2_5.py |
| MoE Router | megatron-core `TopKRouter` | `BailingMoeV2_5MoE` in modeling_bailing_moe_v2_5.py |
| Megatron-LM reference | `/home/admin/Megatron-LM/megatron/core/transformer/attention.py` | Linear attention with `_build_slope_tensor` and `g_gamma=self.slope` |

## Documentation

| Document | Purpose |
|----------|---------|
| `docs/bailing-moe-v2.5-adaptation-plan.md` | Master adaptation plan with 6 phases |
| `docs/bailing-moe-v2.5-code-review.md` | First code review (4C, 3H, 5M, 4L issues) |
| `docs/bailing-moe-v2.5-code-review-v2.md` | Follow-up review confirming fixes |
| `docs/bailing-moe-v2.5-high-loss-bugs-v2.md` | Root cause analysis of high initial loss (7.8) |
| `docs/bailing-moe-v25-comparison-plan.md` | Cross-engine validation methodology and results |

______________________________________________________________________

<!--
================================================================================
                            MAINTAINER GUIDE
================================================================================

Location: .claude/skills/adapt-bailing-moe/SKILL.md
Invocation: /adapt-bailing-moe

## Purpose

Comprehensive guide for BailingMoeV2.5 adaptation into AReaL, covering:
- Implementation steps (config, layer specs, modules, weight mapping, registration)
- Historical bug catalog with root causes and fixes
- Cross-engine validation methodology
- Diagnostic script reference

## How to Update

### When New BailingMoe Variants Are Added
1. Update Architecture Overview section
2. Add variant-specific config conversion notes
3. Update weight mapping if new layer types are introduced

### When New Bugs Are Found and Fixed
1. Add to "Critical Bugs and Fixes" section with root cause and fix
2. Update "Common Mistakes" section
3. Update validation results if re-validated

### When Validation Methodology Improves
1. Update "Cross-Engine Validation Methodology" section
2. Add new diagnostic scripts to the reference table

### When Implementation Files Change
1. Update "Implementation Files" table
2. Update relevant Phase descriptions
3. Verify weight mapping documentation is still accurate

================================================================================
-->
