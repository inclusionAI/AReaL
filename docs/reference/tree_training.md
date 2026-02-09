# Tree Training

This document describes AReaL's tree training feature, which enables efficient RL
training by sharing prefix computations across sequences with common prefixes.

## Overview

Tree training is an optimization technique that exploits prefix sharing among multiple sequences in a batch. 
Instead of processing each sequence independently, sequences with shared prefixes are packed into a tree structure
where common prefix tokens are computed only once.

This is particularly beneficial for agentic RL training where:

- Multiple responses are sampled from the same prompt (e.g., `n_samples > 1`)
- Prompts share common system prompts or few-shot examples
- Reference model log probabilities need to be computed for the same prefix multiple
  times

### Key Benefits

| Benefit                  | Description                                                |
| ------------------------ | ---------------------------------------------------------- |
| **Reduced FLOPs**        | Shared prefixes are computed once instead of N times       |
| **Memory Efficiency**    | Trie compression reduces peak memory for attention masks   |
| **Maintained Accuracy**  | Mathematically equivalent to standard sequence processing  |
| **Automatic Packing**    | Greedy algorithm packs sequences into trees automatically  |

### Supported Backends

| Backend  | Status                   | Notes                                  |
| -------- | ------------------------ | -------------------------------------- |
| FSDP     | Supported                | Via FlexAttention with block masks     |
| Megatron | Supported                | Via PytorchFlexAttention module        |
| Archon   | Supported (Experimental) | Via TreeAttentionWrapper               |

## Configuration

### Enabling Tree Training

Enable tree training via the `enable_tree_training` option in `TrainEngineConfig`:

```yaml
actor:
  enable_tree_training: true
  pad_to_maximum: true
  mb_spec:
    max_tokens_per_mb: 8192  # Must be set for tree training
```

### Required Configuration

| Parameter                  | Type | Required | Description                          |
| -------------------------- | ---- | -------- | ------------------------------------ |
| `enable_tree_training`     | bool | Yes      | Enable tree-based sequence packing   |
| `pad_to_maximum`           | bool | Yes      | Must be `true` for tree training     |
| `mb_spec.max_tokens_per_mb`| int  | Yes      | Max tokens per tree (must be set)    |

Additionally, when tree training is enabled `max_tokens_per_mb` must be a multiple of `BLOCK_SIZE` (128).

### Attention Type Selection

For Archon engine, the attention type is configured via `archon.attn_type`:

```yaml
actor:
  enable_tree_training: true
  archon:
    attn_type: tree  # Automatically set when enable_tree_training=true
```

| Attention Type | Description                                     |
| -------------- | ----------------------------------------------- |
| `varlen`       | Variable-length attention (default, no sharing) |
| `sdpa`         | Scaled dot-product attention                    |
| `tree`         | Tree attention with prefix sharing              |

## Architecture

### Tree Building Process

```
Input Sequences              Packed Tree               Attention Mask

Seq0: [A, B, C, D]               [A]                    Causal mask with
Seq1: [A, B, E, F]               / \                    tree structure:
Seq2: [A, G, H]                [B] [G]                  tokens can attend to
                              / \    \                  their ancestors only
                           [C] [E]   [H]
                            |    |
                           [D]  [F]
```

The tree building process:

1. **Extract sequences**: Parse input_ids using attention_mask to get actual tokens
2. **Greedy packing**: Insert sequences into tries using first-fit decreasing strategy
3. **Trie compression**: Merge linear chains into single compressed nodes
4. **Mask generation**: Build block masks for efficient FlexAttention computation

### Data Structures

**Key file:** `areal/models/tree_attn/tree.py`

#### TrieNode

The `TrieNode` dataclass represents a node in the compressed prefix tree:

```python
@dataclass
class TrieNode:
    tree_id: int           # Identifier of the tree this node belongs to
    start_idx: int         # Starting index in flattened representation
    end_idx: int           # Ending index (inclusive)
    tokens: list[int]      # Token IDs stored in this node
    sequence_ids: list[int]  # IDs of sequences passing through
    children: dict[int, TrieNode]  # Child nodes by diverging token
    ancestors: list[TrieNode]      # Ancestor nodes from root
```

For root nodes, `start_idx` and `end_idx` are -1, and the `nodes` list tracks all
descendant nodes in pre-order traversal.

### Log Probability Computation

**Key file:** `areal/models/tree_attn/functional.py`

Computing log probabilities for packed trees requires special handling because:

1. Labels cannot be obtained by simply rolling `input_ids` (sequences share positions)
2. Each sequence must recover its original token order from the tree structure
3. Caching is used to avoid redundant computation for shared prefixes

The `gather_packed_tree_logprobs_entropy` function:

1. Iterates over each sequence in the tree
2. For each node the sequence passes through:
   - Computes internal logprobs (predictions within the node)
   - Computes transition logprobs (predictions to child nodes)
3. Caches results at node level for sequences sharing the same prefix
4. Concatenates all logprobs to reconstruct per-sequence results

### Attention Mechanisms

Tree training uses two attention implementations depending on hardware support:

#### FlexAttention with Block Masks (Default)

Uses PyTorch's `torch.nn.attention.flex_attention` with `BlockMask`:

- Block size: 128 tokens (configurable via `BLOCK_SIZE`)
- Efficient for GPU computation with sparse attention patterns
- Requires sequences padded to block size multiples

#### Triton Tree Attention (Optional)

When Triton is available and `USE_TRITON_TREE_ATTN=True`:

- Custom Triton kernels for tree-structured attention
- Parent array representation for efficient tree traversal
- Precomputed sparse block indices for kernel dispatch

## Engine Integration

### FSDP Engine

**Key file:** `areal/engine/fsdp_engine.py`

FSDP integration uses monkey patching to replace standard attention with tree attention:

```python
# In FSDPEngine.initialize()
patch_fsdp_for_tree_training(enable=self.enable_tree_training)
```

During forward pass, block masks are lazily created just before model execution:

```python
if self.enable_tree_training and ctx.trie_node is not None:
    if USE_TRITON_TREE_ATTN and TRITON_AVAILABLE:
        triton_attn_data = build_triton_attn_data_from_trie(ctx.trie_node, padded_size)
        inputs["triton_attn_data"] = triton_attn_data
    else:
        block_mask = build_block_mask_from_trie(ctx.trie_node, padded_size, self.device)
        inputs["block_mask"] = block_mask
```

### Megatron Engine

**Key file:** `areal/engine/megatron_engine.py`

Megatron uses `patch_bridge_for_tree_training` context manager during model creation:

```python
with patch_bridge_for_tree_training(self.enable_tree_training):
    self.bridge = mbridge.AutoBridge.from_pretrained(self.config.path)
```

For gradient checkpointing compatibility, dense attention masks (tensors) are used
instead of BlockMask objects, since `save_for_backward()` can only serialize tensors.

### Archon Engine

**Key files:** `areal/experimental/engine/archon_engine.py`, `archon_runner.py`

Archon uses `TreeAttentionWrapper` which handles tree attention internally:

```python
# In SequentialRunner.run()
if ctx.trie_node is not None:
    if USE_TRITON_TREE_ATTN and TRITON_AVAILABLE:
        triton_attn_data = build_triton_attn_data_from_trie(ctx.trie_node, padded_size)
    else:
        block_mask = build_block_mask_from_trie(ctx.trie_node, padded_size, device)

logits = self.model(
    inputs["input_ids"],
    inputs["position_ids"],
    cu_seqlens=cu_seqlens,
    max_seqlen=max_seqlen,
    block_mask=block_mask,
    triton_attn_data=triton_attn_data,
)
```

## Metrics and Monitoring

Tree training tracks the following metrics:

| Metric             | Description                                        |
| ------------------ | -------------------------------------------------- |
| `tree_token_ratio` | Ratio of tree tokens to original tokens (< 1.0)    |

A lower `tree_token_ratio` indicates more prefix sharing and greater efficiency gains.
For example, a ratio of 0.6 means 40% of tokens were saved through prefix sharing.

## Constraints and Limitations

### Current Limitations

| Constraint              | Description                                              |
| ----------------------- | -------------------------------------------------------- |
| No PP with FSDP         | Pipeline parallelism not supported with FSDP tree mode   |
| No CP with tree         | Context parallelism (CP > 1) incompatible with tree mode |
| No SP with tree (FSDP)  | Sequence parallelism (SP > 1) incompatible with tree     |
| Critic not supported    | Tree training with critic models not yet implemented     |
| Block size alignment    | `max_tokens_per_mb` must be multiple of 128              |

### Sequence Length Constraints

- Each sequence must fit within `max_tokens_per_mb`
- Sequences exceeding this limit will raise a `ValueError`
- Consider increasing `max_tokens_per_mb` or truncating long sequences

### Numerical Accuracy

FlexAttention may introduce numerical precision differences compared to standard attention
implementations. This can cause training instability with Mixture of Experts (MoE) models
when tree training is enabled. If you experience unstable training with MoE architectures,
consider disabling tree training.

## Example Configuration

Complete example for GRPO training with tree training enabled:

```yaml
experiment_name: grpo_tree_training
trial_name: run1

actor:
  path: Qwen/Qwen2.5-7B-Instruct
  enable_tree_training: true
  mb_spec:
    max_tokens_per_mb: 16384  # Must be multiple of 128

gconfig:
  n_samples: 8  # Multiple samples benefit most from tree training
  max_new_tokens: 1024

train_dataset:
  batch_size: 64
```

## Troubleshooting

### Common Issues

**"max_tokens_per_tree must be a multiple of BLOCK_SIZE"**

Ensure `mb_spec.max_tokens_per_mb` is divisible by 128:

```yaml
mb_spec:
  max_tokens_per_mb: 8192  # 8192 / 128 = 64 blocks
```

**"Sequence length exceeds max_tokens_per_tree"**

Either increase `max_tokens_per_mb` or reduce `max_new_tokens`:

```yaml
mb_spec:
  max_tokens_per_mb: 32768  # Increase limit
gconfig:
  max_new_tokens: 512       # Or reduce generation length
```

**"Tree training cannot be enabled with context parallelism"**

Disable context parallelism when using tree training:

```yaml
allocation_mode: "sglang:d4t4 + fsdp:d8"  # No 'c' dimension
```

## See Also

- [Allocation Mode](alloc_mode.md) - GPU allocation and parallelism configuration
- [Algorithm Performance](../best_practices/algo_perf.md) - Performance optimization tips
- [Handling OOM](../best_practices/handling_oom.md) - Memory management strategies
