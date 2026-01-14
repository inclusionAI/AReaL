# Tree Data Visualization

This document explains how to dump and visualize tree-packed data from `build_packed_tree_batch` for debugging and analysis purposes.

## Enabling Data Dumping

Set the `TREE_PACK_DUMP_PATH` environment variable to enable dumping:

```bash
export TREE_PACK_DUMP_PATH=/path/to/dump/dir
```

When enabled, each call to `build_packed_tree_batch` will create two files per rank:
- `call{N}_rank{R}.pt` - Tensor data (input and output)
- `call{N}_rank{R}_trie.pkl` - Trie node structures (pickle format)

Where `N` is the call count (1-indexed) and `R` is the rank number.

## Dumped Data Structure

### Tensor Data (`.pt` file)

Loaded via `torch.load()`, contains a dictionary with:

```python
{
    "input_data": {
        "input_ids": torch.Tensor,      # Shape: [batch_size, seq_len]
        "attention_mask": torch.Tensor, # Shape: [batch_size, seq_len]
        # ... other input tensors
    },
    "output_mbs": [
        {
            # N = flattened_tree_#tokens (padded)
            "input_ids": torch.Tensor,      # Shape: [1, N]
            "attention_mask": torch.Tensor, # Shape: [N, N]
            "position_ids": torch.Tensor,   # Shape: [1, N]
            # ... other packed tensors (excludes trie_node)
        },
        # ... one dict per micro-batch
    ]
}
```

### Trie Data (`.pkl` file)

Loaded via `pickle.load()`, contains a list of `TrieNode` objects:

```python
[
    TrieNode(tree_id=0, ...),  # Root node of first tree
    TrieNode(tree_id=1, ...),  # Root node of second tree
    # ... one TrieNode per micro-batch
]
```

Each `TrieNode` has the following attributes:
- `tree_id`: Identifier of the tree
- `start_idx`: Starting index in flattened tree (-1 for root)
- `end_idx`: Ending index in flattened tree (-1 for root)
- `tokens`: List of token IDs in this node
- `sequence_ids`: IDs of sequences passing through this node
- `children`: Dict mapping token ID to child TrieNode
- `ancestors`: List of ancestor TrieNodes
- `nodes`: All descendant nodes in pre-order (only for root)

## Loading Dumped Data Manually

```python
import pickle
import torch

# Load tensor data
data = torch.load("/path/to/dump/call1_rank0.pt", map_location="cpu")
input_data = data["input_data"]
output_mbs = data["output_mbs"]

print(f"Input shape: {input_data['input_ids'].shape}")
print(f"Number of micro-batches: {len(output_mbs)}")

# Load trie nodes
with open("/path/to/dump/call1_rank0_trie.pkl", "rb") as f:
    trie_nodes = pickle.load(f)

print(f"Number of trees: {len(trie_nodes)}")
for trie in trie_nodes:
    print(f"  Tree {trie.tree_id}: {trie.num_tokens} tokens, sequences={trie.all_sequence_ids}")
```

## Using the Visualization Tool

### List Available Dumps

```bash
python -m areal.tools.visualize_tree --dump-dir /path/to/dump --list
```

Output:
```
Available dumps in /path/to/dump:
  call=1, rank=0
  call=1, rank=1
  call=2, rank=0
  call=2, rank=1
```

### Visualize a Specific Dump

```bash
python -m areal.tools.visualize_tree --dump-dir /path/to/dump --call 1 --rank 0
```

### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--dump-dir` | (required) | Directory containing dump files |
| `--call` | None | Call count number to visualize |
| `--rank` | 0 | Rank number to visualize |
| `--mask-granularity` | 128 | Max cells for attention mask display |
| `--max-tokens-display` | 5 | Max tokens shown per trie node |
| `--show-input-data` | False | Show input data summary |
| `--list` | False | List available dumps |

### Example Output

```
================================================================================
Output MicroBatches Summary: 2 micro-batches
================================================================================
  MicroBatch 0:
    input_ids: shape=torch.Size([1, 4096]), dtype=torch.int64
    attention_mask: shape=torch.Size([4096, 4096]), dtype=torch.bool
    position_ids: shape=torch.Size([1, 4096]), dtype=torch.int64

================================================================================
Trie Forest Visualization
================================================================================

============================================================
Forest Summary: 2 trees, 8192 total tokens, 8 sequences
Total shared prefix savings: 2048 tokens (compression ratio: 1.25x)
============================================================

Tree 0 (total tokens: 4096, sequences: [0, 1, 2, 3])
Shared prefix savings: 1024 tokens (compression ratio: 1.25x)
└── [0-511] 512 toks × 4 seqs (shared 1536) tokens=[1, 2, 3, ...] seqs=[0, 1, 2, 3]
    ├── [512-1023] 512 toks × 2 seqs (shared 512) tokens=[51, 52, ...] seqs=[0, 1]
    │   ├── [1024-1535] 512 toks × 1 seq tokens=[81, ...] seqs=[0]
    │   └── [1536-2047] 512 toks × 1 seq tokens=[101, ...] seqs=[1]
    └── [2048-2559] 512 toks × 2 seqs (shared 512) tokens=[121, ...] seqs=[2, 3]
        ├── [2560-3071] 512 toks × 1 seq tokens=[131, ...] seqs=[2]
        └── [3072-4095] 1024 toks × 1 seq tokens=[141, ...] seqs=[3]

================================================================================
Attention Mask Visualizations
================================================================================

MicroBatch 0 Attention Mask:
Attention Mask (4096x4096), displayed at 128x128 (block size: 32x32)
     ...
   0|█
  32|██
  64|███
  96|████
 128|█████
 ...
```

## Programmatic Visualization

You can also use the visualization functions directly in Python:

```python
import pickle
import torch

from areal.models.tree_attn.visualize import (
    visualize_attention_mask,
    visualize_forest,
    visualize_trie,
)

# Load data
data = torch.load("/path/to/dump/call1_rank0.pt", map_location="cpu")
with open("/path/to/dump/call1_rank0_trie.pkl", "rb") as f:
    trie_nodes = pickle.load(f)

# Visualize trie forest
print(visualize_forest(trie_nodes, max_tokens_display=5))

# Visualize individual trie
print(visualize_trie(trie_nodes[0], max_tokens_display=10))

# Visualize attention mask
attention_mask = data["output_mbs"][0]["attention_mask"]
visualize_attention_mask(attention_mask, granularity=64)
```

## Disabling Data Dumping

To disable dumping, unset or clear the environment variable:

```bash
unset TREE_PACK_DUMP_PATH
# or
export TREE_PACK_DUMP_PATH=""
```
