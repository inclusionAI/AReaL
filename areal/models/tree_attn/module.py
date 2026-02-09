from areal.models.tree_attn.constants import BLOCK_SIZE, USE_TRITON_TREE_ATTN
from areal.models.tree_attn.module_fsdp import (
    create_block_mask_from_dense,
    patch_fsdp_for_tree_training,
    restore_patch_fsdp_for_tree_training,
)
from areal.models.tree_attn.tree import (
    build_attention_mask_from_trie,
    build_block_mask_from_trie,
    build_triton_attn_data_from_trie,
)

# Conditionally import Triton functionality
try:
    from areal.models.tree_attn.triton_kernel import (
        TRITON_AVAILABLE,
        TreeAttentionData,
        tree_attention,
    )
except ImportError:
    TRITON_AVAILABLE = False
    TreeAttentionData = None
    tree_attention = None

# Conditionally import Megatron functionality
try:
    from areal.models.tree_attn.module_megatron import (
        PytorchFlexAttention,
        patch_bridge_for_tree_training,
    )
except ImportError:
    PytorchFlexAttention = None
    patch_bridge_for_tree_training = None

# Archon functionality
from areal.models.tree_attn.module_archon import TreeAttentionWrapper

__all__ = [
    # Shared constants
    "BLOCK_SIZE",
    "USE_TRITON_TREE_ATTN",
    # FSDP/common exports
    "create_block_mask_from_dense",
    "patch_fsdp_for_tree_training",
    "restore_patch_fsdp_for_tree_training",
    "build_attention_mask_from_trie",
    "build_block_mask_from_trie",
    "build_triton_attn_data_from_trie",
    # Triton exports (may be None if Triton not installed)
    "TRITON_AVAILABLE",
    "TreeAttentionData",
    "tree_attention",
    # Megatron exports (may be None if Megatron not installed)
    "PytorchFlexAttention",
    "patch_bridge_for_tree_training",
    # Archon exports
    "TreeAttentionWrapper",
]
