from dataclasses import dataclass
from typing import Any

import torch
from mbridge.core.llm_bridge import LLMBridge
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
)
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)

from areal.utils import logging
from areal.utils.data import (
    MicroBatchList,
    MicroBatchSpec,
    amend_position_ids,
    pack_tensor_dict,
)
from areal.utils.perf_tracer import trace_perf, trace_scope

logger = logging.getLogger("Tree Training")


@dataclass
class TreeMicroBatchList(MicroBatchList):
    n_tree_tokens: list[int]
    n_total_tokens: list[int]
    from_tree_indices: list[list[int]]
    to_tree_indices: list[list[int]]


@dataclass
class TrieNode:
    idx: int
    value: int
    pos: int
    children: dict[int, "TrieNode"]


@dataclass
class TokenTree:
    batch_size: int
    n_tree_tokens: int
    n_total_tokens: int
    to_tree_indices: list[int]
    from_tree_indices: list[int]
    attn_mask: torch.Tensor


@trace_perf("tree_training._to_sequence_list")
def _to_sequence_list(data: dict[str, Any]):
    assert "input_ids" in data, "Input data must contain 'input_ids'"
    assert "attention_mask" in data, "Input data must contain 'attention_mask'"
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]

    sequences = []
    for ids, mask in zip(input_ids, attention_mask):
        seq = ids[mask.bool()].tolist()
        sequences.append(seq)
    return sequences


@trace_perf("tree_training.greedy_build_tree")
def greedy_build_tree(
    sequences: list[list[int]], max_tokens_per_tree: int
) -> list[TokenTree]:
    batch_idx = 0

    def _build_mb_tree():
        root = TrieNode(idx=-1, value=-1, children={}, pos=-1)
        nodes: list[TrieNode] = []
        from_tree_indices: list[int] = []
        pos = 0

        nonlocal batch_idx
        batch_size = 0
        # Build a token trie from the sequences
        for seq in sequences[batch_idx:]:
            if pos + len(seq) > max_tokens_per_tree:
                break
            node = root
            for token in seq:
                if token not in node.children:
                    child = TrieNode(idx=len(nodes), value=token, children={}, pos=pos)
                    node.children[token] = child
                    nodes.append(child)
                node = node.children[token]
                pos += 1
                from_tree_indices.append(node.idx)
            batch_idx += 1
            batch_size += 1
            if batch_idx >= len(sequences):
                break

        # Create attention mask by DFS
        with trace_scope("tree_training.greedy_build_tree.build_attention_mask"):
            attn_mask = torch.eye(
                len(nodes), len(nodes), dtype=torch.bool, device="cuda"
            )

            def dfs(node: TrieNode, ancestors: list[int]):
                for child in node.children.values():
                    attn_mask[node.idx, child.idx] = 1
                    for idx in ancestors:
                        attn_mask[idx, child.idx] = 1
                    dfs(child, ancestors + [node.idx])

            for node in root.children.values():
                dfs(node, [])

        return TokenTree(
            batch_size=batch_size,
            n_tree_tokens=len(nodes),
            n_total_tokens=pos,
            to_tree_indices=[node.pos for node in nodes],
            from_tree_indices=from_tree_indices,
            attn_mask=attn_mask,
        )

    trees = []
    while batch_idx < len(sequences):
        tree = _build_mb_tree()
        trees.append(tree)
    return trees


@trace_perf("tree_training.build_tree_input")
def build_tree_input(data: dict[str, Any], max_tokens_per_tree: int):
    """First construct token trees from input data, then convert input data into tree-packed format.
    The return value should be a list of dictionaries, each contains input_ids, attention_mask, and tree infos for a packed tree structure.
    The input id should be a flattened list of token ids in the tree structure with pre-ordered traversal.
    The attention mask represents the causal relationship between tokens in the token tree, in which entries are set to true when
    two tokens are in the same sequence and follows causal relationship (lower triangular causal mask).

    Returns:
        tuple[list[CompressedTokenNode], list[int], list[dict[str, Any]]]:
            ``roots`` of the packed token trees, token counts per tree, and tree-packed inputs with per-sequence indices.
    """
    sequences = _to_sequence_list(data)
    total_batch_size = len(sequences)
    trees = greedy_build_tree(sequences, max_tokens_per_tree=max_tokens_per_tree)

    processed_data = []
    offset = 0
    for tree in trees:
        mb = {k: v[offset : offset + tree.batch_size] for k, v in data.items()}

        mb = amend_position_ids(mb)
        with trace_scope("tree_training.build_tree_input.packing"):
            packed_mb = pack_tensor_dict(mb)
            tree_mb = {
                k: v[tree.from_tree_indices]
                for k, v in packed_mb.items()
                if k not in {"cu_seqlens", "max_seqlen"}
            }
            tree_mb["cu_seqlens"] = packed_mb["cu_seqlens"]
            tree_mb["max_seqlen"] = packed_mb["max_seqlen"]
            tree_mb["attention_mask"] = tree.attn_mask

        processed_data.append(tree_mb)
    return TreeMicroBatchList(
        data=data,
        mb_spec=MicroBatchSpec(n_mbs=1, max_tokens_per_mb=max_tokens_per_tree),
        forward_indices=list(range(total_batch_size)),
        backward_indices=list(range(total_batch_size)),
        group_lens=[tree.batch_size for tree in trees],
        padded_mbs=processed_data,
        n_tree_tokens=[tree.n_tree_tokens for tree in trees],
        n_total_tokens=[tree.n_total_tokens for tree in trees],
        from_tree_indices=[tree.from_tree_indices for tree in trees],
        to_tree_indices=[tree.to_tree_indices for tree in trees],
    )


############################## Model Initialization ##############################


def patch_bridge_for_tree_training():
    """Patch LLMBridge to support tree training with arbitrary attention mask."""
    origin_layer_spec_getter = LLMBridge._get_transformer_layer_spec

    def _patched_getter(self, vp_stage: int | None = None):
        from areal.models.mcore.flex_tree_attn import MegatronFlexTreeAttention

        spec: TransformerBlockSubmodules = origin_layer_spec_getter(self, vp_stage)
        for layer_spec in spec.layer_specs:
            if layer_spec.module is not TransformerLayer:
                logger.info(f"Skipping patch module: {layer_spec.module}")
                continue
            submodules: TransformerLayerSubmodules = layer_spec.submodules
            self_attn_spec = submodules.self_attetion
            if self_attn_spec.module is not SelfAttention:
                logger.info(f"Skipping patch module: {self_attn_spec.module}")
                continue
            self_attn_spec.submodules.core_attention = MegatronFlexTreeAttention

    LLMBridge._get_transformer_layer_spec = _patched_getter


############################## Model Forward ##############################


@trace_perf("tree_training.model_with_tree_attention_forward")
def model_with_tree_attention_forward(model, tree_input: dict[str, torch.Tensor]):
    """Patch LLMBridge.model_forward to support tree training with arbitrary attention mask."""
    input_ids = tree_input["input_ids"]
    attention_mask = tree_input["attention_mask"]
    position_ids = tree_input["position_ids"]

    # Transformer Engine expects True where values should be masked out.
    # attention_mask = (~attention_mask).unsqueeze(0).unsqueeze(0)
    # Add batch dimension for input_ids and position_ids
    input_ids = input_ids.unsqueeze(0)
    position_ids = position_ids.unsqueeze(0)
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
    )
    output = output.squeeze(0)  # Remove batch dimension
    return output
