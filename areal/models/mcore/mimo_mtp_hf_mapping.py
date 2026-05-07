"""MiMo MTP HF name-mapping helper.

The upstream ``mbridge`` ``MiMoBridge`` (as of PR#1176 HEAD) does NOT translate
MTP-layer local keys such as ``mtp.layers.0.enorm.weight`` into their
HuggingFace counterparts under ``model.mtp_layers.0.*``.  As a result, the
MiMo-7B-RL checkpoint's MTP weights are silently skipped during
``load_weights_from_hf_with_mbridge_fast``, leaving the MTP head at random
initialisation (per-token CE \u2248 log(vocab)).

This module provides a pure-data mapping that mirrors
``areal.engine.megatron_utils.megatron._convert_mimo_mtp_param`` (the MCore \u2192 HF
direction already used by the weight-ship path), so that checkpoint loading
can be fixed without modifying the mbridge package itself.

Usage is limited to ``areal.models.mcore.hf_load`` which calls
``augment_local_to_hf_map_with_mtp`` after the bridge has populated the base
mapping.
"""
from __future__ import annotations

import re
from typing import Dict, List

# Matches both ``mtp.layers.{idx}.{rest}`` and the ``decoder.mtp_layers.{idx}.``
# variant that a few megatron-core revisions emit.
_MTP_GLOBAL_RE = re.compile(
    r"^(?:decoder\.)?mtp(?:\.layers|_layers)\.(\d+)\.(.+)$"
)

# MCore MTP suffix  ->  HF suffix under ``model.mtp_layers.{idx}.``.
# Multi-valued entries are merged by the existing qkv / gate-up handling in
# ``hf_load._convert_hf_weights_to_mcore``.
_MTP_SUFFIX_MAP: Dict[str, object] = {
    # MTP-specific layer norms and projections
    "enorm.weight":          "token_layernorm.weight",
    "hnorm.weight":          "hidden_layernorm.weight",
    "eh_proj.weight":        "input_proj.weight",
    "final_layernorm.weight": "final_layernorm.weight",

    # transformer_layer.* (reused Qwen2 decoder block)
    "transformer_layer.input_layernorm.weight":
        "input_layernorm.weight",
    "transformer_layer.self_attention.linear_qkv.layer_norm_weight":
        "input_layernorm.weight",
    "transformer_layer.self_attention.linear_qkv.weight": [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
    ],
    "transformer_layer.self_attention.linear_qkv.bias": [
        "self_attn.q_proj.bias",
        "self_attn.k_proj.bias",
        "self_attn.v_proj.bias",
    ],
    "transformer_layer.self_attention.linear_proj.weight":
        "self_attn.o_proj.weight",

    "transformer_layer.pre_mlp_layernorm.weight":
        "post_attention_layernorm.weight",
    "transformer_layer.mlp.linear_fc1.layer_norm_weight":
        "post_attention_layernorm.weight",
    "transformer_layer.mlp.linear_fc1.weight": [
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
    ],
    "transformer_layer.mlp.linear_fc2.weight":
        "mlp.down_proj.weight",
}


def mtp_mcore_name_to_hf_names(global_name: str) -> List[str]:
    """Return the HF keys matching one MCore MTP-global name.

    Returns an empty list if ``global_name`` does not look like an MTP entry
    or has no explicit mapping rule (e.g. ``_extra_state`` tails, unknown
    subcomponents - these are logged by the caller).
    """
    m = _MTP_GLOBAL_RE.match(global_name)
    if m is None:
        return []
    idx, rest = m.group(1), m.group(2)
    if rest.endswith("_extra_state"):
        return []
    rule = _MTP_SUFFIX_MAP.get(rest)
    if rule is None:
        return []
    prefix = f"model.mtp_layers.{idx}."
    if isinstance(rule, str):
        return [prefix + rule]
    return [prefix + s for s in rule]


def augment_local_to_hf_map_with_mtp(
    local_to_global_map: Dict[str, str],
    local_to_hf_map: Dict[str, List[str]],
    logger=None,
) -> int:
    """Inject MTP HF-name mappings into ``local_to_hf_map`` in-place.

    Iterates the local keys whose global name matches the MTP pattern.  If
    the existing mapping is empty (i.e. the base bridge did not know how to
    translate it), populate it from :func:`mtp_mcore_name_to_hf_names`.

    Returns the number of local keys patched.  A single [MTPCkptLoad-P1]
    summary is emitted via ``logger`` for verification.
    """
    patched = 0
    preview: List[str] = []
    for local_name, global_name in local_to_global_map.items():
        if "_extra_state" in local_name:
            continue
        m = _MTP_GLOBAL_RE.match(global_name)
        if m is None:
            continue
        cur = local_to_hf_map.get(local_name) or []
        if cur:
            continue
        hf_names = mtp_mcore_name_to_hf_names(global_name)
        if not hf_names:
            continue
        local_to_hf_map[local_name] = hf_names
        patched += 1
        if len(preview) < 3:
            preview.append(f"{local_name}->{hf_names}")
    if logger is not None:
        try:
            logger.info(
                "[MTPCkptLoad-P1] augment_local_to_hf_map_with_mtp "
                "patched=%d examples=%s",
                patched, preview,
            )
        except Exception:
            pass
    return patched
