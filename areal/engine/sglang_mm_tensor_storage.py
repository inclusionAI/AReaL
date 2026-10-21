# SPDX-License-Identifier: Apache-2.0
"""Compact split multimodal CPU tensor storage before SGLang pickle transport."""

from pathlib import Path

from areal.engine._sglang_source_patch import patch_installed_module, patch_source

EXPECTED_SHA256 = "08462d0013dec2715fcd12caca6d531ec1279e6409f0a66f4e2987a86a67b62b"
PATCHED_SHA256 = "064023614d3ecc143454cccf28766eb0468eaf53d19a43bfaec7ba6f9efb5f72"
OLD = """    return expanded_mm_items
"""
NEW = """    # Pickle serializes each tensor view's entire backing storage separately.
    # Compact split CPU features before tokenizer IPC and TP broadcasting.
    memo = {}
    compact_items = []
    for item in expanded_mm_items:
        compact_item = copy.copy(item)
        compact_item.feature = _compact_mm_cpu_storage(item.feature, memo)
        compact_item.precomputed_embeddings = _compact_mm_cpu_storage(
            item.precomputed_embeddings, memo
        )
        compact_item.model_specific_data = _compact_mm_cpu_storage(
            item.model_specific_data, memo
        )
        compact_items.append(compact_item)
    return compact_items
"""
ANCHOR = """def get_new_expanded_mm_items(original_mm_items):
"""
HELPER = """def _compact_mm_cpu_storage(value, memo):
    identity = id(value)
    if identity in memo:
        return memo[identity]
    if isinstance(value, torch.Tensor):
        result = value
        if (
            value.device.type == "cpu"
            and value.layout == torch.strided
            and value.untyped_storage().nbytes() > value.numel() * value.element_size()
        ):
            result = value.clone()
    elif isinstance(value, dict):
        result = {key: _compact_mm_cpu_storage(item, memo) for key, item in value.items()}
    elif isinstance(value, list):
        result = [_compact_mm_cpu_storage(item, memo) for item in value]
    elif isinstance(value, tuple):
        result = tuple(_compact_mm_cpu_storage(item, memo) for item in value)
    else:
        result = value
    memo[identity] = result
    return result


"""
TARGET_RELATIVE = Path("srt/managers/mm_utils.py")


def patched_source(source: bytes) -> bytes:
    """Preserve tensor values while avoiding per-view backing-storage copies."""
    return patch_source(
        source,
        target_relative=TARGET_RELATIVE,
        replacements_by_sha256={
            EXPECTED_SHA256: [(ANCHOR, HELPER + ANCHOR), (OLD, NEW)]
        },
        patched_sha256=PATCHED_SHA256,
    )


def main() -> None:
    patch_installed_module(__doc__, TARGET_RELATIVE, patched_source)


if __name__ == "__main__":
    main()
