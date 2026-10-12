# SPDX-License-Identifier: Apache-2.0
"""Apply the validated upstream #38346 fix to a disposable SGLang installation."""

from pathlib import Path

from areal.engine._sglang_source_patch import patch_installed_module, patch_source

EXPECTED_SHA256 = "bb57ce1e9abc4fbfcba2c9aaaf125b9e625983966497df57165b6d4c6461afe2"
PATCHED_SHA256 = "72c5d471a67729446ec9b1c0d810bc87bc85149c381c213adfb4119eddde9a7c"
TARGET_RELATIVE = Path("srt/layers/attention/qsa/qsa_indexer.py")
OLD = "            source_keys = token_k\n            source_rope = metadata.extend_rope_matrix\n"
NEW = (
    "            source_keys = token_k\n"
    "            group_locs = group_locs.clamp_max(source_keys.shape[0] - 1)\n"
    "            source_rope = metadata.extend_rope_matrix\n"
)


def patched_source(source: bytes) -> bytes:
    """Reject unknown or already-patched sources before any write."""
    return patch_source(
        source,
        target_relative=TARGET_RELATIVE,
        replacements_by_sha256={EXPECTED_SHA256: [(OLD, NEW)]},
        patched_sha256=PATCHED_SHA256,
        idempotent=False,
    )


def main() -> None:
    patch_installed_module(__doc__, TARGET_RELATIVE, patched_source)


if __name__ == "__main__":
    main()
