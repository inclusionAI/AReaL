# SPDX-License-Identifier: Apache-2.0
"""Align pinned SGLang Qwen3.5 preprocessing with the training HF processor.

Keep the model's pixel bounds and let HF resize original, CPU-decoded images
once on CPU. Install only in a disposable rollout environment before startup.
"""

from pathlib import Path

from areal.engine._sglang_source_patch import (
    install_source_patches,
    installed_sglang_root,
    patch_source,
)

TARGET_RELATIVE = Path("srt/multimodal/processors/qwen_vl.py")
BASE_RELATIVE = Path("srt/multimodal/processors/base_processor.py")
EXPECTED_SHA256 = "285fd0e0eef80d8c54db71724ef7cc369ae9379023437d3e2015deb65df69e17"
PATCHED_SHA256 = "04b9c53b6153c4c50e93997b298e5e90f8c27b7eadc917a216c84dad029cd1ec"
BASE_EXPECTED_SHA256 = (
    "2396ed6de130e17048eac64e97a2c665b7a33e8cdb42427b7ba5dd1e8be1ef61"
)
BASE_PATCHED_SHA256 = "eca3fb0ae1022ad3d2daeee2eb9710967d560e245b305322687397219bfbddf4"

LIMIT_OLD = (
    "            if envs.SGLANG_VLM_QWEN_LIMIT_PIXEL.get() and self.model_type in (\n"
    '                "qwen3_vl",\n'
    '                "qwen3_vl_moe",\n'
    '                "qwen3_5",\n'
    '                "qwen3_5_moe",\n'
    "            ):\n"
)
LIMIT_NEW = (
    "            if envs.SGLANG_VLM_QWEN_LIMIT_PIXEL.get() and self.model_type in (\n"
    '                "qwen3_vl",\n'
    '                "qwen3_vl_moe",\n'
    "            ):\n"
)
RESIZE_OLD = '            self.model_type != "paddleocr_vl"\n'
RESIZE_NEW = (
    '            self.model_type not in ("paddleocr_vl", "qwen3_5", "qwen3_5_moe")\n'
)
# _load_single_item is a classmethod used by multiprocessing workers. Disable
# nvJPEG decoding here so JPEGs, like PNGs, reach HF as original PIL images.
DECODE_OLD = "    supports_transformers_backend = True\n"
DECODE_NEW = DECODE_OLD + "    gpu_image_decode = False\n"
DEVICE_OLD = "            if _is_cpu or get_global_server_args().rl_on_policy_target is not None:\n"
DEVICE_NEW = (
    "            if (\n"
    "                _is_cpu\n"
    "                or get_global_server_args().rl_on_policy_target is not None\n"
    '                or getattr(self, "model_type", None) in ("qwen3_5", "qwen3_5_moe")\n'
    "            ):\n"
)


def patched_source(source: bytes) -> bytes:
    return patch_source(
        source,
        target_relative=TARGET_RELATIVE,
        replacements_by_sha256={
            EXPECTED_SHA256: (
                (LIMIT_OLD, LIMIT_NEW),
                (RESIZE_OLD, RESIZE_NEW),
                (DECODE_OLD, DECODE_NEW),
            )
        },
        patched_sha256=PATCHED_SHA256,
    )


def patched_base_source(source: bytes) -> bytes:
    return patch_source(
        source,
        target_relative=BASE_RELATIVE,
        replacements_by_sha256={BASE_EXPECTED_SHA256: ((DEVICE_OLD, DEVICE_NEW),)},
        patched_sha256=BASE_PATCHED_SHA256,
    )


def install(root: Path) -> None:
    install_source_patches(
        [
            (root / TARGET_RELATIVE, patched_source),
            (root / BASE_RELATIVE, patched_base_source),
        ],
        lock_path=root / ".areal-sglang-patches.lock",
    )


def main() -> None:
    install(installed_sglang_root())


if __name__ == "__main__":
    main()
