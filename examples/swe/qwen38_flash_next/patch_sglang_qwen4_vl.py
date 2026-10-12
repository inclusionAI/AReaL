# SPDX-License-Identifier: Apache-2.0
"""Align Qwen4 image preprocessing in the pinned SGLang installation."""

from pathlib import Path

from areal.engine._sglang_source_patch import patch_installed_module, patch_source

EXPECTED_SHA256 = "a78b01a05c387d04a185c631aa5037904c5e668d8f8217976b9b167aa1dd6666"
LEGACY_PATCHED_SHA256 = (
    "7c5c20550fed1a385daa07ce4c9b5a0b74151cea22d7762299fae47a8169e189"
)
PATCHED_SHA256 = "3a9d693446185f30d3fd21efc4c68ef6b27c7da757ccf0b0e29d6c32ee1e2e52"
TARGET_RELATIVE = Path("srt/multimodal/processors/qwen_vl.py")
OLD = '        if self.model_type in ("qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"):\n'
NEW = (
    "        if self.model_type in (\n"
    '            "qwen3_vl",\n'
    '            "qwen3_vl_moe",\n'
    '            "qwen3_5",\n'
    '            "qwen3_5_moe",\n'
    '            "qwen4_exp",\n'
    "        ):\n"
)
RESIZE_OLD = '            self.model_type != "paddleocr_vl"\n'
RESIZE_NEW = '            self.model_type not in ("paddleocr_vl", "qwen4_exp")\n'
DEVICE_OLD = "    async def process_mm_data_async(\n"
DEVICE_NEW = (
    "    def _fast_image_processor_device(self, processor):\n"
    '        if self.model_type == "qwen4_exp":\n'
    '            return "cpu"\n'
    "        return super()._fast_image_processor_device(processor)\n\n" + DEVICE_OLD
)


def patched_source(source: bytes) -> bytes:
    """Use the model's image factor and pixel bounds, refusing unknown sources.

    Qwen4 uses patch_size=16 and merge_size=2. Preserve the model's pixel bounds
    and let HF resize the original image on CPU, as it does during training.
    A separate PIL resize or the CUDA resize kernel can preserve token counts
    while changing the pixel tensor.
    """
    preprocessing = [(RESIZE_OLD, RESIZE_NEW), (DEVICE_OLD, DEVICE_NEW)]
    return patch_source(
        source,
        target_relative=TARGET_RELATIVE,
        replacements_by_sha256={
            EXPECTED_SHA256: [(OLD, NEW), *preprocessing],
            LEGACY_PATCHED_SHA256: preprocessing,
        },
        patched_sha256=PATCHED_SHA256,
    )


def main() -> None:
    patch_installed_module(__doc__, TARGET_RELATIVE, patched_source)


if __name__ == "__main__":
    main()
