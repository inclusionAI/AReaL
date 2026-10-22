# SPDX-License-Identifier: Apache-2.0

"""Prepare a common image representation for Qwen-VL training and SGLang."""

from __future__ import annotations

import base64
import math
import os
from functools import lru_cache
from io import BytesIO
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from PIL.Image import Image


@lru_cache(maxsize=8)
def _processor_model_type(model_name_or_path: str) -> str | None:
    # Read raw config: Qwen4-Exp need not be registered with AutoConfig here.
    from transformers import PretrainedConfig

    config, _ = PretrainedConfig.get_config_dict(model_name_or_path)
    return config.get("model_type")


def _sglang_image_size(
    height: int, width: int, factor: int, min_pixels: int, max_pixels: int
) -> tuple[int, int]:
    """Match QwenVLImageProcessor.smart_resize_for_image in SGLang 0.5.18."""
    if min(height, width) <= 0 or max(height, width) / min(height, width) > 200:
        raise ValueError(
            "Qwen-VL images must have positive dimensions and aspect ratio <= 200"
        )
    h = max(factor, round(height / factor) * factor)
    w = max(factor, round(width / factor) * factor)
    if h * w > max_pixels:
        scale = math.sqrt(height * width / max_pixels)
        h = math.floor(height / scale / factor) * factor
        w = math.floor(width / scale / factor) * factor
    elif h * w < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * scale / factor) * factor
        w = math.ceil(width * scale / factor) * factor
    if min(h, w) <= 0:
        raise ValueError(
            "Qwen-VL pixel budget is too small for this image's aspect ratio"
        )
    return h, w


def prepare_sglang_qwen_images(
    processor: Any, images: list[Image]
) -> tuple[list[Image], list[str]] | None:
    """Resize once and use lossless image transport, without patching SGLang.

    Call only for a SGLang backend. Other processor families are left unchanged.
    The proxy and SGLang must inherit the same SGLANG_VLM_QWEN_LIMIT_PIXEL setting. This
    targets the Qwen3-VL/Qwen3.5 image path in SGLang 0.5.18, with an unmodified
    HF processor and no server-side image-size overrides.
    Qwen4-Exp also uses Qwen3VLProcessor, but has its own preprocessing policy;
    resolve model_type from the processor's tokenizer path before applying this
    model-specific pixel cap. Unknown model types are left unchanged.

    Send the returned PNGs, rather than the original JPEG/PNG, to SGLang. Sending
    originals would leave GPU JPEG decoding and the extra resize inconsistent
    with the tensors retained for training. Already resized RGB PNGs make both
    SGLang's PIL/tensor resize paths identities and avoid nvJPEG entirely.
    """
    if not any(cls.__name__ == "Qwen3VLProcessor" for cls in type(processor).__mro__):
        return None
    model_path = getattr(getattr(processor, "tokenizer", None), "name_or_path", None)
    if not isinstance(model_path, (str, os.PathLike)) or not model_path:
        return None
    if _processor_model_type(os.fspath(model_path)) not in (
        "qwen3_vl",
        "qwen3_vl_moe",
        "qwen3_5",
        "qwen3_5_moe",
    ):
        return None

    from PIL import Image as PILImage

    image_processor = processor.image_processor
    factor = int(image_processor.patch_size) * int(image_processor.merge_size)
    min_pixels = int(image_processor.size["shortest_edge"])
    max_pixels = int(image_processor.size["longest_edge"])
    limit_pixels = os.environ.get("SGLANG_VLM_QWEN_LIMIT_PIXEL", "true").lower()
    if limit_pixels not in ("true", "false", "1", "0", "yes", "no", "y", "n"):
        raise ValueError("Invalid SGLANG_VLM_QWEN_LIMIT_PIXEL boolean value")
    if limit_pixels in ("true", "1", "yes", "y") and max_pixels >= (32 * factor) ** 2:
        max_pixels //= 16
    if factor <= 0 or not 0 < min_pixels <= max_pixels:
        raise ValueError("Invalid Qwen-VL image factor or pixel bounds")

    # Match SGLang's PIL branch. The resulting PNG has the final dimensions, so
    # neither the HF tensor resizer nor SGLang's additional resizer changes it.
    resample = getattr(PILImage, os.environ.get("SGLANG_RESIZE_RESAMPLE", ""), None)
    prepared, encoded = [], []
    for image in images:
        height, width = _sglang_image_size(
            image.height, image.width, factor, min_pixels, max_pixels
        )
        image = image.convert("RGB").resize((width, height), resample=resample)
        buffer = BytesIO()
        image.save(buffer, format="PNG", compress_level=1)
        prepared.append(image)
        encoded.append(
            "data:image/png;base64,"
            + base64.b64encode(buffer.getvalue()).decode("ascii")
        )
    return prepared, encoded
