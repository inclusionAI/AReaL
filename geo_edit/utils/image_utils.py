from __future__ import annotations

import base64
import io
import logging
import os
from io import BytesIO
from typing import Any, Dict, Tuple, Union

from PIL import Image

logger = logging.getLogger(__name__)


def image_to_data_url(
    image: Image.Image,
    *,
    image_format: str = "PNG",
    max_base64_bytes: int | None = 4 * 1024 * 1024,
) -> str:
    """Convert a PIL Image to a ``data:`` URL with optional size capping.

    Parameters
    ----------
    image:
        Source image.
    image_format:
        Initial encoding format (default ``"PNG"``).
    max_base64_bytes:
        Maximum length of the base64 payload.  When the initial encoding
        exceeds this limit the image is re-encoded as the highest-quality
        JPEG that still fits within the limit.
        Pass ``None`` to disable the limit entirely.
    """
    # --- fast path: encode with the requested format -------------------------
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    orig_kb = len(encoded) // 1024
    if max_base64_bytes is None or len(encoded) <= max_base64_bytes:
        fmt = image_format.strip().upper()
        mime = "image/jpeg" if fmt in {"JPG", "JPEG"} else "image/png"
        return f"data:{mime};base64,{encoded}"

    # --- slow path: re-encode as JPEG with high quality -----------------------
    rgb_image = image.convert("RGB") if image.mode != "RGB" else image

    best_encoded = None
    best_quality = None
    for quality in (98, 96, 94, 92, 90, 88, 86, 84, 82, 80, 78, 76, 74, 72, 70):
        buffer = io.BytesIO()
        rgb_image.save(buffer, format="JPEG", quality=quality)
        candidate = base64.b64encode(buffer.getvalue()).decode("ascii")
        if len(candidate) <= max_base64_bytes:
            best_encoded = candidate
            best_quality = quality
            break

    if best_encoded is None:
        buffer = io.BytesIO()
        rgb_image.save(buffer, format="JPEG", quality=70)
        best_encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        best_quality = 70

    logger.info(
        "Image exceeded limit (%dKB), compressed to JPEG q%d (%dKB)",
        orig_kb,
        best_quality,
        len(best_encoded) // 1024,
    )
    return f"data:image/jpeg;base64,{best_encoded}"


def load_image_safely(
    image_path: str, default_size: Tuple[int, int] = (336, 336)
) -> Image.Image:
    possible_paths = [
        image_path,
        os.path.abspath(image_path),
        os.path.join("data", image_path),
        os.path.join("../", image_path),
        os.path.join("../../", image_path),
    ]
    if "VLM_DATA_DIR" in os.environ:
        data_dir = os.environ["VLM_DATA_DIR"]
        possible_paths.extend(
            [
                os.path.join(data_dir, image_path),
                os.path.join(data_dir, os.path.basename(image_path)),
            ]
        )
    dataset_dirs = [
        "llava_cot_images",
        "data/llava_cot_images",
        "../llava_cot_images",
        "mulberry_images",
        "data/mulberry_images",
    ]
    for dataset_dir in dataset_dirs:
        if dataset_dir in image_path:
            base_name = image_path.split(dataset_dir)[-1].lstrip("/")
            for dir_variant in dataset_dirs:
                possible_paths.append(os.path.join(dir_variant, base_name))
    for path in possible_paths:
        try:
            if path and os.path.exists(path):
                return Image.open(path).convert("RGB")
        except Exception as exc:
            logger.debug("Failed to load from %s: %s", path, exc)
    logger.warning("Could not load image: %s, using placeholder", image_path)
    placeholder = Image.new("RGB", default_size, color="gray")
    from PIL import ImageDraw

    draw = ImageDraw.Draw(placeholder)
    draw.text((10, 10), "IMAGE NOT FOUND", fill="white")
    draw.rectangle(
        [5, 5, default_size[0] - 5, default_size[1] - 5], outline="white", width=2
    )
    return placeholder


def save_image(image: Union[Image.Image, bytes, Dict[str, Any]], path: str) -> None:
    """Save image from various formats to file.

    Args:
        image: PIL Image, bytes, or dict with "bytes" key
        path: Destination file path
    """
    if isinstance(image, Image.Image):
        image.save(path)
    elif isinstance(image, dict) and "bytes" in image:
        Image.open(BytesIO(image["bytes"])).save(path)
    elif isinstance(image, bytes):
        Image.open(BytesIO(image)).save(path)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
