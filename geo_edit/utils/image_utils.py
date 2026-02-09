from __future__ import annotations

import base64
import io
import logging
import os
from typing import Any, Dict, Tuple

from PIL import Image

logger = logging.getLogger(__name__)


def image_to_data_url(image: Image.Image, *, image_format: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    fmt = image_format.strip().upper()
    mime = "image/jpeg" if fmt in {"JPG", "JPEG"} else "image/png"
    return f"data:{mime};base64,{encoded}"


def load_image_safely(image_path: str, default_size: Tuple[int, int] = (336, 336)) -> Image.Image:
    possible_paths = [image_path, os.path.abspath(image_path), os.path.join("data", image_path), os.path.join("../", image_path), os.path.join("../../", image_path)]
    if "VLM_DATA_DIR" in os.environ:
        data_dir = os.environ["VLM_DATA_DIR"]
        possible_paths.extend([os.path.join(data_dir, image_path), os.path.join(data_dir, os.path.basename(image_path))])
    dataset_dirs = ["llava_cot_images", "data/llava_cot_images", "../llava_cot_images", "mulberry_images", "data/mulberry_images"]
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
    draw.rectangle([5, 5, default_size[0] - 5, default_size[1] - 5], outline="white", width=2)
    return placeholder

