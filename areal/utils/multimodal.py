# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0

# Video processing and template are modified from verl under Apache license.
# Copyright (c) 2025, ByteDance.



import base64
from io import BytesIO
from typing import List, Optional
import torch

from PIL import Image
from PIL.Image import Image as ImageObject
from qwen_vl_utils import  fetch_video


QUESTION_TEMPLATE_VIDEO_QWEN = "You are a helpful assistant. The user asks a question, and then you solves it.\n\nPlease first think deeply about the question based on the given video, and then provide the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> the letter of your choice (A, B, C, or D) </answer>.\n\n Question: {question}"


def image2base64(images: List[ImageObject] | ImageObject) -> List[str] | str:

    if isinstance(images, ImageObject):
        images = [images]

    byte_images = []
    for image in images:
        with BytesIO() as buffer:
            image.save(buffer, format="PNG")
            buffer.seek(0)
            byte_image = base64.b64encode(buffer.read()).decode("utf-8")
            byte_images.append(byte_image)

    return byte_images


def pad_images_batch_to_max_size(images):
    max_width = max(image.size[0] for image in images)
    max_height = max(image.size[1] for image in images)

    padded_images = []

    for image in images:

        width, height = image.size

        padding_left = (max_width - width) // 2
        padding_top = (max_height - height) // 2

        padded_image = Image.new("RGB", (max_width, max_height), (0, 0, 0))
        padded_image.paste(image, (padding_left, padding_top))

        padded_images.append(padded_image)

    return padded_images


def process_video(
    video: str,
    nframes: Optional[int] = None,
    fps: Optional[float] = None,
    fps_min_frames: Optional[int] = None,
    fps_max_frames: Optional[int] = None,
) -> torch.Tensor:
    """
    Converts a video dict into a [n_frames, 3, H, W] tensor
    """


    assert nframes is None or fps is None, "Can't use both `nframes` or `fps`"

    # Shallow copy... since we might want to add some keys
    video_dict = {
        "video":video,
        "type":"video",
    }


    if nframes is not None:
        video_dict["nframes"] = nframes
    elif fps is not None:
        video_dict["fps"] = fps
        if fps_min_frames is not None:
            video_dict["min_frames"] = fps_min_frames
        if fps_max_frames is not None:
            video_dict["max_frames"] = fps_max_frames

    return fetch_video(video_dict)
