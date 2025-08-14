# Copyright 2025 Ant Group Inc.
# Licensed under the Apache License, Version 2.0

# Video processing and template are modified from verl under Apache license.
# Copyright (c) 2025, ByteDance.

import base64
import os
from io import BytesIO
from typing import List, Optional

import torch
from PIL import Image
from PIL.Image import Image as ImageObject

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


