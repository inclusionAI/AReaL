# SPDX-License-Identifier: Apache-2.0

"""Image token transport shared by the SGLang inference adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from areal.api.io_struct import ModelRequest


def compact_image_tokens(req: ModelRequest) -> list[int]:
    """Send one placeholder per image, retaining expanded IDs for training.

    SGLang expands image placeholders itself. Sending expanded runs can force
    it to decode and retokenize the prompt, changing sampled token boundaries.
    Only compact when the runs match the supplied images, preserving unknown
    processor formats. Always return a copy, including for text-only inputs.
    """
    image_token_id = getattr(req.processor, "image_token_id", None)
    if not req.image_data or not isinstance(image_token_id, int):
        return req.input_ids.copy()

    compact_ids: list[int] = []
    image_count = 0
    previous_is_image = False
    for token in req.input_ids:
        is_image = token == image_token_id
        if is_image and previous_is_image:
            continue
        if is_image:
            image_count += 1
        compact_ids.append(token)
        previous_is_image = is_image

    if image_count != len(req.image_data):
        return req.input_ids.copy()
    return compact_ids
