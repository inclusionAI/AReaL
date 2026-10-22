# SPDX-License-Identifier: Apache-2.0
"""Preserve text-domain visual IDs using explicit real-image offsets."""

from dataclasses import replace

import torch
from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput
from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor

from areal.utils.sglang_image import IMAGE_SPANS_KEY, validate_image_spans


class AReaLQwenImageProcessor(QwenVLImageProcessor):
    models = [
        model
        for model in QwenVLImageProcessor.models
        if model.__name__
        in (
            "Qwen3VLForConditionalGeneration",
            "Qwen3VLMoeForConditionalGeneration",
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5MoeForConditionalGeneration",
        )
    ]
    gpu_image_decode = False

    async def process_mm_data_async(
        self, image_data, input_text, request_obj, *args, **kwargs
    ):
        spans = (getattr(request_obj, "mm_sampling_kwargs", None) or {}).get(
            IMAGE_SPANS_KEY
        )
        if spans is None:
            return await super().process_mm_data_async(
                image_data, input_text, request_obj, *args, **kwargs
            )
        if (
            not isinstance(input_text, list)
            or request_obj.video_data
            or request_obj.audio_data
        ):
            raise ValueError("Explicit real-image spans require image-only input_ids")
        validate_image_spans(
            input_text, spans, self.mm_tokens.image_token_id, len(image_data)
        )

        # This prompt is used only for decoding/resize/HF preprocessing. The
        # language model receives the full original IDs below, without escaping.
        safe_ids = [
            token
            for _ in spans
            for token in (
                self.vision_start_token_id,
                self.mm_tokens.image_token_id,
                self.vision_end_token_id,
            )
        ]
        native = await super().process_mm_data_async(
            image_data, safe_ids, request_obj, *args, **kwargs
        )
        if len(native.mm_items) != len(spans) or any(
            not item.is_image() for item in native.mm_items
        ):
            raise ValueError("Processed image count does not match real-image spans")
        # Features/grid may be cached. Copy only per-request metadata, never the
        # tensors, and leave native hash/IPC transport lifecycle unchanged.
        items = [
            replace(
                item,
                offsets=[tuple(span)],
                model_specific_data=dict(item.model_specific_data),
            )
            for item, span in zip(native.mm_items, spans, strict=True)
        ]
        rope = self._compute_image_only_mrope_positions_from_offsets(
            input_len=len(input_text),
            mm_items=items,
            dtype=torch.long,
            device=torch.device("cpu"),
        )
        if rope is None:
            raise ValueError("Real-image spans do not match processed image grids")
        positions, delta = rope
        return replace(
            native,
            input_ids=list(input_text),
            mm_items=items,
            padded_input_ids=MultimodalProcessorOutput.build_padded_input_ids(
                input_text, items
            ),
            mrope_positions=positions.squeeze(1),
            mrope_position_delta=delta,
        )
