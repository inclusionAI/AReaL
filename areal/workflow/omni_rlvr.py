"""Omni multimodal RLVR workflow supporting text, image, audio, and video inputs.

Extends :class:`VisionRLVRWorkflow` with audio and video processing stages.
The processor handles all modalities simultaneously; the resulting
``multi_modal_input`` carries vision tensors (``pixel_values``,
``image_grid_thw``, ``pixel_values_videos``, ``video_grid_thw``) and audio
tensors (``input_features``, ``feature_attention_mask``).
"""

import uuid
from collections.abc import Callable
from typing import Any, cast

import torch
from transformers import AutoProcessor, PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest
from areal.api.reward_api import AsyncRewardWrapper
from areal.dataset.audio import audios2base64
from areal.utils import logging
from areal.utils.dynamic_import import import_from_string
from areal.utils.image import image2base64
from areal.workflow.vision_rlvr import VisionRLVRWorkflow

logger = logging.getLogger("OmniRLVRWorkflow")

_VISION_KEYS = (
    "pixel_values",
    "image_grid_thw",
    "pixel_values_videos",
    "video_grid_thw",
)
_AUDIO_KEYS = ("input_features", "feature_attention_mask")


class OmniRLVRWorkflow(VisionRLVRWorkflow):
    """RLVR workflow for Omni multimodal models (text + image + audio + video)."""

    def __init__(
        self,
        reward_fn: Callable[..., Any] | str,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        processor: AutoProcessor | str,
        enable_thinking: bool,
    ):
        super().__init__(
            reward_fn,
            gconfig,
            tokenizer,
            processor,
            enable_thinking,
        )

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        if isinstance(self.reward_fn, str):
            self.reward_fn = import_from_string(self.reward_fn)
            self.async_reward_fn = AsyncRewardWrapper(self.reward_fn)

        # --- Build processor kwargs for all present modalities ---
        processor_callable = cast(Callable[..., dict[str, Any]], self.processor)
        proc_kwargs: dict[str, Any] = {
            "text": data["messages"],
            "padding": False,
            "return_tensors": "pt",
        }

        # Images may arrive as base64 strings (RPC-safe) or PIL objects.
        pil_images = None
        byte_images = None
        if data.get("images"):
            from areal.utils.image import base642image

            raw_imgs = data["images"]
            if isinstance(raw_imgs[0], str):
                byte_images = raw_imgs
                pil_images = [base642image(b) for b in raw_imgs]
            else:
                pil_images = raw_imgs
                byte_images = image2base64(raw_imgs)
            proc_kwargs["images"] = pil_images

        if data.get("videos"):
            proc_kwargs["videos"] = data["videos"]
        if data.get("audios"):
            proc_kwargs["audios"] = data["audios"]

        processed_input = processor_callable(**proc_kwargs)
        input_ids: list[int] = processed_input["input_ids"].tolist()[0]
        byte_audios = audios2base64(data["audios"]) if data.get("audios") else None
        byte_videos = None
        if data.get("video_paths"):
            from areal.dataset.video import videos2base64

            byte_videos = videos2base64(data["video_paths"])

        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            image_data=byte_images,
            audio_data=byte_audios,
            video_data=byte_videos,
            vision_msg_vllm=(
                [data["messages_chat"]] if "messages_chat" in data else None
            ),
            gconfig=self.gconfig.new(n_samples=1),
            tokenizer=self.tokenizer,
            processor=self.processor,
        )

        prompt_str = self.tokenizer.decode(input_ids)

        resp, reward = await self._collect_samples(engine, req, prompt_str, data)

        # --- Build result tensor dict ---
        seq = resp.input_tokens + resp.output_tokens
        logprobs = [0.0] * resp.input_len + resp.output_logprobs
        loss_mask = [0] * resp.input_len + [1] * resp.output_len
        versions = [-1] * resp.input_len + resp.output_versions

        # --- Build multi_modal_input with vision, video, and audio fields ---
        mm_entry: dict[str, torch.Tensor] = {}
        for key in _VISION_KEYS + _AUDIO_KEYS:
            if key in processed_input:
                mm_entry[key] = processed_input[key]
        multi_modal_input = [mm_entry] if mm_entry else []

        result: dict[str, Any] = {
            "input_ids": torch.tensor(seq, dtype=torch.int32).unsqueeze(0),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.int32).unsqueeze(0),
            "logprobs": torch.tensor(logprobs, dtype=torch.float32).unsqueeze(0),
            "versions": torch.tensor(versions, dtype=torch.int32).unsqueeze(0),
            "attention_mask": torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
            "rewards": torch.tensor(reward, dtype=torch.float32).unsqueeze(0),
        }
        if multi_modal_input:
            result["multi_modal_input"] = multi_modal_input
        return result
