"""Omni multimodal RL dataset loader for interleaved text, image, and audio inputs.

Follows the same conventions as ``geometry3k.py`` and ``clevr_count_70k.py``:
RL datasets return ``messages``, ``messages_chat``, and raw media objects
(``images``, ``audios``) for workflow-level processing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from datasets import Dataset, load_dataset
from PIL.Image import Image as ImageObject

from areal.dataset.audio import DEFAULT_SAMPLING_RATE, load_audio


def _build_messages_chat(
    text: str,
    has_images: bool,
    has_audios: bool,
    num_images: int = 0,
    num_audios: int = 0,
) -> list[dict[str, Any]]:
    """Build OpenAI-style chat messages with multimodal content blocks.

    Image/audio URL placeholders are left empty -- the inference engine
    fills them with base64 data at request time.
    """
    content: list[dict[str, Any]] = []
    for _ in range(num_images):
        content.append({"type": "image_url", "image_url": {"url": ""}})
    for _ in range(num_audios):
        content.append({"type": "audio_url", "audio_url": {"url": ""}})
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def get_omni_rl_dataset(
    path: str,
    split: str | None = None,
    processor=None,
    max_length: int | None = None,
    audio_sr: int = DEFAULT_SAMPLING_RATE,
    **kwargs,
) -> Dataset:
    """Load an Omni multimodal RL dataset.

    The dataset is expected to have at least the following columns:

    * ``messages`` or ``problem`` -- the text prompt
    * ``answer`` -- the ground-truth answer (used by reward functions)

    And optionally:

    * ``images`` -- list of PIL images or image paths
    * ``audios`` or ``audio`` -- list of audio arrays / paths, or a single one

    Parameters
    ----------
    path:
        HuggingFace dataset name or local path.
    split:
        Dataset split (``train``, ``test``, etc.).
    processor:
        ``AutoProcessor`` from the Omni model.  Used for tokenisation and
        for determining special tokens.
    max_length:
        If set, samples whose tokenised length exceeds this are filtered out.
    audio_sr:
        Target sampling rate for audio files loaded from paths.
    """
    dataset = load_dataset(path=path, split=split, **kwargs)

    tokenizer = processor.tokenizer if processor is not None else None

    def process(sample: dict[str, Any]) -> dict[str, Any]:
        # --- text ---
        text = sample.get("messages") or sample.get("problem", "")
        if isinstance(text, list):
            text = text[0] if len(text) == 1 else str(text)

        # --- images ---
        images: list[ImageObject] | None = None
        if "images" in sample and sample["images"]:
            raw_images = sample["images"]
            if not isinstance(raw_images, list):
                raw_images = [raw_images]
            images = raw_images

        # --- audios ---
        audios: list[np.ndarray] | None = None
        raw_audios = sample.get("audios") or sample.get("audio")
        if raw_audios is not None:
            if not isinstance(raw_audios, list):
                raw_audios = [raw_audios]
            loaded: list[np.ndarray] = []
            for a in raw_audios:
                if isinstance(a, str | bytes):
                    loaded.append(load_audio(a, sr=audio_sr))
                elif isinstance(a, np.ndarray):
                    loaded.append(a.astype(np.float32))
                else:
                    loaded.append(np.asarray(a, dtype=np.float32))
            audios = loaded if loaded else None

        # --- chat-template formatted prompt ---
        num_images = len(images) if images else 0
        num_audios = len(audios) if audios else 0

        plain_text = text
        if isinstance(text, str) and "<image>" in text:
            plain_text = text.replace("<image>", "")
        if isinstance(plain_text, str) and "<audio>" in plain_text:
            plain_text = plain_text.replace("<audio>", "")

        messages_chat = _build_messages_chat(
            plain_text.strip(),
            has_images=num_images > 0,
            has_audios=num_audios > 0,
            num_images=num_images,
            num_audios=num_audios,
        )

        # Apply chat template to get the tokeniser-formatted prompt
        if tokenizer is not None:
            formatted_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True,
                tokenize=False,
            )
        else:
            formatted_text = text

        result: dict[str, Any] = {
            "messages": formatted_text,
            "messages_chat": messages_chat,
            "answer": sample.get("answer", ""),
        }
        if images is not None:
            result["images"] = images
        if audios is not None:
            result["audios"] = audios

        return result

    dataset = dataset.map(process)

    if max_length is not None and processor is not None:

        def filter_length(sample: dict[str, Any]) -> bool:
            proc_kwargs: dict[str, Any] = {
                "text": [sample["messages"]],
                "padding": False,
                "return_tensors": "pt",
                "return_length": True,
                "return_attention_mask": False,
            }
            if sample.get("images"):
                proc_kwargs["images"] = sample["images"]
            if sample.get("audios"):
                proc_kwargs["audios"] = sample["audios"]
            processed = processor(**proc_kwargs)
            total_tokens = len(processed["input_ids"].squeeze(0))
            return total_tokens <= max_length

        dataset = dataset.filter(filter_length)

    return dataset
