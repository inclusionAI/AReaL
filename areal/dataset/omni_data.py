"""Omni multimodal RL dataset loader for interleaved text, image, audio, and video.

Follows the same conventions as ``geometry3k.py`` and ``clevr_count_70k.py``:
RL datasets return ``messages``, ``messages_chat``, and raw media objects
(``images``, ``audios``, ``videos``) for workflow-level processing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from datasets import Dataset, load_dataset
from PIL.Image import Image as ImageObject

from areal.dataset.audio import DEFAULT_SAMPLING_RATE, load_audio


def _build_messages_chat(
    text: str,
    num_images: int = 0,
    num_videos: int = 0,
    num_audios: int = 0,
) -> list[dict[str, Any]]:
    """Build OpenAI-style chat messages with multimodal content blocks.

    Image/video/audio URL placeholders are left empty -- the inference engine
    fills them with base64 data at request time.
    """
    content: list[dict[str, Any]] = []
    for _ in range(num_images):
        content.append({"type": "image_url", "image_url": {"url": ""}})
    for _ in range(num_videos):
        content.append({"type": "video_url", "video_url": {"url": ""}})
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
    max_video_frames: int = 32,
    video_fps: float | None = None,
    use_audio_in_video: bool = False,
    **kwargs,
) -> Dataset:
    """Load an Omni multimodal RL dataset.

    The dataset is expected to have at least the following columns:

    * ``messages`` or ``problem`` -- the text prompt
    * ``answer`` -- the ground-truth answer (used by reward functions)

    And optionally:

    * ``images`` -- list of PIL images or image paths
    * ``audios`` or ``audio`` -- list of audio arrays / paths, or a single one
    * ``videos`` or ``video`` -- list of video file paths

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
    max_video_frames:
        Maximum number of frames to sample per video.
    video_fps:
        If set, sample video at this frame rate (capped at *max_video_frames*).
    use_audio_in_video:
        If ``True``, extract audio tracks from video files and include them
        as additional audio inputs.
    """
    dataset = load_dataset(path=path, split=split, **kwargs)

    tokenizer = processor.tokenizer if processor is not None else None

    def process(sample: dict[str, Any]) -> dict[str, Any]:
        # --- text ---
        text = sample.get("messages") or sample.get("problem", "")
        if isinstance(text, list):
            text = text[0] if len(text) == 1 else str(text)

        # --- images: store as base64 strings for RPC-safe transport ---
        images: list[str] | None = None
        raw_images = sample.get("images") or sample.get("image")
        if raw_images is not None:
            if not isinstance(raw_images, list):
                raw_images = [raw_images]
            from areal.utils.image import image2base64

            images = image2base64(raw_images)

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

        # --- videos ---
        videos: list[list[ImageObject]] | None = None
        video_paths: list[str] | None = None
        raw_videos = sample.get("videos") or sample.get("video")
        if raw_videos is not None:
            if not isinstance(raw_videos, list):
                raw_videos = [raw_videos]
            from areal.dataset.video import (
                extract_audio_from_video,
                load_video_frames,
            )

            loaded_videos: list[list[ImageObject]] = []
            paths: list[str] = []
            for v in raw_videos:
                if isinstance(v, str):
                    loaded_videos.append(
                        load_video_frames(v, max_frames=max_video_frames, fps=video_fps)
                    )
                    paths.append(v)
                    if use_audio_in_video:
                        video_audio = extract_audio_from_video(v, sr=audio_sr)
                        if audios is None:
                            audios = []
                        audios.append(video_audio)
                elif isinstance(v, list):
                    loaded_videos.append(v)
            videos = loaded_videos if loaded_videos else None
            video_paths = paths if paths else None

        # --- chat-template formatted prompt ---
        num_images = len(images) if images else 0
        num_videos = len(videos) if videos else 0
        num_audios = len(audios) if audios else 0

        plain_text = text
        for tag in ("<image>", "<audio>", "<video>"):
            if isinstance(plain_text, str) and tag in plain_text:
                plain_text = plain_text.replace(tag, "")

        messages_chat = _build_messages_chat(
            plain_text.strip(),
            num_images=num_images,
            num_videos=num_videos,
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
            "answer": sample.get("answer") or sample.get("original_answer", ""),
        }
        if images is not None:
            result["images"] = images
        if audios is not None:
            result["audios"] = audios
        if videos is not None:
            result["videos"] = videos
        if video_paths is not None:
            result["video_paths"] = video_paths

        return result

    cols_to_remove = [
        c
        for c in dataset.column_names
        if c not in ("messages", "messages_chat", "answer", "images", "audios", "videos", "video_paths")
    ]
    dataset = dataset.map(process, remove_columns=cols_to_remove)

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
            if sample.get("videos"):
                proc_kwargs["videos"] = sample["videos"]
            if sample.get("audios"):
                proc_kwargs["audios"] = sample["audios"]
            processed = processor(**proc_kwargs)
            total_tokens = len(processed["input_ids"].squeeze(0))
            return total_tokens <= max_length

        dataset = dataset.filter(filter_length)

    return dataset
