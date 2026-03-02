"""Video loading, frame sampling, audio extraction, and encoding utilities.

Provides helpers to load video files, sample frames, optionally extract the
audio track, and encode videos as base64 strings for remote inference engines
(vLLM-Omni).
"""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
from PIL.Image import Image as ImageObject

from areal.dataset.audio import DEFAULT_SAMPLING_RATE


def load_video_frames(
    path: str | Path,
    max_frames: int = 32,
    fps: float | None = None,
) -> list[ImageObject]:
    """Load a video file and sample frames as PIL Images.

    Parameters
    ----------
    path:
        Path to the video file (mp4, webm, mkv, etc.).
    max_frames:
        Maximum number of frames to sample.
    fps:
        If set, sample at this frame rate (capped at *max_frames*).
        If ``None``, uniformly sample *max_frames* frames.

    Returns
    -------
    list[ImageObject]
        Sampled video frames as PIL Images.
    """
    try:
        from decord import VideoReader, cpu
    except ImportError as exc:
        raise ImportError(
            "decord is required for video loading. Install it with: pip install decord"
        ) from exc

    from PIL import Image

    vr = VideoReader(str(path), ctx=cpu(0))
    total_frames = len(vr)

    if fps is not None:
        video_fps = vr.get_avg_fps()
        interval = max(1, int(video_fps / fps))
        indices = list(range(0, total_frames, interval))[:max_frames]
    else:
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()

    frames = vr.get_batch(indices).asnumpy()
    return [Image.fromarray(f) for f in frames]


def extract_audio_from_video(
    path: str | Path,
    sr: int = DEFAULT_SAMPLING_RATE,
) -> np.ndarray:
    """Extract the audio track from a video file.

    Uses ``librosa`` which delegates to ffmpeg for demuxing.

    Parameters
    ----------
    path:
        Path to the video file.
    sr:
        Target sampling rate in Hz.

    Returns
    -------
    np.ndarray
        1-D float32 waveform array.
    """
    try:
        import librosa
    except ImportError as exc:
        raise ImportError(
            "librosa is required for audio extraction from video. "
            "Install it with: pip install librosa"
        ) from exc

    audio, _ = librosa.load(str(path), sr=sr, mono=True)
    return audio.astype(np.float32)


def video2base64(path: str | Path) -> str:
    """Read a video file and return its contents as a base64 string.

    vLLM-Omni expects video data as ``data:video/mp4;base64,...`` in the
    ``video_url`` field of chat completion requests.

    Parameters
    ----------
    path:
        Path to the video file.

    Returns
    -------
    str
        Base64-encoded video data.
    """
    with open(str(path), "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def videos2base64(paths: list[str | Path]) -> list[str]:
    """Encode a list of video files as base64 strings."""
    return [video2base64(p) for p in paths]
