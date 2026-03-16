"""Audio loading, resampling, and encoding utilities for Omni multimodal models.

Provides helpers to load audio from file paths, resample to a target rate,
and encode as base64 WAV strings for remote inference engines (vLLM).
"""

import base64
import io
import struct
from pathlib import Path

import numpy as np

DEFAULT_SAMPLING_RATE = 16_000


def load_audio(
    path: str | Path,
    sr: int = DEFAULT_SAMPLING_RATE,
) -> np.ndarray:
    """Load an audio file and resample to *sr* Hz.

    Uses ``librosa`` for format-agnostic loading and high-quality resampling.

    Parameters
    ----------
    path:
        Path to the audio file (wav, mp3, flac, ogg, etc.).
    sr:
        Target sampling rate in Hz.

    Returns
    -------
    np.ndarray
        1-D float32 waveform array normalised to [-1, 1].
    """
    try:
        import librosa
    except ImportError as exc:
        raise ImportError(
            "librosa is required for audio loading. "
            "Install it with: pip install librosa"
        ) from exc

    audio, _ = librosa.load(str(path), sr=sr, mono=True)
    return audio.astype(np.float32)


def _write_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    """Encode a float32 waveform as 16-bit PCM WAV bytes (no external deps)."""
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767).astype(np.int16)
    raw = pcm.tobytes()

    buf = io.BytesIO()
    num_channels = 1
    sample_width = 2  # 16-bit
    data_size = len(raw)
    # RIFF header
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    # fmt chunk
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(
        struct.pack(
            "<HHIIHH",
            1,
            num_channels,
            sr,
            sr * num_channels * sample_width,
            num_channels * sample_width,
            16,
        )
    )
    # data chunk
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(raw)
    return buf.getvalue()


def audio2base64(
    audio: np.ndarray,
    sr: int = DEFAULT_SAMPLING_RATE,
) -> str:
    """Encode a float32 waveform as a base64 WAV string for vLLM requests.

    Parameters
    ----------
    audio:
        1-D float32 waveform.
    sr:
        Sampling rate of *audio*.

    Returns
    -------
    str
        Base64-encoded WAV data.
    """
    wav_bytes = _write_wav_bytes(audio, sr)
    return base64.b64encode(wav_bytes).decode("utf-8")


def audios2base64(
    audios: list[np.ndarray],
    sr: int = DEFAULT_SAMPLING_RATE,
) -> list[str]:
    """Encode a list of waveforms as base64 WAV strings."""
    return [audio2base64(a, sr) for a in audios]
