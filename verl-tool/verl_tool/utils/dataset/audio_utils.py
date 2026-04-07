from io import BytesIO
from typing import Optional

import torch
import numpy
from PIL import Image
from qwen_omni_utils import process_audio_info, process_mm_info

import librosa
import base64
from pathlib import Path
import soundfile as sf

SAMPLE_RATE = 16000

def process_audio(audio: dict) -> numpy.ndarray:
    assert ("audio" in audio or "audio_url" in audio), "audio is not found in the audio dictionary"
    
    audio["type"] = "audio"
    dummy_content = [audio]
    dummy_message = {"role": "user", "content": dummy_content}
    dummy_conversation = [dummy_message]
    return process_audio_info(dummy_conversation, use_audio_in_video=False)[0]

def encode_audio_file(audio_path: str, mime_subtype: str = "wav") -> str:
    with Path(audio_path).open("rb") as file_obj:
        encoded = base64.b64encode(file_obj.read()).decode("ascii")
    return f"data:audio/{mime_subtype};base64,{encoded}"

def encode_audio_data(audio_data: numpy.ndarray, sample_rate: int = SAMPLE_RATE, mime_subtype: str = "wav") -> str:
    """Encodes audio data from a numpy array into a base64 data URL."""
    with BytesIO() as buf:
        sf.write(buf, audio_data, sample_rate, format=mime_subtype)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("ascii")
    return f"data:audio/{mime_subtype};base64,{encoded}"

def decode_audio_url(audio_url: str) -> str:
    return process_audio({"audio_url": audio_url})
