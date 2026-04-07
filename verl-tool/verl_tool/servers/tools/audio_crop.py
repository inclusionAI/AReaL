"""
Audio crop tool for extracting segments from audio assets.
This version is aligned with the I/O format of pixel_reasoner.py.
"""
import base64
import json
import math
import uuid
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import regex as re
import soundfile as sf

from .base import BaseTool, register_tool

TIMEOUT = 20
VALID_CALL_NAMES = {"audio_crop"}
SAMPLE_RATE = 16000
MIN_CROP_DURATION_SECONDS = 1.0

AUDIO_CROP_FUNCTION_SPEC = {
    "type": "function",
    "function": {
        "name": "audio_crop",
        "description": "Crop the audio based on the time window.",
        "parameters": {
            "type": "object",
            "properties": {
                "time_window": {
                    "type": "array",
                    "description": "A tuple of two numbers indicating the start and end time (in seconds) of the audio segment to crop.",
                    "items": {"type": "number"},
                },
                "target_audio": {
                    "type": "number",
                    "description": "The index of the audio to crop. Index from 1 to the number of audios. Choose 1 to operate on original audio.",
                },
            },
            "required": ["time_window", "target_audio"],
        },
    },
}


def _resolve_audio_source(audio_source: dict | str | Any) -> str:
    """Returns a string audio location or data URI from various representations."""
    if isinstance(audio_source, dict):
        if "audio" in audio_source and isinstance(audio_source["audio"], str):
            return audio_source["audio"]
        if "audio_url" in audio_source and isinstance(audio_source["audio_url"], str):
            return audio_source["audio_url"]
        raise ValueError(f"Unsupported audio source dict structure: {audio_source.keys()}")
    if isinstance(audio_source, str):
        return audio_source
    raise TypeError(f"Unsupported audio source type: {type(audio_source)}")


def decode_audio_url(audio_source: str) -> Tuple[np.ndarray, int]:
    """Decodes an audio source (path or data URI) into a numpy array."""
    assert isinstance(audio_source, str), f"system error: audio_source must be a string, got {type(audio_source)}"
    if audio_source.startswith("data:audio/"):
        # It's a data URI
        header, encoded = audio_source.split(",", 1)
        audio_bytes = base64.b64decode(encoded)
        with io.BytesIO(audio_bytes) as buf:
            data, samplerate = sf.read(buf, always_2d=False)
        return data, samplerate
    else:
        # Assume it's a file path
        if audio_source.startswith("file://"):
            audio_source = audio_source[len("file://") :]
        data, samplerate = sf.read(audio_source, always_2d=False)
        return data, samplerate

def encode_audio_data(audio_data: np.ndarray, sample_rate: int, mime_subtype: str = "wav") -> str:
    """Encodes a numpy array into a base64 data URL."""
    with io.BytesIO() as buf:
        sf.write(buf, audio_data, sample_rate, format=mime_subtype)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("ascii")
    return f"data:audio/{mime_subtype};base64,{encoded}"


@register_tool
class AudioCropTool(BaseTool):
    tool_type = "audio_crop"
    timeout = TIMEOUT
    function_spec = AUDIO_CROP_FUNCTION_SPEC
    stop_tokens = ["</tool_call>"]

    def get_usage_inst(self) -> str:
        return (
            'Crop an audio segment with <tool_call>{"name": "audio_crop", "arguments": '
            '{"target_audio": 1, "time_window": [start_sec, end_sec]}}</tool_call>. '
            'Time is measured in seconds and audio indices are 1-based.'
        )

    def load_env(self, trajectory_id):
        env = super().load_env(trajectory_id)
        # use [] as a placeholder for now,
        # will try to load from extra_field later
        env["audios"] = []
        return env

    def parse_action(self, action: str):
        if not isinstance(action, str):
            return {}, False

        payload: Optional[Dict[str, Any]] = None
        match = re.search(r"<tool_call>(.*?)</tool_call>", action, re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                payload = None
        else:
            try:
                data = json.loads(action)
                if isinstance(data, dict):
                    payload = data
            except json.JSONDecodeError:
                payload = None

        if not isinstance(payload, dict):
            return {}, False

        name = payload.get("name") or payload.get("tool_name")
        if not isinstance(name, str) or name not in VALID_CALL_NAMES:
            return {}, False

        arguments = payload.get("arguments", {})
        if not isinstance(arguments, dict):
            return {}, False

        return {"name": "audio_crop", "arguments": arguments}, True

    def _extract_audios_from_extra_field(self, extra_field: Dict[str, Any]) -> None:
        if isinstance(extra_field, dict) and isinstance(extra_field.get("audios"), list):
            return extra_field["audios"]
        return []

    def _parse_time_window(self, value: Any) -> Tuple[float, float]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("time_window must be a list of two numbers.")
        start_raw, end_raw = value
        try:
            start = float(start_raw)
            end = float(end_raw)
        except (TypeError, ValueError):
            raise ValueError("time_window values must be numeric.")
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError("time_window values must be finite.")
        if end <= start:
            raise ValueError("time_window end must be greater than start.")
        if (end - start) < MIN_CROP_DURATION_SECONDS:
            raise ValueError(
                f"time_window duration {end - start:.2f}s is below the minimum {MIN_CROP_DURATION_SECONDS:.2f}s."
            )
        return start, end

    def _parse_target_audio(self, value: Any, max_index: int) -> int:
        try:
            idx = int(value)
        except (TypeError, ValueError):
            raise ValueError("target_audio must be an integer.")
        if idx <= 0 or idx > max_index:
            raise ValueError(f"target_audio must be between 1 and {max_index}.")
        return idx

    def conduct_action(self, trajectory_id, action, extra_field):
        parsed_action, is_valid = self.parse_action(action)
        env = self.load_env(trajectory_id)
        extra_field = extra_field or {}

        if len(env["audios"]) == 0:
            # try to extract audios from extra_field
            env["audios"] = self._extract_audios_from_extra_field(extra_field)

        if not is_valid:
            observation = "Invalid audio crop request."
            done = False
            valid = False
            self.update_env(trajectory_id, env, parsed_action, is_valid, extra_field, observation)
            self.save_env(trajectory_id, env)
            return observation, done, valid

        if not env["audios"]:
            observation = "No audio sources supplied for cropping."
            done = False
            valid = False
            self.update_env(trajectory_id, env, parsed_action, False, extra_field, observation)
            self.save_env(trajectory_id, env)
            return observation, done, valid

        arguments = parsed_action.get("arguments", {})
        try:
            time_window = self._parse_time_window(arguments.get("time_window"))
            target_audio_idx = self._parse_target_audio(arguments.get("target_audio"), len(env["audios"]))
        except ValueError as exc:
            observation = f"Audio crop arguments error: {exc}"
            done = False
            valid = False
            self.update_env(trajectory_id, env, parsed_action, False, extra_field, observation)
            self.save_env(trajectory_id, env)
            return observation, done, valid

        try:
            # 1. Select and decode the target audio
            selected_audio_source: dict = env["audios"][target_audio_idx - 1]
            audio_source = _resolve_audio_source(selected_audio_source)
            audio_data, sample_rate = decode_audio_url(audio_source)

            audio_array = np.asarray(audio_data)
            if audio_array.ndim == 1:
                audio_array = audio_array[:, np.newaxis]  # (samples, 1)
            elif audio_array.ndim == 2:
                # Ensure time dimension is first: (samples, channels)
                if audio_array.shape[0] < audio_array.shape[1]:
                    audio_array = audio_array.T
            else:
                raise ValueError(f"Unsupported audio data shape: {audio_array.shape}")

            total_samples = audio_array.shape[0]

            # 2. Perform the crop in-memory
            start_sec, end_sec = time_window
            start_frame = int(np.round(start_sec * sample_rate))
            end_frame = int(np.round(end_sec * sample_rate))

            # Clamp the frames to be within the audio's bounds
            start_frame_clamped = max(0, min(start_frame, total_samples))
            end_frame_clamped = max(0, min(end_frame, total_samples))

            if end_frame_clamped <= start_frame_clamped:
                raise ValueError("Cropped segment duration is zero or negative.")

            cropped_data = audio_array[start_frame_clamped:end_frame_clamped]
            if cropped_data.size == 0:
                raise ValueError("Cropped segment is empty after slicing.")

            # Preserve mono shape for encoding
            if cropped_data.ndim == 2 and cropped_data.shape[1] == 1:
                cropped_to_encode = cropped_data[:, 0]
            else:
                cropped_to_encode = cropped_data

            actual_start = start_frame_clamped / float(sample_rate)
            actual_end = end_frame_clamped / float(sample_rate)
            actual_duration = cropped_data.shape[0] / float(sample_rate)
            if actual_duration < MIN_CROP_DURATION_SECONDS:
                raise ValueError(
                    f"Cropped audio is only {actual_duration:.2f}s, shorter than the minimum "
                    f"{MIN_CROP_DURATION_SECONDS:.2f}s required. Please request a longer time window."
                )

            # 3. Encode the cropped data back to a data URL
            new_data_uri = encode_audio_data(cropped_to_encode, sample_rate)
            
            # 4. Append the new audio to the environment
            env["audios"].append(new_data_uri)
            new_index = len(env["audios"])

            # 5. Create the observation
            obs_text = (
                f"<audio>This is the audio clip ({actual_start:.2f}s - {actual_end:.2f}s) from the original audio #{target_audio_idx} according to your request."
            )
            observation = {
                "obs": obs_text,
                "audio": new_data_uri,
            }
            done = False
            valid = True
        except Exception as exc:
            observation = f"Audio crop failed: {exc}"
            done = False
            valid = False

        self.update_env(trajectory_id, env, parsed_action, valid, extra_field, observation)
        self.save_env(trajectory_id, env)
        return observation, done, valid
