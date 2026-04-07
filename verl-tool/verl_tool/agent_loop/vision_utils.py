import io
import base64
import numpy as np
import torch
from io import BytesIO
from typing import Union, Optional
from PIL import Image
from .vision_process import fetch_image, fetch_video
from verl.utils.dataset.vision_utils import VIDEO_FORMAT_HELP


def process_image(image: dict | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if "bytes" in image:
        assert "image" not in image, "Cannot have both `bytes` and `image`"
        image["image"] = BytesIO(image["bytes"])

    return fetch_image(image)

def process_video(
    video: dict,
    nframes: Optional[int] = None,
    fps: Optional[float] = None,
    fps_min_frames: Optional[int] = None,
    fps_max_frames: Optional[int] = None,
) -> torch.Tensor:
    """Converts a video dict into a [n_frames, 3, H, W] tensor

    Add video sample FPS in a future MR
    """

    if not isinstance(video, dict) or "video" not in video:
        raise NotImplementedError(VIDEO_FORMAT_HELP)
    assert nframes is None or fps is None, "Can't use both `nframes` or `fps`"

    # Shallow copy... since we might want to add some keys
    video = dict(video)

    contains_sampling_rules = "nframes" in video or "fps" in video
    if not contains_sampling_rules:
        if nframes is not None:
            video["nframes"] = nframes
        elif fps is not None:
            video["fps"] = fps
            if fps_min_frames is not None:
                video["min_frames"] = fps_min_frames
            if fps_max_frames is not None:
                video["max_frames"] = fps_max_frames

    return fetch_video(video)

def encode_image(img: Image.Image) -> str:
    if isinstance(img, Image.Image):
        buffered = io.BytesIO()
        # convert the image to RGB if it is not already
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return img_str
    else:
        raise ValueError(f"Unsupported image type: {type(img)}. Expected str or PIL Image, got {type(img)}.")

def decode_image(img_str):
    img_data = base64.b64decode(img_str)
    img = Image.open(io.BytesIO(img_data))
    return img

def decode_image_url(img_url: str) -> Image.Image:
    return process_image({"image": img_url})

def encode_image_url(img: Union[str, dict, Image.Image]) -> str:
    if isinstance(img, str):
        img = process_image({"image": img})
    else:
        img = process_image(img)
    encoded_img = encode_image(img)
    return f"data:image/jpeg;base64,{encoded_img}"  # Assume img is a base64 string or file path

def encode_video_url(
    video: Union[list, str, dict, np.ndarray], 
    nframes: Optional[int] = None,
    fps: Optional[float] = None,
    fps_min_frames: Optional[int] = None,
    fps_max_frames: Optional[int] = None
) -> str:
    if isinstance(video, list):
        if all(isinstance(frame, np.ndarray) for frame in video) or \
        isinstance(video, np.ndarray) and video.ndim == 4:  # Assuming video is a list of numpy arrays or a 4D numpy array
            # load from numpy arrays
            frames = [Image.fromarray(frame) for frame in video]
        else:
            frames = [process_image({"image": frame}) for frame in video]
    else:
        if isinstance(video, str):
            video = {"video": video}
        else:
            frames = process_video(video, nframes=nframes, fps=fps, fps_min_frames=fps_min_frames, fps_max_frames=fps_max_frames)
    encoded_frames = [encode_image(frame) for frame in frames]
    return f"data:video/jpeg;base64,{','.join(encoded_frames)}"  # Assume video is a list of processed images

def decode_video_url(video_url: str) -> list:
    if video_url.startswith("data:video/jpeg;base64,"):
        video_data = video_url.split(",")[1]
        frames = [process_image("data:image/jpeg;base64," + frame) for frame in video_data.split(",")]
        return frames
    else:
        return process_video({"video": video_url})