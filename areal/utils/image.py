import base64
from io import BytesIO
from typing import List, Optional

from PIL import Image
from PIL.Image import Image as ImageObject

from transformers import AutoProcessor, PreTrainedTokenizerFast


def image2base64(images: List[ImageObject] | ImageObject) -> List[str] | str:

    if isinstance(images, ImageObject):
        images = [images]

    byte_images = []
    for image in images:
        with BytesIO() as buffer:
            image.save(buffer, format="PNG")
            buffer.seek(0)
            byte_image = base64.b64encode(buffer.read()).decode("utf-8")
            byte_images.append(byte_image)

    return byte_images


def pad_images_batch_to_max_size(images):
    max_width = max(image.size[0] for image in images)
    max_height = max(image.size[1] for image in images)

    padded_images = []

    for image in images:

        width, height = image.size

        padding_left = (max_width - width) // 2
        padding_top = (max_height - height) // 2

        padded_image = Image.new("RGB", (max_width, max_height), (0, 0, 0))
        padded_image.paste(image, (padding_left, padding_top))

        padded_images.append(padded_image)

    return padded_images


def get_multimodal_input_ids_len(text:str,tokenizer:PreTrainedTokenizerFast=None,images: List[ImageObject] | ImageObject=None,processor:AutoProcessor=None):
    if tokenizer is None and processor is None:
        raise ValueError("Either tokenizer or processor must be provided.")
    if images is None:
        return len(tokenizer(text,add_special_tokens=False)['input_ids'])
    
    if isinstance(images, ImageObject):
        images = [images]

    text_inputs = [text] * len(images)
    inputs = processor(images=images, text=text_inputs, return_tensors="pt", padding=False, add_special_tokens=False)
    return inputs['input_ids'].shape[1]


def _is_data_url(url: str) -> bool:
    return isinstance(url, str) and url.startswith("data:") and ";base64," in url


def _decode_data_url_to_image(url: str) -> ImageObject:
    header, b64data = url.split(",", 1)
    data = base64.b64decode(b64data)
    with BytesIO(data) as bio:
        img = Image.open(bio)
        return img.convert("RGB")


def load_image(source: str) -> ImageObject:
    """Load an image from a path, http(s) URL, or data URL.

    - data URL: data:image/png;base64,....
    - http(s): fetched via requests
    - local path: opened via PIL
    """
    if _is_data_url(source):
        return _decode_data_url_to_image(source)

    if isinstance(source, str) and (source.startswith("http://") or source.startswith("https://")):
        try:
            import requests  # lazy import

            resp = requests.get(source, timeout=10)
            resp.raise_for_status()
            with BytesIO(resp.content) as bio:
                img = Image.open(bio)
                return img.convert("RGB")
        except Exception:
            # Fall back to returning a black 1x1 image to avoid breaking pipeline
            return Image.new("RGB", (1, 1), (0, 0, 0))

    # Assume local file path
    img = Image.open(source)
    return img.convert("RGB")


def get_image_token(processor: Optional[AutoProcessor]) -> str:
    """Best-effort to retrieve the image placeholder token for a processor.

    Falls back to "<image>" when not available.
    """
    token = None
    if processor is not None:
        # Common attributes seen in different processors
        for attr in ("image_token", "boi_token"):
            if hasattr(processor, attr):
                token = getattr(processor, attr)
                break
        # Some implementations stuff the token on nested objects
        if token is None and hasattr(processor, "tokenizer"):
            tok = getattr(processor, "tokenizer")
            for attr in ("image_token", "boi_token"):
                if hasattr(tok, attr):
                    token = getattr(tok, attr)
                    break
    return token or "<image>"
