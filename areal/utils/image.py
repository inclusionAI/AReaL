import base64
from io import BytesIO
from typing import List

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
