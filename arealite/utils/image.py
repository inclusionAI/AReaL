from io import BytesIO
import base64
import math
from torch import Tensor
from typing import Any, Dict, List, Optional, Union
from PIL.Image import Image as ImageObject


def image2base64(images: List[ImageObject]|ImageObject)-> List[str]|str:

    if isinstance(images, ImageObject):
        images = [images]
    
    byte_images = []
    for image in images:
        with BytesIO() as buffer:
            image.save(buffer, format="PNG")
            buffer.seek(0) 
            byte_image = base64.b64encode(buffer.read()).decode('utf-8')
            byte_images.append(byte_image)
    
    return byte_images

def convert_image(
    image: Union[Dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")
    return image

def process_image(
    images: List[Union[ImageObject, Tensor]],
    processor: Any,
) -> Dict[str, Tensor]:

    if isinstance(images, Tensor):
        images = [images]

    images=processor.image_processor(images=images, return_tensors="pt")
    pixel_values = images["pixel_values"]
    image_grid_thw = images["image_grid_thw"]
    return {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}