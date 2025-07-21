import base64
import math
from dataclasses import MISSING
from io import BytesIO
from typing import List

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



