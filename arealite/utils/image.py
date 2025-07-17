from io import BytesIO
import base64
import math
from torch import Tensor
from typing import Any, Dict, List, Optional, Union
from PIL.Image import Image as ImageObject
from dataclasses import MISSING

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



