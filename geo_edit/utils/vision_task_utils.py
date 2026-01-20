import base64
import io
from typing import Final

from PIL import Image

def image_to_data_url(image: Image.Image, *, image_format: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    fmt = image_format.strip().upper()
    if fmt in {"JPG", "JPEG"}:
        mime ="image/jpeg"
    elif fmt == "PNG":
        mime = "image/png"
    return f"data:{mime};base64,{encoded}"
