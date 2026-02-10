"""Image Label Tool."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_FONT_PATH = Path(__file__).parent.parent.parent / "assets" / "fonts" / "arial.ttf"

DECLARATION = {
    "name": "image_label",
    "description": """
    Calling an image labeling tool with existing image index (e.g. 0 from 'Observation 0', 1 from 'Observation 1'), text and position (x,y) to label the image. All (x,y) should be larger than or equal to 0 and smaller than or equal to 1000 as a unified image size (1000x1000).
    Returns the labeled image.
    If you call this functions multiple times in one action, all labels will be added to the select image and only the final labeled image will be returned.
    For example, to label a specific area in the image Observation 0 with the text "Tree" at position (100,150), you can provide the image index 0, text "Tree", and position "(100,150)". You can use this to annotate features such as buildings or landmarks in the image.
    """,
    "parameters": {
        "type": "object",
        "properties": {
            "image_index": {
                "type": "integer",
                "description": "The index of the image to be labeled. Each image is assigned an index when uploaded.Like 'Observation 0', 'Observation 1', etc.",
            },
            "text": {"type": "string", "description": "Text to label on the image."},
            "position": {"type": "string", "description": "Relative Position (x,y) to place the label on the image."},
        },
        "required": ["image_index", "text", "position"],
    },
}

RETURN_TYPE = "image"


def execute(image_list, image_index: int, text: str | list, position: str) -> str | Image.Image:
    if image_index < 0 or image_index >= len(image_list):
        return "Error: Invalid image index."
    image_to_label = image_list[image_index]
    draw = ImageDraw.Draw(image_to_label)
    width, height = image_to_label.size
    coords = position.strip("()").split(",")
    x, y = int(int(coords[0]) * width / 1000), int(int(coords[1]) * height / 1000)
    font = ImageFont.truetype(str(_FONT_PATH), 30)
    draw.text((x, y), text, fill="red", font=font)
    return image_to_label
