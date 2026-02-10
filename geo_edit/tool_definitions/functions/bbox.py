"""Bounding Box Tool."""

from PIL import Image, ImageDraw

DECLARATION = {
    "name": "bounding_box",
    "description": """
    Calling an bounding box adding tool with existing image index (e.g. 0 from 'Observation 0', 1 from 'Observation 1') to add a bounding box as per the given bounding box coordinates in the image. Returns the image with the bounding box added.
    The bounding box should be provided in the format "\\boxed{x1,y1,x2,y2}" where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner of the box. All (x,y) should be larger than or equal to 0 and smaller than or equal to 1000 as a unified image size (1000x1000).
    If you call this functions multiple times in one action, all bounding boxes will be added to the select image and only the final image with all bounding boxes will be returned.
    For example, to add a bounding box to a specific area in the image Observation 0 , you can provide the bounding box coordinates like "\\boxed{190,840,200,200}" along with the image index 0.
    """,
    "parameters": {
        "type": "object",
        "properties": {
            "image_index": {
                "type": "integer",
                "description": "The index of the image to extract the bounding box from. Each image is assigned an index when uploaded.Like 'Observation 0', 'Observation 1', etc.",
            },
            "bounding_box": {"type": "string", "description": "Relative bounding box coordinates to crop the image."},
        },
        "required": ["image_index", "bounding_box"],
    },
}

RETURN_TYPE = "image"


def execute(image_list, image_index: int, bounding_box: str) -> str | Image.Image:
    if image_index < 0 or image_index >= len(image_list):
        return "Error: Invalid image index."
    image_to_box = image_list[image_index]
    draw = ImageDraw.Draw(image_to_box)
    coords = bounding_box.strip("\\boxed{}").split(",")
    width, height = image_to_box.size
    x1, y1, x2, y2 = [int(int(c) * width / 1000) if i % 2 == 0 else int(int(c) * height / 1000) for i, c in enumerate(coords)]
    draw.rectangle((x1, y1, x2, y2), outline="green", width=3)
    return image_to_box
