"""Draw Line Tool."""

from PIL import Image, ImageDraw

DECLARATION = {
    "name": "draw_line",
    "description": """
    Calling an image drawing tool with existing image index (e.g. 0 from 'Observation 0', 1 from 'Observation 1') to draw a line on the image as per the given start and end coordinates. Returns the modified image.
    The coordinates should be in the format "\\boxed{x1,y1,x2,y2}" where (x1, y1) is the start point and (x2, y2) is the end point of the line. All (x,y) should be larger than or equal to 0 and smaller than or equal to 1000 as a unified image size (1000x1000). Only two points are allowed; if you want to draw multiple lines, please call this function multiple times.
    If you call this functions multiple times in one action, all lines will be added to the select image and only the final modified image will be returned.
    For example, to draw a line from point (50,50) to point (200,200) on the image Observation 0, you can provide the coordinates "\\boxed{50,50,200,200}" along with the image index 0. You can use this to highlight features such as roads or crossings in the image.
    """,
    "parameters": {
        "type": "object",
        "properties": {
            "image_index": {
                "type": "integer",
                "description": "The index of the image to draw the line on. Each image is assigned an index when uploaded.Like 'Observation 0', 'Observation 1', etc.",
            },
            "coordinates": {"type": "string", "description": "Relative coordinates to draw the line."},
        },
        "required": ["image_index", "coordinates"],
    },
}

RETURN_TYPE = "image"


def execute(image_list, image_index: int, coordinates: str) -> str | Image.Image:
    if image_index < 0 or image_index >= len(image_list):
        return "Error: Invalid image index."
    image_to_draw = image_list[image_index]
    draw = ImageDraw.Draw(image_to_draw)
    coords = coordinates.strip("\\boxed{}").split(",")
    width, height = image_to_draw.size
    x1, y1, x2, y2 = [int(int(c) * width / 1000) if i % 2 == 0 else int(int(c) * height / 1000) for i, c in enumerate(coords)]
    draw.line((x1, y1, x2, y2), fill="blue", width=3)
    return image_to_draw
