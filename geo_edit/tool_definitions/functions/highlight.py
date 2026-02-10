"""Image Highlight Tool."""

from PIL import Image, ImageDraw

DECLARATION = {
    "name": "image_highlight",
    "description": """
    Calling an image highlighting tool with existing image index (e.g. 0 from 'Observation 0', 1 from 'Observation 1') to highlight a specific area on the image as per the given bounding box coordinates. Returns the highlighted image whose highlighted areas are yellow marked.
    The bounding box should be provided in the format "\\boxed{x1,y1,x2,y2}" where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner of the box. All (x,y) should be larger than or equal to 0 and smaller than or equal to 1000 as a unified image size (1000x1000).
    If you call this functions multiple times in one action, all highlighted areas will be added to the select image and only the final highlighted image will be returned.
    For example, to highlight a specific area from the image Observation 0, you can provide the bounding box coordinates like "\\boxed{300,400,500,600}" along with the image index 0.
    """,
    "parameters": {
        "type": "object",
        "properties": {
            "image_index": {
                "type": "integer",
                "description": "The index of the image to be highlighted. Each image is assigned an index when uploaded.Like 'Observation 0', 'Observation 1', etc.",
            },
            "bounding_box": {"type": "string", "description": "Relative bounding box coordinates to highlight the area on the image."},
        },
        "required": ["image_index", "bounding_box"],
    },
}

RETURN_TYPE = "image"


def execute(image_list, image_index: int, bounding_box: str) -> str | Image.Image:
    if image_index < 0 or image_index >= len(image_list):
        return "Error: Invalid image index."
    image_to_highlight = image_list[image_index].convert("RGBA")
    overlay = Image.new("RGBA", image_to_highlight.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    coords = bounding_box.strip("\\boxed{}").split(",")
    width, height = image_to_highlight.size
    x1, y1, x2, y2 = [int(int(c) * width / 1000) if i % 2 == 0 else int(int(c) * height / 1000) for i, c in enumerate(coords)]
    draw.rectangle((x1, y1, x2, y2), fill=(255, 255, 0, 100))
    highlighted_image = Image.alpha_composite(image_to_highlight, overlay).convert("RGB")
    return highlighted_image
