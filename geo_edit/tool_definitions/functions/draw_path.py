"""Draw Path Tool."""

from PIL import Image, ImageDraw

DECLARATION = {
    "name": "draw_path",
    "description": """
    Calling an image drawing tool with existing image index (e.g. 0 from 'Observation 0', 1 from 'Observation 1') to draw a connected path (multiple line segments) on the image. Returns the modified image.
    The points should be provided as a series of (x,y) coordinates in the format "\\boxed{x1,y1,x2,y2,x3,y3,...}" where each pair (xi, yi) is a waypoint. The tool draws line segments connecting consecutive points: (x1,y1)->(x2,y2)->(x3,y3)->... All (x,y) should be larger than or equal to 0 and smaller than or equal to 1000 as a unified image size (1000x1000). At least two points (4 values) are required.
    For example, to draw a path from (50,50) through (200,50) to (200,200) on Observation 0, provide "\\boxed{50,50,200,50,200,200}" along with image index 0. This is useful for tracing routes on maps, drawing maze solutions, or marking multi-step paths.
    """,
    "parameters": {
        "type": "object",
        "properties": {
            "image_index": {
                "type": "integer",
                "description": "The index of the image to draw the path on. Each image is assigned an index when uploaded. Like 'Observation 0', 'Observation 1', etc.",
            },
            "points": {
                "type": "string",
                "description": "Relative coordinates of waypoints to draw the path, e.g. '\\boxed{x1,y1,x2,y2,x3,y3,...}'.",
            },
        },
        "required": ["image_index", "points"],
    },
}

RETURN_TYPE = "image"


def execute(image_list, image_index: int, points: str) -> str | Image.Image:
    if image_index < 0 or image_index >= len(image_list):
        return "Error: Invalid image index."

    raw = points.strip("\\boxed{}").split(",")
    if len(raw) < 4 or len(raw) % 2 != 0:
        return "Error: Need at least 2 points (4 coordinate values) and an even number of values."

    image_to_draw = image_list[image_index]
    draw = ImageDraw.Draw(image_to_draw)
    width, height = image_to_draw.size

    pixel_coords = []
    for i in range(0, len(raw), 2):
        x = int(int(raw[i]) * width / 1000)
        y = int(int(raw[i + 1]) * height / 1000)
        pixel_coords.append((x, y))

    for i in range(len(pixel_coords) - 1):
        draw.line([pixel_coords[i], pixel_coords[i + 1]], fill="blue", width=3)

    # Draw small circles at each waypoint for visibility
    r = 4
    for px, py in pixel_coords:
        draw.ellipse((px - r, py - r, px + r, py + r), fill="red", outline="red")

    return image_to_draw
