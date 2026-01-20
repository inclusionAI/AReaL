from .image_edition_nanob import image_edition_function, image_edit_function_declaration
from .image_edition_tool import (
    image_label_function_declaration,
    draw_line_function_declaration,
    image_crop_function_declaration,
    bounding_box_function_declaration,
    image_crop_function,
    image_label_function,
    draw_line_function,
    bounding_box_function,
)

TOOL_FUNCTIONS_DECLARE = [
    # image_edit_function_declaration,
    # image_crop_function_declaration,
    image_label_function_declaration,
    draw_line_function_declaration,
    bounding_box_function_declaration,
]
TOOL_FUNCTIONS={
    # "image_edition":image_edition_function,
    # "image_crop":image_crop_function,
    "image_label":image_label_function,
    "draw_line":draw_line_function,
    "bounding_box":bounding_box_function,
}

