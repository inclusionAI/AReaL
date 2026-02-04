from geo_edit.environment.action.image_edition_tool import (
    image_label_function_declaration,
    draw_line_function_declaration,
    image_crop_function_declaration,
    bounding_box_function_declaration,
    image_highlight_function_declaration,
    image_crop_function,
    image_label_function,
    draw_line_function,
    bounding_box_function,
    image_highlight_function,
)

TOOL_FUNCTIONS_DECLARE = [
    # image_crop_function_declaration,
    image_label_function_declaration,
    draw_line_function_declaration,
    bounding_box_function_declaration,
    image_highlight_function_declaration,
]
TOOL_FUNCTIONS={
    # "image_crop":image_crop_function,
    "image_label":image_label_function,
    "draw_line":draw_line_function,
    "bounding_box":bounding_box_function,
    "image_highlight":image_highlight_function,
}
