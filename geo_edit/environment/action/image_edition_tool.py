from PIL import Image, ImageDraw, ImageFont
image_crop_function_declaration = {
    "name":"image_crop",
    "description":'''
    Calling an image cropping tool with existing image index (e.g. 0 from 'Observation 0', 1 from 'Observation 1') to crop the image as per the given bounding box coordinates. Returns the cropped image.
    The bounding box should be provided in the format "\\boxed{x1,y1,x2,y2}" where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner of the box. All (x,y) should be larger than or equal to 0 and smaller than or equal to 1000 as a unified image size (1000x1000). 
    If you call this functions multiple times in one action, the return order of cropped images will be consistent with the calling order.
    For example, to crop a specific area from the image Observation 0, you can provide the bounding box coordinates like "\\boxed{100,700,200,200}" along with the image index 0.
    ''',
    "parameters":{
        "type":"object",
        "properties":{
            "image_index":{
                "type":"integer",
                "description":"The index of the image to be cropped. Each image is assigned an index when uploaded.Like 'Observation 0', 'Observation 1', etc."
            },
            "bounding_box":{
                "type":"string",
                "description":"Relative bounding box coordinates to crop the image."
            }
        },
        "required":["image_index", "bounding_box"]
    }
}
image_label_function_declaration = {
    "name":"image_label",
    "description":'''
    Calling an image labeling tool with existing image index (e.g. 0 from 'Observation 0', 1 from 'Observation 1'), text and position (x,y) to label the image. All (x,y) should be larger than or equal to 0 and smaller than or equal to 1000 as a unified image size (1000x1000).
    Returns the labeled image.
    If you call this functions multiple times in one action, all labels will be added to the select image and only the final labeled image will be returned.
    For example, to label a specific area in the image Observation 0 with the text "Tree" at position (100,150), you can provide the image index 0, text "Tree", and position "(100,150)".
    ''',
    "parameters":{
        "type":"object",
        "properties":{
            "image_index":{
                "type":"integer",
                "description":"The index of the image to be labeled. Each image is assigned an index when uploaded.Like 'Observation 0', 'Observation 1', etc."
            },
            "text":{
                "type":"string",
                "description":"Text to label on the image."
            },
            "position":{
                "type":"string",
                "description":"Relative Position (x,y) to place the label on the image."
            }
        },
        "required":["image_index", "text", "position"]
    }
}
draw_line_function_declaration = {
    "name":"draw_line",
    "description":'''
    Calling an image drawing tool with existing image index (e.g. 0 from 'Observation 0', 1 from 'Observation 1') to draw a line on the image as per the given start and end coordinates. Returns the modified image.
    The coordinates should be in the format "\\boxed{x1,y1,x2,y2}" where (x1, y1) is the start point and (x2, y2) is the end point of the line. All (x,y) should be larger than or equal to 0 and smaller than or equal to 1000 as a unified image size (1000x1000). Only two points are allowed; if you want to draw multiple lines, please call this function multiple times.
    If you call this functions multiple times in one action, all lines will be added to the select image and only the final modified image will be returned.
    For example, to draw a line from point (50,50) to point (200,200) on the image Observation 0, you can provide the coordinates "\\boxed{50,50,200,200}" along with the image index 0.
    ''',
    "parameters":{
        "type":"object",
        "properties":{
            "image_index":{
                "type":"integer",
                "description":"The index of the image to draw the line on. Each image is assigned an index when uploaded.Like 'Observation 0', 'Observation 1', etc."
            },
            "coordinates":{
                "type":"string",
                "description":"Relative coordinates to draw the line."
            }
        },
        "required":["image_index", "coordinates"]
    },
}

bounding_box_function_declaration = {
    "name":"bounding_box",
    "description":'''
    Calling an bounding box adding tool with existing image index (e.g. 0 from 'Observation 0', 1 from 'Observation 1') to add a bounding box as per the given bounding box coordinates in the image. Returns the image with the bounding box added.
    The bounding box should be provided in the format "\\boxed{x1,y1,x2,y2}" where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner of the box. All (x,y) should be larger than or equal to 0 and smaller than or equal to 1000 as a unified image size (1000x1000). 
    If you call this functions multiple times in one action, all bounding boxes will be added to the select image and only the final image with all bounding boxes will be returned.
    For example, to add a bounding box to a specific area in the image Observation 0 , you can provide the bounding box coordinates like "\\boxed{190,840,200,200}" along with the image index 0. 
    ''',
    "parameters":{
        "type":"object",
        "properties":{
            "image_index":{
                "type":"integer",
                "description":"The index of the image to extract the bounding box from. Each image is assigned an index when uploaded.Like 'Observation 0', 'Observation 1', etc."
            },
            "bounding_box":{
                "type":"string",
                "description":"Relative bounding box coordinates to crop the image."
            }
        },
        "required":["image_index", "bounding_box"]
    },
}

def image_crop_function(image_list, image_index: int, bounding_box: str) -> str | Image.Image:
    # Dummy implementation for illustration
    if image_index < 0 or image_index >= len(image_list):
        return "Error: Invalid image index."
    image_to_crop = image_list[image_index]
    # Parse bounding box
    coords = bounding_box.strip("\\boxed{}").split(",")
    # map from (1000x1000) to actual image size
    width, height = image_to_crop.size
    x1, y1, x2, y2 = [int(int(c) * width / 1000) if i % 2 == 0 else int(int(c) * height / 1000) for i, c in enumerate(coords)]
    cropped_image = image_to_crop.crop((x1, y1, x2, y2))
    return cropped_image

def image_label_function(image_list, image_index: int, text: str| list, position: str) -> str | Image.Image:
    if image_index < 0 or image_index >= len(image_list):
        return "Error: Invalid image index."
    image_to_label = image_list[image_index]
    draw = ImageDraw.Draw(image_to_label)
    width, height = image_to_label.size
    coords = position.strip("()").split(",")
    x, y = int(int(coords[0]) * width / 1000), int(int(coords[1]) * height / 1000)
    # Using a default font
    font = ImageFont.truetype("arial.ttf", 15)
    draw.text((x, y), text, fill="red", font=font)
    
    return image_to_label

def draw_line_function(image_list, image_index: int, coordinates: str) -> str | Image.Image:
    if image_index < 0 or image_index >= len(image_list):
        return "Error: Invalid image index."    
    image_to_draw = image_list[image_index]
    draw = ImageDraw.Draw(image_to_draw)
    coords = coordinates.strip("\\boxed{}").split(",")
    width, height = image_to_draw.size
    x1, y1, x2, y2 = [int(int(c) * width / 1000) if i % 2 == 0 else int(int(c) * height / 1000) for i, c in enumerate(coords)]
    draw.line((x1, y1, x2, y2), fill="blue", width=3)
    
    return image_to_draw

def bounding_box_function(image_list, image_index: int, bounding_box: str) -> str | Image.Image:
    if image_index < 0 or image_index >= len(image_list):
        return "Error: Invalid image index."
    image_to_box = image_list[image_index]
    draw = ImageDraw.Draw(image_to_box)
    coords= bounding_box.strip("\\boxed{}").split(",")
    width, height = image_to_box.size
    x1, y1, x2, y2 = [int(int(c) * width / 1000) if i % 2 == 0 else int(int(c) * height / 1000) for i, c in enumerate(coords)]
    draw.rectangle((x1, y1, x2, y2), outline="green", width=3)
    return image_to_box