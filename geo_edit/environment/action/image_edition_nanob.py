from google import genai
from google.genai import types
from PIL import Image
import io
from geo_edit.constants import API_KEY

system_prompt='''
You are an image editing AI model. Your task is to edit images based on the user's instructions. You will receive an image along with specific editing instructions, and you need to apply those edits to the image accurately.
'''

image_edit_function_declaration = {
    "name":"image_edition",
    "description":'''
    Calling an image editing tool with proper prompt and existing image index (e.g. 0 from 'Observation 0', 1 from 'Observation 1') to edit the image as instructions. Returns the edited image.
    For example, to remove an object from the image, you can provide instructions like 'Remove the red car from the image' along with the appropriate image index; to change colors, you can say 'Change the sky to a sunset orange' etc.
    REMEMBER that this function can ONLY edit images without any reasoning or calculations, so please provide clear and concise instructions on how to edit the image.
    ''',
    "parameters":{
        "type":"object",
        "properties":{
            "image_index":{
                "type":"integer",
                "description":"The index of the image to be edited. Each image is assigned an index when uploaded.Like 'Observation 0', 'Observation 1', etc."
            },
            "prompt":{
                "type":"string",
                "description":"Instructions on how to edit the image."
            }
        },
        "required":["image_index", "prompt"]
    }
}


def image_edition_function(image_list, image_index: int, prompt: str) -> str | Image.Image:

    client = genai.Client(api_key=API_KEY)
    image_to_edit = image_list[image_index]
    
    response = client.models.generate_content(
        model="gemini-3-pro-image-preview",
        contents=[
            f"Please edit the image as per the following instructions: {prompt}",
            image_to_edit
        ],
    )
    captured_text = ""
    for part in response.candidates[0].content.parts:
            if part.inline_data:
                print("Image data found in response parts.")
                image_bytes = part.inline_data.data
                img = Image.open(io.BytesIO(image_bytes))
                return img 
            
            if part.text:
                captured_text += part.text

    print("No image found, returning text.")
    return captured_text 
            

            
