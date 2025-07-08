from datasets import Dataset
from typing import Any, Dict, List, Optional, Tuple, Union,Literal
import math
from PIL import Image
from PIL.Image import Image as ImageObject
from io import BytesIO
import base64
def process_image(
    image: Union[Dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    '''
    Process an image to ensure it is in RGB format and resized if necessary.
    '''
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image



def process_clevr_count_70k_sft_dataset(dataset: Dataset, processor):
    '''
    "clevr_count_70k": {
        "image_key": "images",
        "question_key": "problem",
        "answer_key": "answer"
    },
    '''
    tokenizer = processor.tokenizer
    image_token=processor.image_token if processor is not None else "<image>"      
    def process_example(example, idx):
        # Add query_id column
        example["query_id"] = str(idx)
        prompt_str = example["problem"].replace("<image>", image_token)
        example["prompt"] = prompt_str

        return example

    dataset = dataset.map(
        lambda example, idx: process_example(example, idx),
        with_indices=True,
        remove_columns=['problem'],
    )

    def _process(example):
        images=example["images"]
        processed_images=[]
        for image in images:
            processed_image=process_image(image,113*113,336*336)
            buffer = BytesIO()
            processed_image.save(buffer, format="PNG")  # 或者根据你的图像格式调整，如 PNG
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
            processed_images.append(img_base64)
        seq=example["prompt"] + example["answer"] + tokenizer.eos_token

        processed_input=processor(images,[seq],add_special_tokens=False, return_tensors="pt",return_length=True,padding=False,truncation=True,return_attention_mask=False)

        example["seq"] =processed_input["input_ids"]
        


        
        example["pixel_values"] = processed_input["pixel_values"]
        example["image_grid_thw"] = processed_input["image_grid_thw"]
        example["prompt"] = tokenizer(example["prompt"],add_special_tokens=False, return_tensors="pt",
        return_length=True,padding=False,truncation=True,return_attention_mask=False,)[
            "input_ids"
        ]
        return example

    dataset = dataset.map(lambda x: _process(x),remove_columns=["images"])
    return dataset