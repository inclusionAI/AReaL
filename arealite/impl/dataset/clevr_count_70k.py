from datasets import Dataset
from typing import Any, Dict, List, Optional, Tuple, Union,Literal
import math
from PIL import Image
from PIL.Image import Image as ImageObject
from io import BytesIO
import base64
from jinja2 import Template
def process_image(
    image: Union[Dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
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
    buffer = BytesIO()
    image.save(buffer, format="PNG")  # 或者根据你的图像格式调整，如 PNG
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    return img_base64

def _build_messages(example: Dict[str, Any], prompt_key: str = "seq", image_key: str = "images", video_key: str = "videos") -> List[Dict[str, Any]]:
        prompt_str: str = example[prompt_key]


        if image_key in example:
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image", "image": f"data:image/jpeg;base64,{example[image_key][i-1]}"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        # elif video_key in example:
        #     content_list = []
        #     for i, content in enumerate(prompt_str.split("<video>")):
        #         if i != 0:
        #             content_list.append({"type": "video", "video": f"data:video/mp4;base64,{example[video_key][i-1]}"})

        #         if content:
        #             content_list.append({"type": "text", "text": content})

        #     return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]



def process_clevr_count_70k_sft_dataset(dataset: Dataset, processor):
    '''
    "clevr_count_70k": {
        "image_key": "images",
        "question_key": "problem",
        "answer_key": "answer"
    },
    '''
    tokenizer = processor.tokenizer 
    def process_example(example, idx):
        # Add query_id column
        example["query_id"] = str(idx)
        images=example["images"]
        processed_images=[]
        for image in images:
            processed_images.append(process_image(image,113*113,336*336))
        example["images"] = processed_images
        example["seq"] = example["problem"] + example["answer"] + tokenizer.eos_token
        example["messages"] = _build_messages(example, prompt_key="seq", image_key="images")
        return example

    dataset = dataset.map(
        lambda example, idx: process_example(example, idx),
        with_indices=True,
        remove_columns=["seq","images","answer"],
    )

    def _process(example):
        text=processor.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=True)

        processed_input=processor(
            text=[text],
            images=example["images"],
            padding=False,
            return_tensors="pt",
            return_length=True,
            return_attention_mask=False,
        )

        example["seq"] =processed_input["input_ids"]
        example["pixel_values"] = processed_input["pixel_values"]
        example["image_grid_thw"] = processed_input["image_grid_thw"]
        example["prompt"] = tokenizer(example["prompt"],return_tensors="pt",return_length=True,padding=False,return_attention_mask=False,)["input_ids"]
        return example

    dataset = dataset.map(lambda x: _process(x))
    return dataset