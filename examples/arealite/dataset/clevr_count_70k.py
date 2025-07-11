from datasets import Dataset
from typing import Any, Dict, List, Optional, Tuple, Union,Literal
import math
from PIL import Image
from PIL.Image import Image as ImageObject
from io import BytesIO
import base64
from jinja2 import Template
import os
from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node

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

    return image

def get_clevr_count_70k_sft_dataset(path, split, processor, rank, world_size):
    '''
    "clevr_count_70k": {
        "image_key": "images",
        "question_key": "problem",
        "answer_key": "answer"
    },
    '''
    dataset = load_dataset(path=path, split=split)
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    
    tokenizer = processor.tokenizer 
    def process_example(example, idx):
        # Add query_id column
        images = example["images"]
        image_token = processor.image_token if processor is not None else "<image>"
        example["problem"] = example["problem"].replace("<image>", image_token)
        processed_images = []
        for image in images:
            processed_images.append(process_image(image,113*113,336*336))
        example["images"] = processed_images
        example["seq"] = example["problem"] + example["answer"] + tokenizer.eos_token
        return example

    dataset = dataset.map(
        lambda example, idx: process_example(example, idx),
        with_indices=True,
        remove_columns=["answer"],
        num_proc=os.cpu_count()
    )

    def _process(example):
        text=example["seq"]
        processed_input=processor(
            text=[text],
            images=example["images"],
            padding=False,
            return_tensors="pt",
            return_length=True,
            return_attention_mask=False,
        )

        example["input_ids"] =processed_input["input_ids"].squeeze(0)
        example["pixel_values"] = processed_input["pixel_values"]
        example["image_grid_thw"] = processed_input["image_grid_thw"]
        prompt_token = tokenizer.encode(example["problem"])
        prompt_mask = [1] * len(prompt_token) + [0] * (
            len(example["input_ids"]) - len(prompt_token)
        )
        example["prompt_mask"]=prompt_mask
        return example

    dataset = dataset.map(lambda x: _process(x),remove_columns=["images","seq","problem"],num_proc=os.cpu_count())
    return dataset

# def get_clevr_count_70k_rl_dataset(dataset: Dataset, processor):
#     tokenizer = processor.tokenizer 
#     def process_example(example, idx):
#         # Add query_id column
#         example["query_id"] = str(idx)
#         images=example["images"]
#         image_token = processor.image_token if processor is not None else "<image>"
#         example["problem"] = example["problem"].replace("<image>", image_token)
#         processed_images=[]
#         for image in images:
#             processed_images.append(process_image(image,113*113,336*336))
#         example["images"] = processed_images
#         example["seq"] = example["problem"] + example["answer"] + tokenizer.eos_token
#         return example

#     dataset = dataset.map(
#         lambda example, idx: process_example(example, idx),
#         with_indices=True,
#         num_proc=os.cpu_count()
#     )
#     return dataset