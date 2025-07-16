from typing import Any, Dict,  Optional, Union
import math
from PIL.Image import Image as ImageObject
import os
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
import base64
from io import BytesIO

def input_text(text:str):
    return {"type": "input_text", "text": text}
def input_image(base64_image: str):
    return {"type": "input_image", "image_url":  f"data:image/jpeg;base64,{base64_image}"}
def build_raw_message(sample: Dict[str, Any], base64_images: list[str]) -> list[Dict[str, Any]]:
    
    raw_message = []
    problem_parts = [part.strip() for part in sample["problem"].split("<image>") if part.strip()]
    insert_list = []
    for i, part in enumerate(problem_parts):
        if i > 0 or sample["problem"].startswith("<image>"):  
            insert_list.append("image")
        part = part.strip()  
        if part:  
            insert_list.append("text")
    image_index = 0
    text_index = 0

    for insert_type in insert_list:
        if insert_type == "text" and text_index < len(problem_parts):
            raw_message.append(input_text(problem_parts[text_index].strip()))
            text_index += 1
        elif insert_type == "image" and image_index < len(base64_images):
            raw_message.append(input_image(base64_images[image_index]))
            image_index += 1
    messages = [{"role": "user", "content": raw_message}]
    return messages


def encode_image(image_file):
    return base64.b64encode(image_file).decode("utf-8")
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
    with BytesIO() as output:
        image.save(output, format="JPEG")
        return output.getvalue()

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

# def get_clevr_count_70k_rl_dataset(path, split,  rank, world_size):
#     dataset = load_dataset(path=path, split=split)
#     dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
#     def process(sample):
#         processed_images = [process_image(image, 113*113, 336*336) for image in sample["images"]]
#         base64_images = [encode_image(image) for image in processed_images]
#         messages = build_raw_message(sample, base64_images)

#         return {"messages": messages,"images":processed_images}
    
#     dataset = dataset.map(process).remove_columns(["problem"])
#     breakpoint()
#     return dataset

def get_clevr_count_70k_rl_dataset(path, split,processor,  rank, world_size):
    dataset = load_dataset(path=path, split=split)
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)

    def process(sample):
        processed_images = [process_image(image, 113*113, 336*336) for image in sample["images"]]
        image_token = processor.image_token if processor is not None else "<image>"
        messages = sample["problem"].replace("<image>", image_token)
        return {"messages": messages, "images": processed_images}

    dataset = dataset.map(process).remove_columns(["problem"])
    return dataset