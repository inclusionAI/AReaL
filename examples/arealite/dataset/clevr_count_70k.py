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
def convert_image(
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
        if 'qwen' in processor.image_processor.image_processor_type.lower():
            image_token="<|vision_start|><|image_pad|><|vision_end|>"
        else:
            image_token = processor.image_token if processor is not None else "<image>"
        example["problem"] = example["problem"].replace("<image>", image_token)
        processed_images = []
        for image in images:
            processed_images.append(convert_image(image,113*113,336*336))
        example["images"] = processed_images
        example["seq"] = example["problem"] + example["answer"] + tokenizer.eos_token
        
        return example

    dataset = dataset.map(
        lambda example, idx: process_example(example, idx),
        with_indices=True,
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
        answer_token = tokenizer.encode(example["answer"])
        loss_mask =  [0] * (len(example["input_ids"]) - len(answer_token))+[1]*len(answer_token)
        example["loss_mask"]=loss_mask
        return example

    dataset = dataset.map(lambda x: _process(x),remove_columns=["images","seq","problem","answer"])
    return dataset

def get_clevr_count_70k_rl_dataset(path, split,processor,  rank, world_size):
    dataset = load_dataset(path=path, split=split)
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)

    def process(sample):
        processed_images = [convert_image(image, 113*113, 336*336) for image in sample["images"]]
        if 'qwen' in processor.image_processor.image_processor_type.lower():
            image_token="<|vision_start|><|image_pad|><|vision_end|>"
        else:
            image_token = processor.image_token if processor is not None else "<image>"
        system_prompt = {
            "role": "system", 
            "content": (
                "Solve the following question step by step. Mark your thought process by using the word 'thinking' in your response. "
                "First, analyze the image and identify the distinct items present in the image. "
                "Then, count the number of items and provide the final answer in [ ] format, ensuring that only the number is inside the brackets "
                "without any additional text or explanations. "
                "Make sure to clearly state the number of items after your thought process."
            )
        }

        messages =[{"role": "user", "content": sample["problem"].replace("<image>", image_token)}] 
        messages.insert(0, system_prompt)
        messages=processor.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return {"messages": messages, "images": processed_images}

    dataset = dataset.map(process).remove_columns(["problem"])
    return dataset