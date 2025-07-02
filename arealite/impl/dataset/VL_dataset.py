from datasets import Dataset
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
from PIL import Image
from PIL.Image import Image as ImageObject
from io import BytesIO
import re
import math

#Standard dataset Key:"id", "image", "question", "solution", "hash"

def process_image(
    image: Union[Dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
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

def generate_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()
'''
DatasetDict({
    train: Dataset({
        features: ['images', 'problem', 'answer'],
        num_rows: 2871988
    })
})
'''

def process_clevr_count_dataset(dataset: Dataset, processor, reward_mode):

    def process_example(example):
        img= process_image(example["images"])
        example["question"]=example["problem"]
        example["solution"]=example["answer"]
        example["hash"] = generate_hash(example["images"]+example["question"])
        example["image"]=img
        return example

    dataset = dataset.map(
        lambda example: process_example(example),
        batched=True,
        remove_columns=["problem","answer"]
    )
    return dataset

'''
{
"id": ID,
"image": IMAGE_PATH,
"conversations": [{"from": "human", "value": QUESTION},{"from": "gpt", "value": ANSWER}]
}

'''

def process_llava_cot_dataset(dataset: Dataset, processor,):
    solution_pattern =r"^(.*?)<CONCLUSION>"
    answer_pattern = r"<CONCLUSION>\n\n(.*?)\n\n</CONCLUSION>"
    def process_example(example):
        # the query text used as input (prompt) for the evaluation model
        example["hash"] = generate_hash(example["conversations"]+example["image"])
        example["question"] = example["conversations"][0]["value"]
        solution_match = re.search(solution_pattern, example["conversations"][1]["value"], re.DOTALL)
        if solution_match:
            example["solution"] = solution_match.group(1).strip()
        else:
            example["solution"] =None
        answer_match = re.search(answer_pattern, example["conversations"][1]["value"], re.DOTALL)
        if answer_match:
            example["answer"] = answer_match.group(1).strip()

        return example

    dataset = dataset.map(
        lambda example: process_example(example),
        batched=True,
    )
    return dataset

'''
DatasetDict({
    train: Dataset({
        features: ['id', 'image', 'question', 'solution', 'image_path'],
        num_rows: 2871988
    })
})
'''

def process_MathInstruct_dataset(dataset: Dataset):
    def process_example(example):
        img= Image.open(BytesIO(example["image"]))
        example["hash"] = generate_hash(example["image"]+example["question"])
        example["image"]=img
        return example

    dataset = dataset.map(
        lambda example: process_example(example),
        batched=True,
    )
    return dataset