from datasets import Dataset
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
from PIL import Image
from PIL.Image import Image as ImageObject
from io import BytesIO
import os
import math

# def process_image(
#     image: Union[Dict[str, Any], Image.Image, str], min_pixels: Optional[int] = None, max_pixels: Optional[int] = None
# ) -> BytesIO:
#     # 处理不同格式的图像输入
#     if isinstance(image, str):
#         image = Image.open(image)
#     elif isinstance(image, dict):
#         image = Image.open(BytesIO(image["bytes"]))
#     elif isinstance(image, bytes):
#         image = Image.open(BytesIO(image))

#     image.load()  # 避免 "Too many open files" 错误

#     # 调整图像大小（如果超出最大像素限制，或者小于最小像素限制）
#     if max_pixels is not None and (image.width * image.height) > max_pixels:
#         resize_factor = math.sqrt(max_pixels / (image.width * image.height))
#         width, height = int(image.width * resize_factor), int(image.height * resize_factor)
#         image = image.resize((width, height))

#     if min_pixels is not None and (image.width * image.height) < min_pixels:
#         resize_factor = math.sqrt(min_pixels / (image.width * image.height))
#         width, height = int(image.width * resize_factor), int(image.height * resize_factor)
#         image = image.resize((width, height))

#     # 如果图像模式不是 RGB，则转换为 RGB 模式
#     if image.mode != "RGB":
#         image = image.convert("RGB")

#     # 将图像保存到 BytesIO 对象
#     img_byte_arr = BytesIO()
#     image.save(img_byte_arr, format="PNG")  # 你可以选择不同的格式，如PNG或JPEG
#     img_byte_arr.seek(0)  # 将指针重置到字节流的开始位置

#     return img_byte_arr


#Standard dataset Key:"id", "image", "question", "solution", "hash"
VL_DATASET=["clevr_count_70k", "llava_cot", "mm_mathinstruct"]

def process_VL_dataset(
    dataset: Dataset,
):
    if dataset.info.dataset_name.lower() == "clevr_count_70k":

        return process_clevr_count_dataset(dataset)
    elif dataset.info.dataset_name.lower() == "llava_cot":
        return process_llava_cot_dataset(dataset)
    elif dataset.info.dataset_name.lower() == "mm_mathinstruct":
        return process_MathInstruct_dataset(dataset)
    else:
        raise ValueError(f"Unsupported VL dataset: {dataset.info.dataset_name}. Supported datasets are: {VL_DATASET}")



def generate_image_hash(image: Image.Image) -> str:
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG') 
    img_byte_arr = img_byte_arr.getvalue()
    return hashlib.sha256(img_byte_arr).hexdigest()

def generate_question_hash(question: str) -> str:
    return hashlib.sha256(question.encode('utf-8')).hexdigest()

def generate_hash(images: list, question: str) -> str:
    image_hashes = [generate_image_hash(image) for image in images]
    
    question_hash = generate_question_hash(question)
    
    combined_hash_input = question_hash + ''.join(image_hashes)
    return hashlib.sha256(combined_hash_input.encode('utf-8')).hexdigest()

'''
DatasetDict({
    train: Dataset({
        features: ['images', 'problem', 'answer'],
        num_rows: 2871988
    })
})
'''

def process_clevr_count_dataset(dataset: Dataset):

    def process_example(example):
        # processed_image=[]

        # for image in example["images"]:
        #     processed_image.append(process_image(image))
        example["question"]=example["problem"]
        example["solution"]=example["answer"]
        example["image"]=example["images"]
        # example["hash"] = generate_hash(example["image"], example["question"])

        return example
    dataset = dataset.map(
        lambda example: process_example(example),
        remove_columns=["problem","answer","images"],
        num_proc=os.cpu_count()
    )

    return dataset

'''
{
"id": ID,
"image": IMAGE_PATH,
"conversations": [{"from": "human", "value": QUESTION},{"from": "gpt", "value": ANSWER}]
}

'''

def process_llava_cot_dataset(dataset: Dataset):

    def process_example(example):
        processed_image=[]
        for image in example["image"]:
            processed_image.append(process_image(image))
        example["image"]=processed_image
        # the query text used as input (prompt) for the evaluation model
        example["hash"] = generate_hash(example["image"], example["conversations"])
        example["question"] = example["conversations"][0]["value"]
        example["solution"]= example["conversations"][1]["value"]

        return example

    dataset = dataset.map(
        lambda example: process_example(example),
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
        processed_image=[]
        for image in example["image"]:
            processed_image.append(process_image(image))
        example["image"]=processed_image
        example["hash"] = generate_hash(example["image"], example["question"])
        return example

    dataset = dataset.map(
        lambda example: process_example(example),
    )
    return dataset