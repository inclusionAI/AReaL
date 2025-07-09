from typing import Dict, List, Union

from datasets import Dataset
from PIL import Image
from PIL.Image import Image as ImageObject

VL_DATASET_KEY = {
    "clevr_count_70k": {
        "image_key": "images",
        "question_key": "problem",
        "answer_key": "answer",
    },
    "mm_mathinstruct": {
        "image_key": "image",
        "question_key": "question",
        "answer_key": "solution",
        "image_dir": "image_path",
    },
}


def register_VL_dataset(dataset_name: str) -> Dict[str, Union[str, List[str]]]:
    if dataset_name.lower() not in VL_DATASET_KEY:
        raise ValueError(
            f"VL dataset {dataset_name} is not supported. Supported datasets are: {VL_DATASET_KEY}"
        )
    return VL_DATASET_KEY[dataset_name.lower()]


# def process_VL_dataset(
#     dataset: Dataset,
# ):
#     if dataset.info.dataset_name.lower() == "clevr_count_70k":

#         return process_clevr_count_dataset(dataset)
#     elif dataset.info.dataset_name.lower() == "llava_cot":
#         return process_llava_cot_dataset(dataset)
#     elif dataset.info.dataset_name.lower() == "mm_mathinstruct":
#         return process_MathInstruct_dataset(dataset)
#     else:
#         raise ValueError(f"Unsupported VL dataset: {dataset.info.dataset_name}. Supported datasets are: {VL_DATASET}")


# def generate_image_hash(image: Image.Image) -> str:
#     img_byte_arr = BytesIO()
#     image.save(img_byte_arr, format='PNG')
#     img_byte_arr = img_byte_arr.getvalue()
#     return hashlib.sha256(img_byte_arr).hexdigest()

# def generate_question_hash(question: str) -> str:
#     return hashlib.sha256(question.encode('utf-8')).hexdigest()

# def generate_hash(images: list, question: str) -> str:
#     image_hashes = [generate_image_hash(image) for image in images]

#     question_hash = generate_question_hash(question)

#     combined_hash_input = question_hash + ''.join(image_hashes)
#     return hashlib.sha256(combined_hash_input.encode('utf-8')).hexdigest()

# '''
# DatasetDict({
#     train: Dataset({
#         features: ['images', 'problem', 'answer'],
#         num_rows: 2871988
#     })
# })
# '''

# def process_clevr_count_dataset(dataset: Dataset):

#     def process_example(example):
#         # processed_image=[]

#         # for image in example["images"]:
#         #     processed_image.append(process_image(image))
#         example["question"]=example["problem"]
#         example["solution"]=example["answer"]
#         example["image"]=example["images"]
#         # example["hash"] = generate_hash(example["image"], example["question"])

#         return example
#     dataset = dataset.map(
#         lambda example: process_example(example),
#         remove_columns=["problem","answer","images"],
#         num_proc=os.cpu_count()
#     )

#     return dataset

# '''
# {
# "id": ID,
# "image": IMAGE_PATH,
# "conversations": [{"from": "human", "value": QUESTION},{"from": "gpt", "value": ANSWER}]
# }

# '''

# def process_llava_cot_dataset(dataset: Dataset):

#     def process_example(example):
#         processed_image=[]
#         for image in example["image"]:
#             processed_image.append(process_image(image))
#         example["image"]=processed_image
#         # the query text used as input (prompt) for the evaluation model
#         example["hash"] = generate_hash(example["image"], example["conversations"])
#         example["question"] = example["conversations"][0]["value"]
#         example["solution"]= example["conversations"][1]["value"]

#         return example

#     dataset = dataset.map(
#         lambda example: process_example(example),
#     )
#     return dataset

# '''
# DatasetDict({
#     train: Dataset({
#         features: ['id', 'image', 'question', 'solution', 'image_path'],
#         num_rows: 2871988
#     })
# })
# '''

# def process_MathInstruct_dataset(dataset: Dataset):
#     def process_example(example):
#         processed_image=[]
#         for image in example["image"]:
#             processed_image.append(process_image(image))
#         example["image"]=processed_image
#         example["hash"] = generate_hash(example["image"], example["question"])
#         return example

#     dataset = dataset.map(
#         lambda example: process_example(example),
#     )
#     return dataset
