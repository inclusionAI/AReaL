from datasets import Dataset,load_dataset
import hashlib
from PIL import Image
from io import BytesIO
'''
DatasetDict({
    train: Dataset({
        features: ['id', 'image', 'question', 'solution', 'image_path'],
        num_rows: 2871988
    })
})
'''

def process_MathInstruct_dataset(dataset: Dataset, processor, reward_mode):
    def generate_hash(text):
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    def process_example(example):
        img= Image.open(BytesIO(example["image"]))
        example["hash"] = generate_hash(example["image"]+example["question"])
        conversation =  [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": example["image"]},
                    {"type": "text", "text": example["question"]},
                ],
            },
        ]
        example["input"]= processor.apply_chat_template(
            conversation,  return_tensors="pt"
        )
        return example

    dataset = dataset.map(
        lambda example: process_example(example),
        batched=True,
    )
    return dataset