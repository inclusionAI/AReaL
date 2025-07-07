from datasets import Dataset
from PIL import Image

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
        example["seq"] = example["prompt"] + example["answer"] + tokenizer.eos_token
        example["image_path"]= example["images"]["path"]
        return example

    dataset = dataset.map(
        lambda example, idx: process_example(example, idx),
        with_indices=True,
    )

    def _process(example):
        example["prompt"] = tokenizer(example["prompt"], return_attention_mask=False)[
            "input_ids"
        ]
        example["seq"] = tokenizer(example["seq"], return_attention_mask=False)[
            "input_ids"
        ]
        images=[Image.open(image_path) for image_path in example["image_path"]]

        processed_image=processor.image_processor(images)
        example["pixel_values"] = processed_image["pixel_values"]
        example["image_grid_thw"]= processed_image["image_grid_thw"]
        return example

    dataset = dataset.map(lambda x: _process(x), batched=True)
    return dataset