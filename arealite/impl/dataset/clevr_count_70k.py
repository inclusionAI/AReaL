from datasets import Dataset
import os

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
        processed_input=processor(images,[example["prompt"] + example["answer"] + tokenizer.eos_token]
        ,add_special_tokens=False, return_tensors="pt"
        ,return_length=True,padding=False,truncation=True,return_attention_mask=False,)
        example["seq"] =processed_input["input_ids"]
        


        
        example["pixel_values"] = processed_input["pixel_values"]
        example["image_grid_thw"] = processed_input["image_grid_thw"]
        example["prompt"] = tokenizer(example["prompt"],add_special_tokens=False, return_tensors="pt",
        return_length=True,padding=False,truncation=True,return_attention_mask=False,)[
            "input_ids"
        ]
        return example

    dataset = dataset.map(lambda x: _process(x),remove_columns=["images"],num_proc=os.cpu_count())
    return dataset