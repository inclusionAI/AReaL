import os
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from arealite.utils.image import convert_image

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

def get_clevr_count_70k_rl_dataset(path, split,processor,  rank, world_size):
    dataset = load_dataset(path=path, split=split)
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)

    def process(sample):
        processed_images = [convert_image(image, 113*113, 336*336) for image in sample["images"]]
        if 'qwen' in processor.image_processor.image_processor_type.lower():
            image_token="<|vision_start|><|image_pad|><|vision_end|>"
        else:
            image_token = processor.image_token if processor is not None else "<image>"
        messages =[{"role": "user", "content": sample["problem"].replace("<image>", image_token)}] 
        messages=processor.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return {"messages": messages, "images": processed_images}

    dataset = dataset.map(process).remove_columns(["problem"])
    return dataset