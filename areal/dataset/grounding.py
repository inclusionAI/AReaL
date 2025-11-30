import math
import os
from io import BytesIO
from typing import Any
import torch.distributed as dist
from datasets import load_dataset, Sequence, Image as DsImage
from PIL.Image import Image as ImageObject
from PIL import Image
from areal.utils import logging
logger = logging.getLogger(__name__)

DATASET_NUM_PROC = 16


def get_grounding_rl_dataset(
    path: str,      # 现在只支持单个 parquet 文件
    split: str,
    processor,
    max_length: int | None = None,
):
    def _do_preprocess(
        path: str,
        split: str,
        processor,
        max_length: int | None = None,
        num_proc: int | None = DATASET_NUM_PROC,
    ):

        data_files = {split: path}
        dataset = load_dataset("parquet", data_files=data_files, split=split)

        def process(sample):
            processed_images =sample["images"]
            image_processor_type = (
                processor.image_processor.image_processor_type.lower()
            )
            if "qwen" in image_processor_type:
                image_token = "<|vision_start|><|image_pad|><|vision_end|>"
            elif "gemma3" in image_processor_type:
                image_token = processor.boi_token
            else:
                image_token = (
                    processor.image_token if processor is not None else "<image>"
                )

            problem_text = sample["prompt"][0]["content"]

            messages = [
                {
                    "role": "user",
                    "content": problem_text
                    .replace("<image>", image_token),
                }
            ]
            messages = processor.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )

            return {
                "messages": messages,
                "images": processed_images,
                "reward_model": sample["reward_model"],
                "extra_info": sample["extra_info"],
            }
        keep_columns = ["messages", "images", "reward_model", "extra_info"]
        remove_columns = [c for c in dataset.column_names if c not in keep_columns]
        dataset = dataset.map(process, num_proc=num_proc,remove_columns=remove_columns)
        dataset = dataset.cast_column("images", Sequence(DsImage(decode=True)))
        if max_length is not None:
            def filter_length(sample):
                processed_input = processor(
                    text=[sample["messages"]],
                    images=sample["images"],
                    padding=False,
                    return_tensors="pt",
                    return_length=True,
                    return_attention_mask=False,
                )
                total_tokens = len(processed_input["input_ids"].squeeze(0))
                return total_tokens <= max_length

            dataset = dataset.filter(filter_length, num_proc=num_proc)

        return dataset

    if dist.is_initialized():
        num_proc = max(16, min(os.cpu_count(), DATASET_NUM_PROC))
        if int(os.getenv("RANK", "0")) == 0:
            dataset = _do_preprocess(path, split, processor, max_length, num_proc)
        dist.barrier()
    else:
        num_proc = None

    dataset = _do_preprocess(path, split, processor, max_length, num_proc)
    return dataset
