import math
import os
import json
from io import BytesIO
from typing import Any, Dict, Optional, Union
import base64
import torch
import torch.distributed as dist
from datasets import load_dataset, concatenate_datasets
from datasets.distributed import split_dataset_by_node
from PIL.Image import Image as ImageObject

from areal.utils import logging
from areal.utils.image import load_image

logger = logging.getLogger(__name__)

DATASET_NUM_PROC = 16


def convert_image(
    image: Union[Dict[str, Any], ImageObject, str],
    max_pixels: Optional[int],
) -> str:
    if isinstance(image, str):
        # Assume path; open to PIL
        from PIL import Image as PILImage
        image = PILImage.open(image)

    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(
            image.height * resize_factor
        )
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")
    with BytesIO() as output:
        image.save(output, format="JPEG")
        b64 = base64.b64encode(output.getvalue()).decode("utf-8")
        return b64


def get_multimodal_dataset(
    path: list[str],
    split: str,
    processor,
    rank: int,
    world_size: int,
    max_length: Optional[int] = None,
):
    def _do_preprocess(
        path: str,
        split: str,
        processor,
        max_length: int | None = None,
        num_proc: int | None = None,
    ):
        # Load deepeyes (HF dataset) or visual_probe (JSON) and normalize format
        if isinstance(path, str) and (path.endswith(".json") or path.endswith(".jsonl")):
            dataset = load_dataset("json", data_files=path, split=split)
        else:
            dataset = load_dataset(path=path, split=split)

        def process(sample):
            '''
            deepeyes:{
                images: List[str],
                doc_id: str,
                problem: str,
                solution: str,
                data_source: str,
            }
            visual_probe(train.json):{
                images: List[str],
                doc_id: str,
                problem: str,
                solution: str,
                data_source: str,
            }
            
            require:
            data:{
                question: str,
                qid:str,
                answer:str,
                images: List[PIL.Image.Image],
            }
            '''
            from PIL import Image as PILImage

            processed_images = []
            for img in sample.get("images", []):
                if isinstance(img, str):
                    try:
                        img_obj = PILImage.open(img)
                    except Exception:
                        print(f"Failed to open image from path: {img}")
                        continue
                elif isinstance(img, ImageObject):
                    img_obj = img
                elif isinstance(img, dict) and "path" in img:
                    try:
                        img_obj = PILImage.open(img["path"])  # type: ignore[index]
                    except Exception:
                        print(f"Failed to open image from path: {img['path']}")
                        continue
                else:
                    print(f"Unexpected image format: {img}")
                    continue
                processed_images.append(convert_image(img_obj, 336 * 336))
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

            # Build instruction + user prompt; replace <image> placeholders
            user_text = sample["problem"].replace("<image>", image_token)

            return {
                "question": user_text,
                "messages": user_text,
                "images": processed_images,
                "qid": str(sample["doc_id"]),
                "answer": sample["solution"],
                "data_source": sample.get("data_source", "unknown"),
            }

        dataset = dataset.map(process, num_proc=num_proc).remove_columns(["problem","solution","doc_id"]) if "problem" in dataset.column_names else dataset.map(process, num_proc=num_proc)

        # Filter out sequences longer than max_length if max_length is provided
        if max_length is not None:

            def filter_length(sample):
                # Convert base64 images to PIL for processor
                imgs = []
                for img in sample.get("images", []):
                    src = img if (isinstance(img, str) and img.startswith("data:")) else f"data:image/jpeg;base64,{img}"
                    try:
                        imgs.append(load_image(src))
                    except Exception:
                        pass
                if not imgs:
                    return True
                processed_input = processor(
                    text=[sample.get("messages", sample.get("question", ""))],
                    images=imgs,
                    padding=False,
                    return_tensors="pt",
                    return_length=True,
                    return_attention_mask=False,
                )
                total_tokens = len(processed_input["input_ids"].squeeze(0))
                return total_tokens <= max_length

            dataset = dataset.filter(filter_length)
        return dataset

    if dist.is_initialized():
        # Use multi-processing to accelerate data-processing
        # FIXME: processor process data extremely slowly in transformers > 4.53.1
        num_proc = max(1, min(os.cpu_count(), DATASET_NUM_PROC))
        logger.warning("Please set HF_HOME to your NFS directory")
        if rank == 0:
            # First process data in rank 0, and use HF cache to load pre-processed dataset in other ranks
            dataset = _do_preprocess(path, split, processor, max_length, num_proc)
        dist.barrier()
    else:
        # Do not use multi-processing (slow)
        num_proc = None


    # If use multiprocessing, it will load dataset in HF cache
    datasets = []
    for p in path:
        datasets.append(_do_preprocess(p, split, processor, max_length, num_proc))
    dataset = concatenate_datasets(datasets).shuffle(seed=42)

    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    return dataset
