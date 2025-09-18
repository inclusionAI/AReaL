from io import BytesIO
from typing import Any, Dict, Optional, Union

from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from PIL import Image
from PIL.Image import Image as ImageObject
from torchvision import transforms


def pad_to_square(img: Image.Image, fill=(0, 0, 0)) -> Image.Image:

    w, h = img.size
    side = max(w, h)
    new_img = Image.new(img.mode, (side, side), color=fill)
    offset = ((side - w) // 2, (side - h) // 2)
    new_img.paste(img, offset)
    return new_img


def convert_image(
    image: Union[Dict[str, Any], ImageObject, str],
    fixed_width: Optional[int] = None,
    fixed_height: Optional[int] = None,
) -> ImageObject:
    if (
        fixed_width is not None
        and fixed_height is not None
        and (image.width != fixed_width or image.height != fixed_height)
    ):
        preprocess = transforms.Compose(
            [
                transforms.CenterCrop((fixed_width, fixed_height)),  # <─ 核心操作
            ]
        )
        image = preprocess(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    with BytesIO() as output:
        image.save(output, format="JPEG")
        return output.getvalue()


def get_geometry3k_sft_dataset(
    path: str,
    split: str,
    processor,
    rank: int,
    world_size: int,
    max_length: Optional[int] = None,
):
    """
    "geometry3k": {
        "image_key": "images",
        "question_key": "problem",
        "answer_key": "answer"
    },
    """
    dataset = load_dataset(path=path, split=split)

    tokenizer = processor.tokenizer

    def process_example(example, idx):
        # Add query_id column
        images = example["images"]
        image_processor_type = processor.image_processor.image_processor_type.lower()
        if "qwen" in image_processor_type:
            image_token = "<|vision_start|><|image_pad|><|vision_end|>"
        elif "gemma3" in image_processor_type:
            image_token = processor.boi_token
        else:
            image_token = processor.image_token if processor is not None else "<image>"
        example["problem"] = (
            example["problem"].replace("<image>", image_token).replace("different", "")
        )
        processed_images = []
        for image in images:
            processed_images.append(convert_image(image, 512, 512))
        example["images"] = processed_images
        example["seq"] = example["problem"] + example["answer"] + tokenizer.eos_token

        return example

    dataset = dataset.map(
        lambda example, idx: process_example(example, idx),
        with_indices=True,
    )

    def _process(example):
        text = example["seq"]
        processed_input = processor(
            text=[text],
            images=example["images"],
            padding=False,
            return_tensors="pt",
            return_length=True,
            return_attention_mask=False,
        )

        example["input_ids"] = processed_input["input_ids"].squeeze(0)
        multi_modal_input = {}
        multi_modal_input["pixel_values"] = processed_input["pixel_values"]
        if "image_grid_thw" in processed_input:
            multi_modal_input["image_grid_thw"] = processed_input[
                "image_grid_thw"
            ].squeeze(0)
        example["multi_modal_input"] = [multi_modal_input]
        answer_token = tokenizer.encode(example["answer"])
        loss_mask = [0] * (len(example["input_ids"]) - len(answer_token)) + [1] * len(
            answer_token
        )
        example["loss_mask"] = loss_mask
        return example

    dataset = dataset.map(
        lambda x: _process(x), remove_columns=["images", "seq", "problem", "answer"]
    )

    if max_length is not None:
        # Filter out sequences longer than max_length
        dataset = dataset.filter(lambda x: len(x["input_ids"]) <= max_length)

    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    return dataset


def get_geometry3k_rl_dataset(
    path: str,
    split: str,
    processor,
    rank: int,
    world_size: int,
    max_length: Optional[int] = None,
):
    dataset = load_dataset(path=path, split=split)

    def process(sample):
        processed_images = [
            convert_image(image, 448, 448) for image in sample["images"]
        ]
        image_processor_type = processor.image_processor.image_processor_type.lower()
        if "qwen" in image_processor_type:
            image_token = "<|vision_start|><|image_pad|><|vision_end|>"
        elif "gemma3" in image_processor_type:
            image_token = processor.boi_token
        else:
            image_token = processor.image_token if processor is not None else "<image>"
        system_prompt = {
            "role": "system",
            "content": (
                "Solve the following geometric problem based on the image. You may explain your reasoning before providing the final answer. The answer should be enclosed in [ ] and can be a number, decimal, or LaTeX format (e.g. \frac { 4 }{ 9 } \sqrt { 3 }).\n"
            ),
        }

        messages = [
            {
                "role": "user",
                "content": sample["problem"]
                .replace("<image>", image_token)
                .replace("different", ""),
            }
        ]
        messages.insert(0, system_prompt)
        messages = processor.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        return {"messages": messages, "images": processed_images}

    dataset = dataset.map(process).remove_columns(["problem"])

    # Filter out sequences longer than max_length if max_length is provided
    if max_length is not None:

        def filter_length(sample):
            # Process the sample to get the total token count including image tokens
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

        dataset = dataset.filter(filter_length)

    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
    return dataset
