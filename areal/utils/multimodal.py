import base64
import os
from io import BytesIO
from typing import List, Optional, Dict, Any, Union, Tuple, Dict
from transformers import AutoProcessor, PreTrainedTokenizerFast
import torch
from PIL import Image
from tensordict import TensorDict
from PIL.Image import Image as ImageObject
from areal.utils.qwen_vl_utils import process_vision_info
from areal.utils.data import list_of_dict2dict_of_list

QUESTION_TEMPLATE_VIDEO_QWEN = "You are a helpful assistant. The user asks a question, and then you solves it.\n\nPlease first think deeply about the question based on the given video, and then provide the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> the letter of your choice (A, B, C, or D) </answer>.\n\n Question: {question}"


def image2base64(images: List[ImageObject] | ImageObject) -> List[str] | str:

    if isinstance(images, ImageObject):
        images = [images]

    byte_images = []
    for image in images:
        with BytesIO() as buffer:
            image.save(buffer, format="PNG")
            buffer.seek(0)
            byte_image = base64.b64encode(buffer.read()).decode("utf-8")
            byte_images.append(byte_image)

    return byte_images


class VisionCollator:
    def __init__(self, processor: AutoProcessor, tokenizer: PreTrainedTokenizerFast):
        self.processor = processor
        self.tokenizer = tokenizer

    def __call__(self, data: List[Dict[str, Any]]) -> Dict:
        data=list_of_dict2dict_of_list(data)
        prompts = data["messages"]
        
        if "videos" in data:
            conversations = data["conversation"]
            _, videos = process_vision_info(conversations)
            processed_input = self.processor(
                videos=videos,
                text=prompts,
                padding=True,
                return_tensors="pt",
            )
            input_ids = self.processor.tokenizer(prompts, padding=True, return_tensors="pt")['input_ids'].tolist()

            # Special extension and padding are applied for image but not video. Thus video input has to use tokenizer tokenized input_ids. Ref to https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py#L1295
            processed_input["input_ids"] = input_ids
            processed_input["videos"] = data["videos"] 

        elif "images" in data:
            images=data["images"] 

            processed_input = self.processor(
                images=images,
                text=prompts,
                padding=True,
                return_tensors="pt",
            )
            processed_input["images"] = image2base64(images) 
            processed_input["input_ids"]=processed_input["input_ids"].tolist()
        else:
            raise ValueError("Unsupported data format")

        # batch = TensorDict(processed_input, batch_size=processed_input["input_ids"].shape[0])
        # print(processed_input)
        return [processed_input]

