# https://github.com/hiyouga/EasyR1/blob/main/verl/utils/dataset.py#
import math
import os
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from .VLdataset_dic import register_VL_dataset


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = value

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(
    image: Union[Dict[str, Any], ImageObject, str],
    min_pixels: Optional[int],
    max_pixels: Optional[int],
) -> ImageObject:
    """
    Process an image to ensure it is in RGB format and resized if necessary.
    """
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(
            image.height * resize_factor
        )
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(
            image.height * resize_factor
        )
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


# def pad_sequence_to_length(
#     tensor: torch.Tensor, max_seq_len: int, pad_token_id: int, left_pad: bool = False
# ) -> torch.Tensor:
#     """Pad a nD tensors in the last dim to max_seq_len."""
#     if tensor.size(-1) >= max_seq_len:
#         return tensor

#     pad_shape = list(tensor.shape)
#     pad_shape[-1] = max_seq_len - tensor.size(-1)
#     pad_tensor = torch.full(pad_shape, fill_value=pad_token_id, dtype=tensor.dtype, device=tensor.device)
#     return torch.cat((pad_tensor, tensor), dim=-1) if left_pad else torch.cat((tensor, pad_tensor), dim=-1)

# def postprocess_data(
#     input_ids: torch.Tensor,
#     attention_mask: torch.Tensor,
#     position_ids: torch.Tensor,
#     max_length: int,
#     pad_token_id: int,
#     left_pad: bool = True,
#     truncation: Literal["left", "right", "error"] = "error",
# ):
#     """Pad or truncate data."""
#     assert truncation in ["left", "right", "error"]
#     seq_length = len(input_ids)
#     if seq_length < max_length:
#         input_ids = pad_sequence_to_length(
#             input_ids, max_seq_len=max_length, pad_token_id=pad_token_id, left_pad=left_pad
#         )
#         attention_mask = pad_sequence_to_length(
#             attention_mask, max_seq_len=max_length, pad_token_id=0, left_pad=left_pad
#         )
#         position_ids = pad_sequence_to_length(position_ids, max_seq_len=max_length, pad_token_id=0, left_pad=left_pad)
#     elif seq_length > max_length:
#         if truncation == "left":  # actually, left truncation may not be reasonable
#             input_ids = input_ids[..., -max_length:]
#             attention_mask = attention_mask[..., -max_length:]
#             position_ids = position_ids[..., -max_length:]
#         elif truncation == "right":
#             input_ids = input_ids[..., :max_length]
#             attention_mask = attention_mask[..., :max_length]
#             position_ids = position_ids[..., :max_length]
#         elif truncation == "error":
#             raise RuntimeError(f"Input sequence length {seq_length} is longer than max length {max_length}.")
#         else:
#             raise NotImplementedError(f"Unknown truncation method {truncation}.")

#     return input_ids, attention_mask, position_ids


def get_rope_index(
    processor: "Qwen2VLProcessor",
    input_ids: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Gets the position ids for Qwen2-VL, it should be generated before sharding the sequence.
    The batch dim has been removed and the input_ids should be a 1D tensor representing a single example.
    https://github.com/huggingface/transformers/blob/v4.52.4/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py#L1405
    """
    spatial_merge_size = processor.image_processor.merge_size
    tokens_per_second = 2
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids(
        "<|vision_start|>"
    )
    if input_ids is not None and (
        image_grid_thw is not None or video_grid_thw is not None
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        position_ids = torch.ones(
            3, input_ids.size(0), dtype=input_ids.dtype, device=input_ids.device
        )  # (3, seqlen)
        image_index, video_index = 0, 0
        input_ids = input_ids[attention_mask == 1]
        image_nums, video_nums = 0, 0
        vision_start_indices = torch.argwhere(input_ids == vision_start_token_id)
        vision_tokens = input_ids[vision_start_indices + 1]
        image_nums = (vision_tokens == image_token_id).sum()
        video_nums = (vision_tokens == video_token_id).sum()
        input_tokens = input_ids.tolist()
        llm_pos_ids_list: list = []
        st = 0
        remain_images, remain_videos = image_nums, video_nums
        for _ in range(image_nums + video_nums):
            if image_token_id in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(image_token_id, st)
            else:
                ed_image = len(input_tokens) + 1
            if video_token_id in input_tokens and remain_videos > 0:
                ed_video = input_tokens.index(video_token_id, st)
            else:
                ed_video = len(input_tokens) + 1
            if ed_image < ed_video:
                t, h, w = (
                    image_grid_thw[image_index][0],
                    image_grid_thw[image_index][1],
                    image_grid_thw[image_index][2],
                )
                second_per_grid_t = 0
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = (
                    video_grid_thw[video_index][0],
                    video_grid_thw[video_index][1],
                    video_grid_thw[video_index][2],
                )
                if second_per_grid_ts is not None:
                    second_per_grid_t = second_per_grid_ts[video_index]
                else:
                    second_per_grid_t = 1.0

                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t, llm_grid_h, llm_grid_w = (
                t.item(),
                h.item() // spatial_merge_size,
                w.item() // spatial_merge_size,
            )
            text_len = ed - st

            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
            )

            t_index = (
                torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w)
            )
            t_index = (t_index * second_per_grid_t * tokens_per_second).long().flatten()
            h_index = (
                torch.arange(llm_grid_h)
                .view(1, -1, 1)
                .expand(llm_grid_t, -1, llm_grid_w)
                .flatten()
            )
            w_index = (
                torch.arange(llm_grid_w)
                .view(1, 1, -1)
                .expand(llm_grid_t, llm_grid_h, -1)
                .flatten()
            )
            llm_pos_ids_list.append(
                torch.stack([t_index, h_index, w_index]) + text_len + st_idx
            )
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
            )

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[..., attention_mask == 1] = llm_positions.to(position_ids.device)
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1).to(input_ids.device)
        else:
            position_ids = (
                torch.arange(input_ids.shape[1], device=input_ids.device)
                .view(1, -1)
                .expand(3, -1)
            )

    return position_ids


# def process_video(
#     video: str, min_pixels: Optional[int], max_pixels: Optional[int], video_fps: float, return_fps: bool = False
# ) -> Union[List[ImageObject], Tuple[List[ImageObject], List[float]]]:
#     vision_info = {"video": video, "min_pixels": min_pixels, "max_pixels": max_pixels, "fps": video_fps}
#     return fetch_video(vision_info, return_video_sample_fps=return_fps)


class VLDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "videos",
        image_dir: Optional[str] = None,
        video_fps: float = 2.0,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = False,
        filter_overlong_prompts_workers: int = 16,
        data_split: str = "train",
    ):
        """
        Universal Vision Language Dataset
        loading a dataset from huggingface hub or local directory
        register the dataset in VL_DATASET_KEY
        operator:format_prompt, filter_overlong_prompts
        """
        self.tokenizer = tokenizer
        self.processor = processor
        # self.prompt_key = prompt_key
        # self.answer_key = answer_key
        # self.image_key = image_key
        # self.video_key = video_key
        # self.image_dir = image_dir
        # self.video_fps = video_fps
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        # if "@" in data_path:
        #     data_path, data_split = data_path.split("@")
        # else:
        #     data_split = "train"

        # if os.path.isdir(data_path):
        #     breakpoint()
        #     # when we use dataset builder, we should always refer to the train split
        #     file_type = os.path.splitext(os.listdir(data_path)[0])[-1][1:].replace("jsonl", "json")
        #     self.dataset = load_dataset(file_type, data_dir=data_path, split=data_split)
        # elif os.path.isfile(data_path):
        #     file_type = os.path.splitext(data_path)[-1][1:].replace("jsonl", "json")
        #     self.dataset = load_dataset(file_type, data_files=data_path, split=data_split)
        # else:
        #     # load remote dataset from huggingface hub
        #     self.dataset = load_dataset(data_path, split=data_split)
        self.dataset = load_dataset(data_path, split=data_split)

        self.dataset_info = register_VL_dataset(self.dataset.info.dataset_name)
        self.prompt_key = self.dataset_info["question_key"]
        self.answer_key = self.dataset_info["answer_key"]
        self.image_key = self.dataset_info.get("image_key", None)
        self.image_dir = self.dataset_info.get("image_dir", None)

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        if filter_overlong_prompts:
            self.dataset = self.dataset.filter(
                self._filter_overlong_prompts,
                desc="Filtering overlong prompts",
                num_proc=filter_overlong_prompts_workers,
            )

    def _build_vl_question(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Build the VL question from the example.
        input: example dict with keys: prompt, answer, images
        output standard format according to the processor
        """
        prompt_str = example[self.prompt_key]
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)
        if self.image_key in example:
            image_token = (
                self.processor.image_token if self.processor is not None else "<image>"
            )
            prompt_str = prompt_str.replace("<image>", image_token)

        # if self.image_key in example:
        #     content_list = []
        #     for i, content in enumerate(prompt_str.split("<image>")):
        #         if i != 0:
        #             content_list.append({"type": "image"})

        #         if content:
        #             content_list.append({"type": "text", "text": content})

        #     return [{"role": "user", "content": content_list}]
        # elif self.video_key in example:
        #     content_list = []
        #     for i, content in enumerate(prompt_str.split("<video>")):
        #         if i != 0:
        #             content_list.append({"type": "video"})

        #         if content:
        #             content_list.append({"type": "text", "text": content})

        #     return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]
        return prompt_str

    def _filter_overlong_prompts(self, example: Dict[str, Any]) -> bool:
        messages = self._build_vl_question(example)
        if self.image_key in example:
            # prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            prompt = messages
            images = example[self.image_key]
            if (
                self.image_dir is not None
                and len(images) != 0
                and isinstance(images[0], str)
            ):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(
                    process_image(image, self.min_pixels, self.max_pixels)
                )

            model_inputs = self.processor(
                processed_images,
                [prompt],
                add_special_tokens=False,
                return_tensors="pt",
            )
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        # elif self.video_key in example:
        #     prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        #     videos = example[self.video_key]
        #     if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
        #         videos = [os.path.join(self.image_dir, video) for video in videos]

        #     processed_videos = [] if len(videos) != 0 else None  # text-only data
        #     for video in videos:
        #         processed_videos.append(process_video(video, self.min_pixels, self.max_pixels, self.video_fps))

        #     model_inputs = self.processor(
        #         videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
        #     )
        #     return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        else:
            input_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
            return len(input_ids) <= self.max_prompt_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Get item from the dataset.
        input:example key: prompt, answer, images
        output: example dict with keys:
            vl_prompt_input_ids: input ids of the prompt
            vl_prompt_length: length of the prompt
            answer_input_ids: input ids of the answer
            answer_length: length of the answer
            pixel_values: processed images if available
            multi_modal_data: dict with images or videos if available
        """
        example: dict = self.dataset[index]
        messages = self._build_vl_question(example)
        if self.image_key in example:
            prompt = messages
            # prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            images = example.pop(self.image_key)
            if (
                self.image_dir is not None
                and len(images) != 0
                and isinstance(images[0], str)
            ):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(
                    process_image(image, self.min_pixels, self.max_pixels)
                )

            model_inputs = self.processor(
                images=processed_images,
                text=[prompt],
                add_special_tokens=False,
                return_tensors="pt",
                return_length=True,
                padding=False,
                truncation=True,
                return_attention_mask=False,
                max_length=self.max_prompt_length,
            )
            vl_prompt_input_ids = model_inputs.pop("input_ids")[0]
            vl_prompt_length = model_inputs.pop("length")[0]
            # attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"images": images}
            example["pixel_values"] = model_inputs.pop("pixel_values")
            example["image_grid_thw"] = model_inputs.pop("image_grid_thw", None)
        # elif self.video_key in example:
        #     prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        #     videos = example.pop(self.video_key)
        #     if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
        #         videos = [os.path.join(self.image_dir, video) for video in videos]

        #     processed_videos = [] if len(videos) != 0 else None  # text-only data
        #     video_fps_list = []
        # for video in videos:
        #     processed_video, video_fps = process_video(
        #         video, self.min_pixels, self.max_pixels, self.video_fps, return_fps=True
        #     )
        #     processed_videos.append(processed_video)
        #     video_fps_list.append(video_fps)

        # model_inputs = self.processor(
        #     videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
        # )
        # if "second_per_grid_ts" in self.processor.model_input_names:
        #     model_inputs["second_per_grid_ts"] = [2.0 / video_sample_fps for video_sample_fps in video_fps_list]

        # input_ids = model_inputs.pop("input_ids")[0]
        # attention_mask = model_inputs.pop("attention_mask")[0]
        # example["multi_modal_data"] = {"videos": videos}
        else:
            # prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            prompt = messages
            model_inputs = self.tokenizer(
                prompt,
                add_special_tokens=False,
                return_tensors="pt",
                return_length=True,
                padding=False,
                truncation=True,
                max_length=self.max_prompt_length,
            )
            vl_prompt_input_ids = model_inputs.pop("input_ids")[0]
            vl_prompt_length = model_inputs.pop("length")[0]
            model_inputs.pop("attention_mask")[0]

        # if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
        #     # qwen2vl mrope
        #     position_ids = get_rope_index(
        #         self.processor,
        #         input_ids=vl_prompt_input_ids,
        #         image_grid_thw=model_inputs.get("image_grid_thw", None),
        #         video_grid_thw=model_inputs.get("video_grid_thw", None),
        #         second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
        #         attention_mask=attention_mask,
        #     )  # (3, seq_length)
        # else:
        #     position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)
        # position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)

        # vl_prompt_input_ids, attention_mask, position_ids = postprocess_data(
        #     input_ids=vl_prompt_input_ids,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     max_length=self.max_prompt_length,
        #     pad_token_id=self.tokenizer.pad_token_id,
        #     left_pad=True,
        #     truncation=self.truncation,
        # )
        answer_input = self.tokenizer(
            example[self.answer_key],
            add_special_tokens=False,
            return_tensors="pt",
            return_length=True,
            padding=False,
            truncation=True,
            max_length=self.max_prompt_length,
        )
        answer_input_ids = answer_input.pop("input_ids")[0]
        answer_length = answer_input.pop("length")[0]
        # raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        # if len(raw_prompt_ids) > self.max_prompt_length:
        #     if self.truncation == "left":
        #         raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
        #     elif self.truncation == "right":
        #         raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
        #     elif self.truncation == "error":
        #         raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["vl_prompt_input_ids"] = vl_prompt_input_ids
        # example["attention_mask"] = attention_mask
        # example["position_ids"] = position_ids
        # example["raw_prompt_ids"] = raw_prompt_ids
        example["vl_prompt_length"] = vl_prompt_length
        example["answer_input_ids"] = answer_input_ids
        example["answer_length"] = answer_length
        example["answer"] = example.pop(self.answer_key)

        return example
