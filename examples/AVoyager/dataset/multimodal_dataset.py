# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import io
import os
import json
import copy
import math
import torch
import random
import datasets
import numpy as np
import pandas as pd
from omegaconf import ListConfig
from collections import defaultdict
from torch.utils.data import Dataset
from typing import List, Union, Optional
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from transformers import PreTrainedTokenizer, ProcessorMixin

datasets.disable_caching()

def make_conversation_multimodal(df, system_prompt=None):
    def make_conv(row):
        prompt = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": row["problem"],
            },
        ]
        images = row["images"]
        base_image_dir = os.getenv("BASE_IMAGE_DIR", "./datasets")
        row["images"] = [os.path.join(base_image_dir, image) for image in images]
        row["prompt"] = prompt
        return row
    df = df.apply(make_conv, axis=1)
    return df

def collate_fn(data_list: list[dict]) -> dict:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        if key == 'raw_prompt_ids': # FIXME: hack to cope with dataset whose input_ids is all the same
            array_list = np.empty(len(val), dtype=object)
            for i, lst in enumerate(val):
                array_list[i] = lst
            non_tensors[key] = array_list
        else:
            non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}

def process_image(image: dict, max_pixels: int = 2048 * 2048, min_pixels: int = 512 * 512, return_original_image=False):
    from io import BytesIO
    from PIL import Image

    if isinstance(image, dict):
        image = Image.open(BytesIO(image['bytes']))
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image).convert('RGB')
    elif isinstance(image, str):
        assert os.path.exists(image), f"Image path {image} not exists."
        try:
            image = Image.open(image).convert('RGB')
        except Exception as exn:
            print(f"Failed to open image {image}. Exception:", exn)
            raise exn

    if return_original_image:
        return image

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.Resampling.LANCZOS)

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.Resampling.LANCZOS)

    assert image.width >= 28 and image.height >= 28, "Qwen image size should be larger than 28 * 28."

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image

class MultiModalDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 data_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 processor: Optional[ProcessorMixin] = None,
                 prompt_key='prompt',
                 answer_key='answer',
                 image_key='images',
                 max_prompt_length=1024,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error',
                 system_prompt: str = None,
                 max_pixels: int = 2048 * 2048,
                 min_pixels: int = 512 * 512,
                 mask_blank: bool = False,
                 use_3drope: bool = True,
                 general_qa_reward_fn: str = 'v1',
                 use_raw_image: bool = False,
                ):
        import warnings
        warnings.filterwarnings("ignore")
        import logging
        logging.disable(logging.CRITICAL)

        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer
        self.processor = processor

        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation

        self.system_prompt = system_prompt
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        
        self.mask_blank = mask_blank
        self.use_3drope = use_3drope
        self.general_qa_reward_fn = general_qa_reward_fn
        self.use_raw_image = use_raw_image

        print("self.use_3drope: ", self.use_3drope)
        print("self.general_qa_reward_fn: ", self.general_qa_reward_fn)
        print("self.use_raw_image:", self.use_raw_image)

        # whether to store the dataset in state_dict()
        # default not store
        self.serialize_dataset = False
        # self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_data_files=False):
        from verl.utils.fs import copy_to_local
        data_files = self.data_files if not use_origin_data_files else self.original_data_files
        for i, data_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=data_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        dataframes = []
        for data_file in self.data_files:
            with open(data_file, "r") as f:
                json_data = json.load(f)
            dataframes.append(pd.DataFrame(json_data))
        self.dataframe = pd.concat(dataframes)
        print(f">>> Original Data Size: {len(self.dataframe)}")

        # filter out too long prompts
        tokenizer = self.tokenizer
        prompt_key = self.prompt_key
        self.dataframe = make_conversation_multimodal(self.dataframe, self.system_prompt)   # Assume 'prompt' is not built in the dataframe
        self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(
            tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)) <= self.max_prompt_length, axis=1)]

        print(f'>>> Filtered Dataset Size: {len(self.dataframe)}')

    def resume_dataset_state(self):
        self.serialize_dataset = False if hasattr(self, 'original_arrow_files') else True
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_arrow=True)  # download and resume from original arrow files
            self._read_files_and_tokenize()
        else:
            print(r'old dataloader ckpt file is used, please train from scratch for better ckpt performance')
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe.iloc[item].to_dict()
        chat = row_dict.pop(self.prompt_key)
        prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

        if self.image_key in row_dict:  # expand image token
            images = row_dict[self.image_key]
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            try:
                row_dict['multi_modal_data'] = {'image': [process_image(image, self.max_pixels, self.min_pixels) for image in row_dict[self.image_key]]}
            except Exception as e:
                print(str(e))
                return self.__getitem__(item+1) if item + 1 < len(self) else self.__getitem__(0)
            
            row_dict['image_size_used'] = [_.size for _ in row_dict['multi_modal_data']['image']]

            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']

            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                              self.processor.image_token)
        else:
            raw_prompt = prompt_with_chat_template

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        if self.mask_blank and images[0].split("/")[-1].split(".")[0] == "blank_image":
            image_token_id = self.tokenizer.encode(self.processor.image_token)[0]
            attention_mask[input_ids == image_token_id] = 0

        if self.use_raw_image:
            try:
                assert len(row_dict[self.image_key]) == 1
                row_dict['multi_modal_data']['image'].extend([image for image in row_dict[self.image_key]])
            except Exception as e:
                print(str(e))
                return self.__getitem__(item+1) if item + 1 < len(self) else self.__getitem__(0)

        if self.use_3drope and self.image_key in row_dict:
            from verl.models.transformers.qwen2_vl import get_rope_index
            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=image_grid_thw,
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)
            # Add raw image at the end of multimodal_dataset to avoid <|image_pad|> error
            row_dict.pop(self.image_key)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]
        row_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if 'reward_model' not in row_dict:
            row_dict['ground_truth'] = row_dict.pop('solution')
        # encode prompts without chat template
        if self.return_raw_chat:
            assert chat[-1]['role'] == 'user'
            row_dict['raw_prompt'] = chat[-1]['content']

        # add index for each prompt
        index = row_dict.get("doc_id", 0)
        row_dict["index"] = index
        
        import pickle
        try:
            pickle.dumps(row_dict)
        except Exception as e:
            print("Pickle error:", e)

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()
            if 'dataframe' in state:
                del state['dataframe']
            return state
        return self.__dict__.copy()