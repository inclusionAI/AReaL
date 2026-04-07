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
"""
Preprocess the GSM8k dataset to parquet format
"""
import fire
import os
import datasets
import zipfile
import cv2
import os
import regex as re
from glob import glob
from pathlib import Path
from huggingface_hub import hf_hub_download
from collections import defaultdict
from copy import deepcopy

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

system_prompt = """You are a helpful assistant.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "crop_image", "description": "Zoom in on the image based on the bounding box coordinates.", "parameters": {"type": "object", "properties": {"bbox_2d": {"type": "array", "description": "coordinates for bounding box of the area you want to zoom in. minimum value is 0 and maximum value is the width/height of the image.", "items": {"type": "number"}}, "target_image": {"type": "number", "description": "The index of the image to crop. Index from 1 to the number of images. Choose 1 to operate on original image."}}, "required": ["bbox_2d", "target_image"]}}}
{"type": "function", "function": {"name": "select_frames", "description": "Select frames from a video.", "parameters": {"type": "object", "properties": {"target_frames": {"type": "array", "description": "List of frame indices to select from the video (no more than 8 frames in total).", "items": {"type": "integer", "description": "Frame index from 1 to 16."}}}, "required": ["target_frames"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""

guideline = """Guidelines: Understand the given visual information and the user query. Determine if it is beneficial to employ the given visual operations (tools). For a video, we can look closer by `select_frames`. For an image, we can look closer by `crop_image`. Reason with the visual information step by step, and put your final answer within \\boxed{}."""

def images_to_video(image_folder, output_path, fps=24):
    images = sorted(glob(os.path.join(image_folder, "*.jpg")))
    if not images:
        raise ValueError("No .jpg images found in folder.")

    # Read the first image to get size
    frame = cv2.imread(images[0])
    height, width, _ = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img_path in images:
        frame = cv2.imread(img_path)
        out.write(frame)
    out.release()
    print(f"Video saved to {output_path}")

def get_mm_content_len(processor, example):
    messages = deepcopy(example['prompt'])
    for message in messages:
        content = message["content"]
        content_list = []
        segments = re.split("(<image>|<video>)", content)
        segments = [item for item in segments if item != ""]
        segment_idx = defaultdict(int)
        for segment in segments:
            if segment == "<image>":
                content_list.append({"type": "image", "image": example['images'][segment_idx[segment]]["image"]})
                segment_idx[segment] += 1
            elif segment == "<video>":
                content_list.append({"type": "video", "video": example['videos'][segment_idx[segment]]["video"]})
                segment_idx[segment] += 1
            else:
                content_list.append({"type": "text", "text": segment})

        message["content"] = content_list
    raw_prompt = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[raw_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs.input_ids.shape[1]

def main(
    dataset_path: str = 'JasperHaozhe/MVBench-EvalData-PixelReasoner',
    split: str = 'test',
    local_dir: str = 'data/pixel_reasoner/mvbench',
    image_sep = "<image>",
):
    
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    dataset = datasets.load_dataset(dataset_path, split=split)
    image_dir = local_dir / 'images'
    image_zip_file = hf_hub_download(repo_id=dataset_path, filename='images.zip', local_dir=local_dir, repo_type='dataset')
    image_extraction_marker = image_dir / 'finish_extracting.txt'

    image_dir.mkdir(parents=True, exist_ok=True)
    if not image_extraction_marker.exists():
        print(f"Extracting images from {image_zip_file} to {image_dir}")
        with zipfile.ZipFile(image_zip_file, 'r') as zip_ref:
            zip_ref.extractall(image_dir)
        with open(image_dir / 'finish_extracting.txt', 'w') as f:
            f.write('Images extracted successfully.')
        print(f"Images extracted successfully to {image_dir}.")
    else:
        print(f"Images already extracted at {image_dir}. Skipping extraction.")
    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            question_id = example.get("qid")
            question_raw = example.pop('question')
            question_raw += f"\n\n{guideline}"
            # image = example.pop('image')[0]
            images = example.pop('image')
            is_video = example.get('is_video', False)
            image_paths = [image_dir / image for image in images]
            answer = example.pop('answer')
            answer = [answer] if isinstance(answer, str) else answer

            assert all([image_path.exists() for image_path in image_paths]), f"Some images do not exist: {image_paths}"
            mm_content = image_sep + question_raw

            data = {
                "data_source": dataset_path,
                "prompt": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": mm_content,
                    }
                ],
                "images": [{"image": image_path.absolute().as_posix()} for image_path in image_paths],
                "ability": "visual_reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer,
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'qid': question_id,
                    'is_video': is_video,
                    'images': [image_path.absolute().as_posix() for image_path in image_paths]
                }
            }
            return data

        return process_fn
    dataset = dataset.map(function=make_map_fn(split), with_indices=True, remove_columns=dataset.column_names, num_proc=32)
    print(dataset[0])
    dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    print(f"Saved to {len(dataset)} testing samples to {local_dir}/test.parquet")

if __name__ == '__main__':
    fire.Fire(main)
    
"""
python examples/data_preprocess/pixel_reasoner/mvbench.py --dataset_path=JasperHaozhe/MVBench-EvalData-PixelReasoner --split=test --local_dir=data/pixel_reasoner/mvbench
"""