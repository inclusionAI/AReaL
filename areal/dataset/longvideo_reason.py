
import os

from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from typing import Union
from areal.utils.multimodal import QUESTION_TEMPLATE_VIDEO_QWEN

def get_longvideo_reason_rl_dataset(path, split, processor, rank, world_size, video_dir=None):
    """
    longvideo:{
        problem: str
        reasoning:str
        videos:path
        answer:str
    }
    """
    dataset = load_dataset(path=path, split=split)
    dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)

    def process(sample):
        
        # video_token = processor.video_token if processor is not None else "<|video_pad|>"
        assert isinstance(sample["videos"], str), "video path should be a string"
        video_path=os.path.join(video_dir, sample["videos"])
        question= QUESTION_TEMPLATE_VIDEO_QWEN.format(question=sample["problem"])
        content = [{"type": "video", "video": video_path, "max_pixels": processor.video_processor.max_pixels, "min_pixels": processor.video_processor.min_pixels, "min_frames": processor.video_processor.min_frames, "max_frames": processor.video_processor.max_frames}, {"type": "text", "text": question}]
        conversation = [
            {
                "role": "user",
                "content":content
            }
        ]
        messages = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        
        return {
            "messages": messages,
             "conversation":conversation
        }
    dataset = dataset.map(process).remove_columns(["problem", "reasoning"])
    return dataset

    
    
    
    