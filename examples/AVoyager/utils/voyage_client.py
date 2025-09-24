from ast import pattern
from json import tool
from numpy import isin
import requests
import random
import time
import json
import asyncio
import html
import os
from typing import Dict, Any, List

import aiohttp
from transformers import AutoProcessor
from PIL import Image
from AVoyager.utils.voyage_utils import prepare_grounding_inputs_multi_turn, crop_image, process_image
from AVoyager.train.constants import TOOL_CALL_CROP_MULTI_TRUN_PROMPT, ERROR_INFO_MULTI_TURN_PROMPT

VOYAGE_CLIENTS={
    "image-grounding": ImageGroundingClient,
}

class ImageGroundingClient:
    def __init__(self, original_image: Image.Image, processor: AutoProcessor = None):
       # original image is observation 0
       self.observation_list=[original_image]
       self.processor = processor

    def query(self, req_meta: Dict[str, Any]) -> Dict[str, Any]:
        error_info = None
        query = req_meta["query"]
        current_iteration = req_meta["current_iteration"]
        try:
            pattern = ".*{\"bbox_2d\": (.*),.*\"source\": [\',\"](.*)[\',\"]}"
            match = re.match(pattern, query, re.DOTALL)
            bbox, source= match.groups(1), match.groups(2)
            json_objects = [{"bbox_2d": eval(bbox), "source": source}]
            image, bbox = prepare_grounding_inputs_multi_turn(json_objects, self.observation_list)
        except Exception as e:
            error_info = f"Error parsing query: {e}"
            print(error_info)
            image = None
            bbox = None
        
        if error_info is not None:
            tool_outputs = f"ERROR occurs during grounding. Error Information: {error_info}.\n"
        else:
            tool_outputs = crop_image(image, bbox)
            self.observation_list.append(tool_outputs)
        
        if isinstance(tool_outputs, Image.Image):
            tool_call_prompt_message = "<|im_end|>\n<|im_start|>user\n" + TOOL_CALL_CROP_MULTI_TRUN_PROMPT.format(action_turn=current_iteration, observation_turn=current_iteration+1) + "<|im_end|>\n<|im_start|>assistant\n"
            
            resize_image= process_image(tool_outputs)

            input_ids=self.processor(text=tool_call_prompt_message, images=resize_image, return_tensors="pt").input_ids.to_list()[0]
            
            tool_outputs=dict{
                "image": resize_image,
                "input_ids": input_ids,
                "error_info": error_info,
            }       
        else:
            tool_call_prompt_message = "<|im_end|>\n<|im_start|>user\n" + tool_outputs + ERROR_INFO_MULTI_TURN_PROMPT + "<|im_end|>\n<|im_start|>assistant\n"
            input_ids=self.processor.tokenizer.encode(tool_call_prompt_message)
            tool_outputs=dict{
                "image": None,
                "input_ids": input_ids,
                "error_info": error_info,
            }
        return tool_outputs



        

def make_voyage_client(voyage_client_type:str=None, **kwargs) -> Any:
    if voyage_client_type not in VOYAGE_CLIENTS:
        raise ValueError(f"Unknown voyage client type: {voyage_client_type}")
    return VOYAGE_CLIENTS[voyage_client_type](**kwargs)