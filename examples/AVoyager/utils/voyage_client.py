from typing import Dict, Any, List
import re
from transformers import AutoProcessor
from PIL import Image
from examples.AVoyager.utils.voyage_utils import prepare_grounding_inputs_multi_turn, crop_image, process_image
from examples.AVoyager.train.constants import TOOL_CALL_CROP_MULTI_TRUN_PROMPT, ERROR_INFO_MULTI_TURN_PROMPT

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
            pattern = r".*\{\"bbox_2d\": (.*),.*\"source\": ['\"](.*)['\"]}"
            match = re.match(pattern, query, re.DOTALL)
            bbox, source = match.group(1), match.group(2)
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
            
            resize_image = process_image(tool_outputs)

            input_ids = self.processor(text=tool_call_prompt_message, images=resize_image, return_tensors="pt").input_ids.tolist()[0]

            tool_outputs = {
                "image": resize_image,
                "input_ids": input_ids,
                "text": tool_call_prompt_message,
                "error_info": error_info,
            }
        else:
            tool_call_prompt_message = "<|im_end|>\n<|im_start|>user\n" + tool_outputs + ERROR_INFO_MULTI_TURN_PROMPT + "<|im_end|>\n<|im_start|>assistant\n"
            input_ids = self.processor.tokenizer.encode(tool_call_prompt_message)
            tool_outputs = {
                "image": None,
                "input_ids": input_ids,
                "text": tool_call_prompt_message,
                "error_info": error_info,
            }
        return tool_outputs



VOYAGE_CLIENTS={
    "image-grounding": ImageGroundingClient,
}


def make_voyage_client(voyage_client_type: str = None, **kwargs) -> Any:
    if voyage_client_type not in VOYAGE_CLIENTS:
        raise ValueError(f"Unknown voyage client type: {voyage_client_type}")
    return VOYAGE_CLIENTS[voyage_client_type](**kwargs)
