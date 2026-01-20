from typing import Tuple, Dict, Any, List, Optional
import io
from google.genai import types
from pathlib import Path
from PIL import Image
import numpy as np
import logging
import json
import os
from .base import AbstractVLMTask
from ...constants import TOOL_EXECUTION_SUCCESS_PROMPT, TOOL_EXECUTION_FAILURE_PROMPT
from ...utils.logger import setup_logger

logger = setup_logger(__name__)



class VisionQATask(AbstractVLMTask):
    """vision qa task"""
    
    def __init__(
        self,
        task_id: str,
        task_prompt: str,
        task_answer: str,
        task_image_path: str,
        save_dir: Path | str,
        tool_functions: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        super().__init__(task_id)
        self.task_prompt = task_prompt
        self.task_answer = task_answer
        self.task_image_path = task_image_path
        self.tool_functions = tool_functions 
        self.state=True
        self.options = kwargs.get("options", None)
        
        
        self.image_path_map : Dict[int, str] = {}
        self.image_list=[Image.open(self.task_image_path).convert("RGB")]
        
        self.contents=[self.task_prompt]
        self.conversation_history: List[Dict[str, Any]] = []
        
        self.contents.append("Observation 0:")
        self.contents.append(self.image_list[0])
        
        self.save_dir=save_dir
        os.makedirs(self.save_dir, exist_ok=True) 
        self.output_jsonl_path=os.path.join(self.save_dir, "output.jsonl")
        self.extra_info_jsonl_path=os.path.join(self.save_dir, "extra_info.jsonl")
        self.meta_info_jsonl_path=os.path.join(self.save_dir, "meta_info.jsonl")
        self.image_save_dir=os.path.join(self.save_dir, "images")
        os.makedirs(self.image_save_dir, exist_ok=True)
        
        

    def validate(
        self,
        chat_history: List[Dict],
        last_observation: Any,
        full_history: List[Any]
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """verify the task"""
        return 0.0, False, {}
    
    def get_info(self) -> Dict[str, Any]:
       pass

       
    def _stringify_observation_item(self, item: Any) -> Dict[str, Any]:
        if isinstance(item, Image.Image):
            return {"image_data": self.image_path_map.get(id(item))}
        if isinstance(item, types.Content):
            parts=item.parts
            listofdict_parts = []
            for part in parts:
                dict_part = {
                    "text": part.text if part.text else None,
                    "thought": part.thought,
                    "function_call": {
                        "name": part.function_call.name,
                        "args": part.function_call.args,
                    } if part.function_call else None,
                    "function_response": {
                        "name": part.function_response.name,
                        "response": part.function_response.response,
                    } if part.function_response else None,
                }
                listofdict_parts.append(dict_part)
            item = {"parts": listofdict_parts, "role": item.role}
        if isinstance(item, str) and item.startswith("parts=") and " role=" in item:
            parts_str, role_part = item.split(" role=", 1)
            role_part = role_part.strip()
            if role_part.startswith("'") and role_part.endswith("'"):
                role_part = role_part[1:-1]
            return {
                "parts": parts_str[len("parts="):],
                "role": role_part,
            }
        return item
    
    def parse_action(self, step: int, action: types.Content, extra_info: Dict[str, Any]):
        """update task contents from action"""
        self.contents.append(action)
        thinking_process = ""
        output_text = ""
        function_call_part_list = []
        for part in action.parts:
            if part.thought:
                thinking_process += part.text
            elif part.function_call:
                function_call_part_list.append(part)
            elif part.text:
                output_text += part.text
            else:
                continue
        contents_for_save=[self._stringify_observation_item(item) for item in self.contents]
        self.conversation_history.append({
            "step": step,
            "observation": contents_for_save,
            "action": self._stringify_observation_item(action),
            "thinking_process": thinking_process,
            "output_text": output_text,
            "function_call": [(function_call_part.function_call.name, function_call_part.function_call.args) for function_call_part in function_call_part_list] if function_call_part_list else None,
            "extra_info": extra_info,
        })
        
        return function_call_part_list

    def _check_function_calls_legal(self, function_call_part_list: Any) -> Tuple[bool, Optional[str], Optional[str], Optional[int]]:
        if not function_call_part_list:
            logging.warning("No function calls found in the action.")
            return True, None, None, None
        first_call = function_call_part_list[0].function_call
        expected_name = first_call.name
        expected_index = first_call.args.get("image_index") if first_call.args else None
        for part in function_call_part_list[1:]:
            call = part.function_call
            if call.name != expected_name:
                logging.warning(f"Inconsistent function call names: expected {expected_name}, got {call.name}")
                return False, "Function call names are inconsistent in the same action.", None, None
            call_index = call.args.get("image_index") if call.args else None
            if call_index != expected_index:
                logging.warning(f"Inconsistent image_index values: expected {expected_index}, got {call_index}")
                return False, "Function call image_index values are inconsistent in the same action.", None, None
        return True, None, expected_name, expected_index
    def update_observation_from_action(self, function_call_part_list: Any):  
        is_legal, illegal_reason, expected_name, expected_index = self._check_function_calls_legal(function_call_part_list)
        dynamic_image=None
        last_success_function_call=None
        error_result=[]
        dynamic_image_index = expected_index
        if not is_legal:
            logging.warning(illegal_reason)
            for part in function_call_part_list:
                self.contents.append(
                    types.Content(
                        role="tool",
                        parts=[
                            types.Part.from_function_response(
                                name=part.function_call.name,
                                response={"error": illegal_reason}
                            )
                        ]
                    )
                )
            self.contents.append(TOOL_EXECUTION_FAILURE_PROMPT)
            return
        
        if expected_name == "image_crop":
            call_results = []
            for function_call_part in function_call_part_list:
                function_call = function_call_part.function_call
                logging.info(f"Processing function call: {function_call.name} with args: {function_call.args}")
                if function_call.name in self.tool_functions.keys():
                    function_to_call = self.tool_functions[function_call.name]
                    try:
                        result = function_to_call(self.image_list, **function_call.args)
                        if isinstance(result, Image.Image):
                            call_results.append(("image", function_call, result))
                        else:
                            call_results.append(
                                ("error", function_call, f"Function call {function_call.name} with args {function_call.args} failed with error: {result}")
                            )
                    except Exception as e:
                        call_results.append(
                            ("error", function_call, f"Function call {function_call.name} with args {function_call.args} failed with error: {str(e)}")
                        )
                else:
                    call_results.append(("error", function_call, f"Unknown function {function_call.name}"))

            for result_type, function_call, payload in call_results:
                if result_type == "image":
                    dynamic_image = payload
                    self.image_list.append(dynamic_image.copy())
                    image_name = f"output_{len(self.image_list)-1}.jpg"
                    image_path = os.path.join(self.image_save_dir, image_name)
                    dynamic_image.save(image_path)
                    self.image_path_map[id(dynamic_image)] = image_path
                    image_bytes_io = io.BytesIO()
                    dynamic_image.save(image_bytes_io, format="JPEG")
                    image_bytes = image_bytes_io.getvalue()

                    function_response_data = {
                        "image_ref": {f"Observation {len(self.image_list)-1}": image_name},
                    }
                    function_response_multimodal_data = types.FunctionResponsePart(
                        inline_data=types.FunctionResponseBlob(
                            mime_type="image/jpeg",
                            display_name=image_name,
                            data=image_bytes,
                        )
                    )
                    self.contents.append(
                        types.Content(role="tool",
                                      parts=[
                                          types.Part.from_function_response(
                                                name=function_call.name,
                                                response=function_response_data,
                                                parts=[function_response_multimodal_data]
                                                )
                                            ]
                                      )
                        )
                else:
                    self.contents.append(
                        types.Content(
                            role="tool",
                            parts=[
                                types.Part.from_function_response(
                                    name=function_call.name,
                                    response={"error": payload}
                                )
                            ]
                        )
                    )
            had_error = any(result_type != "image" for result_type, _, _ in call_results)
            self.contents.append(
                TOOL_EXECUTION_FAILURE_PROMPT if had_error else TOOL_EXECUTION_SUCCESS_PROMPT
            )
            return
        
        for function_call_part in function_call_part_list:
            function_call=function_call_part.function_call
            logging.info(f"Processing function call: {function_call.name} with args: {function_call.args}")
            if function_call.name in self.tool_functions.keys():
                function_to_call=self.tool_functions[function_call.name]
                target_index = dynamic_image_index
                if dynamic_image is None and function_call.args:
                    target_index = function_call.args.get("image_index", dynamic_image_index)
                dynamic_image_list = list(self.image_list)
                if target_index is not None and 0 <= target_index < len(self.image_list):
                    if dynamic_image is not None:
                        dynamic_image_list[target_index] = dynamic_image.copy()
                    else:
                        dynamic_image_list[target_index] = self.image_list[target_index].copy()
                try:
                    
                    result=function_to_call(dynamic_image_list, **function_call.args)
                    
                    dynamic_image = result
                    if dynamic_image_index is None:
                        dynamic_image_index = function_call.args.get("image_index", dynamic_image_index)
                    last_success_function_call = function_call
                    
                except Exception as e:
                    result = {"function_name": function_call.name, "error_msg":f"Function call {function_call.name} with args {function_call.args} failed with error: {str(e)}"}
                    logging.warning(f"Function call failed as {result}")
                    error_result.append(result)
            else:
                result = {"function_name": function_call.name, "error_msg":f"Unknown function {function_call.name}"}    
                logging.warning(f"Function call failed as {result}")
                error_result.append(result)

        if isinstance(dynamic_image, Image.Image):
            self.image_list.append(dynamic_image.copy())
           
            
            image_name=f"output_{len(self.image_list)-1}.jpg"
            image_path=os.path.join(self.image_save_dir, image_name)
            dynamic_image.save(image_path)
            self.image_path_map[id(dynamic_image)] = image_path
            image_bytes_io = io.BytesIO()
            dynamic_image.save(image_bytes_io, format="JPEG")
            image_bytes = image_bytes_io.getvalue()
            
            function_response_data = {
                "image_ref": {f"Observation {len(self.image_list)-1}": image_name},
            }
            function_response_multimodal_data = types.FunctionResponsePart(
                inline_data=types.FunctionResponseBlob(
                    mime_type="image/jpeg",
                    display_name=image_name,
                    data=image_bytes,
                )
            )
            self.contents.append(
                types.Content(role="tool",
                              parts=[
                                  types.Part.from_function_response(
                                        name=last_success_function_call.name,
                                        response=function_response_data,
                                        parts=[function_response_multimodal_data]
                                        )
                                    ]
                              )
                )
        else:
            for err in error_result:
                self.contents.append(
                    types.Content(
                        role="tool",
                        parts=[
                            types.Part.from_function_response(
                                name=err["function_name"],
                                response={"error": err["error_msg"]}
                            )
                        ]
                    )
                )
        had_error = bool(error_result) or not isinstance(dynamic_image, Image.Image)
        self.contents.append(
            TOOL_EXECUTION_FAILURE_PROMPT if had_error else TOOL_EXECUTION_SUCCESS_PROMPT
        )
        
    def save_trajectory(self):
        """save the trajectory to jsonl files"""
        extra_info_list = []
        function_call_total_count = 0
        function_call_each_count = {}
        function_call_per_step = []
        tokens_used_total = 0
        tokens_used_per_step = []
        
        for record in self.conversation_history:
            function_call = record.get("function_call")
            if function_call:
                function_call_total_count += len(function_call)
                function_names = []
                for function_name, _ in function_call:
                    function_call_each_count[function_name] = function_call_each_count.get(function_name, 0) + 1
                    function_names.append(function_name)
                function_call_per_step.append(function_names)
            else:
                function_call_per_step.append(None)
            tokens_used = record.get("extra_info", {}).get("tokens_used", 0)
            tokens_used_total += tokens_used
            tokens_used_per_step.append(tokens_used)
        
        meta_info = {
            "question": self.task_prompt,
            "options": self.options,
            "answer": self.task_answer,
            "image_path": self.task_image_path,
            "function_call_total_count": function_call_total_count,
            "total_steps": len(self.conversation_history),
            "function_call_each_count": function_call_each_count,
            "function_call_per_step": function_call_per_step,
            "tokens_used_total": tokens_used_total,
            "tokens_used_per_step": tokens_used_per_step,
            "output_text": self.conversation_history[-1]["output_text"] if self.conversation_history else "",
        }
        
        last_step_index = len(self.conversation_history) - 1
        for idx, record in enumerate(self.conversation_history):
            observation = record.get("observation")
            extra_info_list.append({
                "step": record["step"],
                "extra_info": record.pop("extra_info"),
                "observation": observation,
            })
            if idx != last_step_index:
                record.pop("observation", None)
        
        with open(self.extra_info_jsonl_path, "w", encoding="utf-8") as f:
            for record in extra_info_list:
                f.write(json.dumps(record) + "\n")
        
        with open(self.output_jsonl_path, "w", encoding="utf-8") as f:
            for record in self.conversation_history:
                f.write(json.dumps(record) + "\n")
        
        with open(self.meta_info_jsonl_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(meta_info) + "\n")
            
        return meta_info
