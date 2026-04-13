from .base import BaseTool, register_tool
import regex as re
import json
import asyncio
import concurrent.futures
from typing import Tuple, Union, List, Dict, Any
import os

import base64
import io
from PIL import Image
from pathlib import Path
from verl_tool.agent_loop.vision_utils import process_image

def crop(str_image, bbox_2d, padding=(0.1,0.1)):
    """
    Crop the image based on the bounding box coordinates.
    """
    if isinstance(str_image, list):
        str_image = str_image[0]
    if isinstance(str_image, Path) and str_image.exists() or \
        isinstance(str_image, str) and os.path.exists(str_image):
        # If the image is a file path, open it directly
        image = Image.open(str_image)
    elif isinstance(str_image, Image.Image):
        image = str_image
    else:
        image = decode_image_url(str_image)
    img_x, img_y = image.size
    padding_tr = (600.0/img_x, 600.0/img_y)
    padding = (min(padding[0], padding_tr[0]), min(padding[1], padding_tr[1]))

    if bbox_2d[0] < 1 and bbox_2d[1] < 1 and bbox_2d[2] < 1 and bbox_2d[3] < 1:
        normalized_bbox_2d = (float(bbox_2d[0])-padding[0], float(bbox_2d[1])-padding[1], float(bbox_2d[2])+padding[0], float(bbox_2d[3])+padding[1])
    else:
        normalized_bbox_2d = (float(bbox_2d[0])/img_x-padding[0], float(bbox_2d[1])/img_y-padding[1], float(bbox_2d[2])/img_x+padding[0], float(bbox_2d[3])/img_y+padding[1])
    normalized_x1, normalized_y1, normalized_x2, normalized_y2 = normalized_bbox_2d
    normalized_x1 = min(max(0, normalized_x1), 1)
    normalized_y1 = min(max(0, normalized_y1), 1)
    normalized_x2 = min(max(0, normalized_x2), 1)
    normalized_y2 = min(max(0, normalized_y2), 1)
    cropped_img = image.crop((int(normalized_x1*img_x), int(normalized_y1*img_y), int(normalized_x2*img_x), int(normalized_y2*img_y)))
    return cropped_img

def encode_image(img: Image.Image) -> str:
    buffered = io.BytesIO()
    # convert the image to RGB if it is not already
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def decode_image(img_str):
    img_data = base64.b64decode(img_str)
    img = Image.open(io.BytesIO(img_data))
    return img

def encode_image_url(img: Image.Image) -> str:
    encoded_img = encode_image(img)
    return f"data:image/jpeg;base64,{encoded_img}"

def decode_image_url(img_str):
    if img_str.startswith("data:image/jpeg;base64,"):
        img_str = img_str.split("data:image/jpeg;base64,")[1]
    return decode_image(img_str)

def rm_tree(pth: Path):
    for child in pth.iterdir():
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    pth.rmdir()

@register_tool
class PixelReasonerTool(BaseTool):
    tool_type = "pixel_reasoner"

    stop_tokens = ["</tool_call>"]
    valid_mcp_func_names = ['zoom_in', 'crop_image_normalized', 'select_frames', 'crop_image']

    def __init__(self, num_workers=1):
        super().__init__(num_workers)
        # Create a thread pool for CPU-intensive image processing
        self.image_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(32, (os.cpu_count() or 1) + 4),
            thread_name_prefix="image_processor"
        )

    def get_usage_inst(self):
        return ""
    
    def parse_action(self, action: str) -> Tuple[str, bool]:
        """
        Parse the raw action string (which is the llm response) into an actual action and its contents.
        Ensures that the parsed code is valid and safe for execution.
        
        Args:
            action: Raw action string containing bbox_2d & target_image
            
        Returns:
            Tuple containing the extracted code and a validity flag
        """
        try:
            call = json.loads(action.split('<tool_call>')[1].split('</tool_call>')[0])
            name = call.get('name', '')
            if name not in self.valid_mcp_func_names:
                return "", False
        except:
            return "", False
        
        return call, True

    def load_env(self, trajectory_id):
        """
        Load the environment for the given trajectory_id
        """
        env = self.env_cache.get(trajectory_id)
        if env is None:
            env = {
                "trajectory_id": trajectory_id,
                "metadata": {
                    "turns": 0,
                },
                "previous_obs": [],
                "images": None,
                "temporary_images": [],
                "temporary_image_folder": Path(f"tmp/crop_images/{trajectory_id}"),
            }
            env['temporary_image_folder'].mkdir(parents=True, exist_ok=True)
        return env
    
    def update_env(self, trajectory_id, env, action, is_valid, extra_field, observation, **kwargs):
        """
        Update the environment for the given trajectory_id
        """
        # save image
        if isinstance(observation, dict) and 'image' in observation:
            if isinstance(observation['image'], str):
                env['images'].append(self.save_image_to_env(trajectory_id, observation['image']))
            elif isinstance(observation['image'], list):
                env['images'].extend([self.save_image_to_env(trajectory_id, img) for img in observation['image']])
        env["metadata"]["turns"] += 1
        env["previous_obs"].append({
            "action": action,
            "is_valid": is_valid,
            "observation": observation,
            "extra_field": extra_field,
            **kwargs
        })
    
    def delete_env(self, trajectory_id):
        """
        Delete the environment for the given trajectory_id
        """
        env = self.env_cache.pop(trajectory_id, None)

    def save_image_to_env(self, trajectory_id, image: Union[Image.Image, str]) -> str:
        """
        Save the image to the environment for the given trajectory_id
        """
        env = self.load_env(trajectory_id)
        env['temporary_images'].append(image)
        return image

    async def _process_single_image(self, img_source, bbox_2d):
        """Process a single image crop operation asynchronously."""
        def _crop_and_process():
            cropped_img = crop(img_source, bbox_2d)
            processed_img = process_image({"image": cropped_img})
            return processed_img
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.image_executor, _crop_and_process)

    async def _process_multiple_images(self, img_sources, bbox_2d=(0, 0, 1, 1)):
        """Process multiple images concurrently."""
        def _crop_and_process_single(img_source):
            cropped_img = crop(img_source, bbox_2d)
            return process_image({"image": cropped_img})
        
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(self.image_executor, _crop_and_process_single, img_source)
            for img_source in img_sources
        ]
        return await asyncio.gather(*tasks)

    async def conduct_zoom_in_action_async(self, parameters, env):
        """
        Execute the zoom-in action asynchronously.
        """
        valid = False
        missing_parameters = []
        if 'bbox_2d' not in parameters:
            missing_parameters.append('bbox_2d')
        if 'target_image' not in parameters:
            missing_parameters.append('target_image')
        try:
            parameters['target_image'] = int(parameters['target_image'])
        except:
            pass
        if missing_parameters:
            observation = f"Missing parameters: {', '.join(missing_parameters)}"
        elif not isinstance(parameters['bbox_2d'], list) or len(parameters['bbox_2d']) != 4:
            observation = "Invalid bbox_2d format. It should be a list of four numbers."
        elif not isinstance(parameters['target_image'], int) or parameters['target_image'] <= 0 or parameters['target_image'] > len(env['images']):
            observation = f"Invalid target_image index. It should be an integer between 1 and the number of previous images ({len(env['images'])})."
        else:
            try:
                previous_images = env['images']
                img_to_crop = previous_images[parameters['target_image']-1]
                
                # Process image asynchronously
                processed_img = await self._process_single_image(img_to_crop, parameters['bbox_2d'])
                
                encoded_cropped_img = encode_image_url(processed_img)
                image_width, image_height = processed_img.size
                observation = {
                    'obs': f"Here is the cropped image. (Image Size: {image_width}x{image_height})\n<image>",
                    'image': encoded_cropped_img,
                }
                valid = True
            except Exception as e:
                with open('test.json', 'w') as f:
                    json.dump(parameters, f, indent=4)
                observation = f"Error processing image: {str(e)}"
                print(f"Error processing zoom-in action: {str(e)}; parameters: {parameters}")
        return observation, valid
    
    async def conduct_select_frames_action_async(self, parameters, env):
        """
        Execute the select frames action asynchronously with concurrent processing.
        """
        valid = False
        missing_parameters = []
        if 'target_frames' not in parameters:
            missing_parameters.append('target_frames')
        if missing_parameters:
            observation = f"Missing parameters: {', '.join(missing_parameters)}"
        elif not isinstance(parameters['target_frames'], list):
            observation = "Invalid target_frames format. It should be a list of integers."
        elif not all(isinstance(frame, int) and 1 <= frame <= len(env['images']) for frame in parameters['target_frames']):
            observation = f"Invalid target_frames indices. Each index should be an integer between 1 and the number of previous images ({len(env['images'])})."
        else:
            try:
                target_frame_sources = [env['images'][frame - 1] for frame in parameters['target_frames']]
                
                # Process all frames concurrently
                target_frames = await self._process_multiple_images(target_frame_sources)
                
                target_frame_width, target_frame_height = target_frames[0].size
                num_frames = len(target_frames)
                observation = {
                    'obs': f"Here are the selected frames. (Frame Size: {target_frame_width}x{target_frame_height}, Numbered 1 to {num_frames}):" + "<image>" * len(target_frames),
                    'image': [encode_image_url(img) for img in target_frames]
                }
                valid = True
            except Exception as e:
                observation = f"Error processing frames: {str(e)}"
                with open('test.json', 'w') as f:
                    json.dump(parameters, f, indent=4)
                print(f"Error processing select frames action: {str(e)}; parameters: {parameters}")
        return observation, valid

    async def aget_observations(self, trajectory_ids: List[str], actions: List[str], extra_fields: List[Dict[str, Any]]):
        """
        Async version of get_observations for concurrent processing.
        """
        observations = []
        dones = []
        valids = []
        
        # Process all actions concurrently
        tasks = []
        for i, (trajectory_id, action, extra_field) in enumerate(zip(trajectory_ids, actions, extra_fields)):
            task = self._conduct_action_async(trajectory_id, action, extra_field)
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                observations.append(f"Processing error: {str(result)}")
                dones.append(False)
                valids.append(False)
            else:
                obs, done, valid = result
                observations.append(obs)
                dones.append(done)
                valids.append(valid)
        
        return observations, dones, valids

    async def _conduct_action_async(self, trajectory_id: str, action: str, extra_field: Dict[str, Any]):
        """
        Execute the parsed action asynchronously.
        """
        parsed_action, is_valid = self.parse_action(action)
        env = self.load_env(trajectory_id)
        if env['images'] is None:
            env['images'] = [Path(x) if not x.startswith("data:image") else decode_image_url(x) for x in extra_field.get('images', [])]
        
        if not is_valid:
            observation = ""
            done = False
            valid = False
        else:
            done = False
            valid = True
            if 'arguments' not in parsed_action:
                observation = "Missing 'arguments' in the tool call."
                valid = False
            elif not isinstance(parsed_action['arguments'], dict):
                observation = f"'arguments' should be a dictionary of parameters key-value pairs, got {type(parsed_action['arguments'])}."
                valid = False
            elif parsed_action['name'] in ['zoom_in', 'crop_image_normalized', 'crop_image']:
                try:
                    observation, valid = await self.conduct_zoom_in_action_async(parsed_action['arguments'], env)
                except Exception as e:
                    observation = f"Error processing {parsed_action['name']} action: {str(e)}"
                    valid = False
                    print(f"Error processing {parsed_action['name']} action: {str(e)}; parameters: {parsed_action['arguments']}")
            elif parsed_action['name'] == 'select_frames':
                try:
                    observation, valid = await self.conduct_select_frames_action_async(parsed_action['arguments'], env)
                except Exception as e:
                    observation = f"Error processing select frames action: {str(e)}"
                    valid = False
                    print(f"Error processing select frames action: {str(e)}; parameters: {parsed_action['arguments']}")
            else:
                observation = "Unknown action name."
                valid = False

        self.update_env(trajectory_id, env, parsed_action, is_valid, extra_field, observation)
        self.save_env(trajectory_id, env)
        
        return observation, done, valid

    def conduct_zoom_in_action(self, parameters, env):
        """
        Synchronous wrapper for zoom-in action.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.conduct_zoom_in_action_async(parameters, env))
        finally:
            loop.close()
    
    def conduct_select_frames_action(self, parameters, env):
        """
        Synchronous wrapper for select frames action.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.conduct_select_frames_action_async(parameters, env))
        finally:
            loop.close()

    def conduct_action(self, trajectory_id, action, extra_field):
        """
        Synchronous wrapper for backward compatibility.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._conduct_action_async(trajectory_id, action, extra_field))
        finally:
            loop.close()

    def __del__(self):
        """Cleanup when tool is destroyed."""
        if hasattr(self, 'image_executor'):
            self.image_executor.shutdown(wait=False)