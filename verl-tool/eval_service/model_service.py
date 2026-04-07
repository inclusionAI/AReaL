import time
import uuid
import aiohttp
import requests
import regex as re
import openai
import os
import torch
from vllm import SamplingParams
from typing import Dict, Any, List, Tuple
from config import ModelConfig, ToolConfig
from transformers import AutoTokenizer
import asyncio
import random
import subprocess

# 1) A sanitizer that strips all embedded NULs (and, optionally, any
#    other C0 control characters except common whitespace).
CONTROL_CHAR_RE = re.compile(
    # this matches U+0000 through U+001F, excluding tab(09), LF(0A), CR(0D)
    r'[\x00-\x08\x0B\x0C\x0E-\x1F]'
)

def sanitize_request(obj: Any) -> Any:
    """
    Recursively walk through obj and:
      - For dicts: sanitize each value
      - For lists/tuples: sanitize each element
      - For strings: remove embedded nulls (and other control chars)
      - Leave other types untouched
    """
    if isinstance(obj, dict):
        return {sanitize_request(key): sanitize_request(val) for key, val in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(sanitize_request(item) for item in obj)
    elif isinstance(obj, str):
        # strip NUL (\x00) and other C0 control chars
        return CONTROL_CHAR_RE.sub('', obj)
    else:
        return obj
    
class ModelService:
    """verl-tool model inference service"""
    
    def __init__(self, model_config: ModelConfig, tool_config: ToolConfig):
        """initialize model service"""
        self.model_config = model_config
        self.tool_config = tool_config
        self.model = None
        self.session = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model)
        self.encode_lock = asyncio.Lock()
        if self.tool_config.mtrl_sep is None:
            messages = [{"role": "system", "content": "{obs}"}]
            self.tool_config.mtrl_sep = "\n" + self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # self.tool_config.mtrl_sep = self.tool_config.mtrl_sep.replace("system", "user")
    
    def call_tool_server(self, trajectory_ids: List[str], actions: List[str], finish: List[bool], **kwargs: Dict[str, List[Any]]) -> Dict[str, Any]:
        """querying the tool server for the observation and done flag"""
        server_url = self.tool_config.tool_server_url
        # prepare payload
        data = {
            "trajectory_ids": trajectory_ids,
            "actions": actions,
            "finish": finish,
            **kwargs
        }
        try:
            data = sanitize_request(data)
            response = requests.post(server_url, json=data)
            response.raise_for_status()
            result = response.json()
            return result   
        except Exception as e:
            print(f"Error calling tool server: {str(e)}")
            return {
                "observations": [f"Error calling tool server: {str(e)}" for _ in range(len(trajectory_ids))],
                "dones": [True for _ in range(len(trajectory_ids))],
                "valids": [False for _ in range(len(trajectory_ids))]
            }
    
    async def call_tool_server_async(self, trajectory_ids: List[str], actions: List[str], finish: List[bool], **kwargs: Dict[str, List[Any]]) -> Dict[str, Any]:
        """querying the tool server for the observation and done flag using aiohttp"""
        server_url = self.tool_config.tool_server_url
        # prepare payload
        data = {
            "trajectory_ids": trajectory_ids,
            "actions": actions,
            "finish": finish,
            **kwargs
        }
        
        # Create aiohttp session if it doesn't exist
        if self.session is None:
            self.session = aiohttp.ClientSession()
            
        try:
            data = sanitize_request(data)
            async with self.session.post(server_url, json=data) as response:
                response.raise_for_status()
                result = await response.json()
                return result
        except Exception as e:
            print(f"Error calling tool server: {str(e)}")
            return {
                "observations": [f"Error calling tool server: {str(e)}" for _ in range(len(trajectory_ids))],
                "dones": [True for _ in range(len(trajectory_ids))],
                "valids": [False for _ in range(len(trajectory_ids))]
            }
    
    async def post_process_observations(self, next_obs: List[str], dones: List[bool], valid_action: List[bool], finishs: List[bool]):
        """Process observations using the tokenizer with proper async locks"""
        next_obs = [obs if not done else "" for obs, done in zip(next_obs, dones)]
        async with self.encode_lock:
            mtrl_sep = self.tool_config.mtrl_sep
            if self.tool_config.truncate_obs_side == 'left':
                next_obs_ids = self.tokenizer(
                    next_obs,
                    padding='longest',
                    return_tensors='pt',
                    add_special_tokens=False,  # Prevents adding special tokens
                    padding_side='left',
                )['input_ids'].to(torch.int64)
                if next_obs_ids.shape[1] > self.tool_config.max_obs_length:
                    print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.tool_config.max_obs_length}")
                    next_obs_ids = next_obs_ids[:, -self.tool_config.max_obs_length:]
            elif self.tool_config.truncate_obs_side == 'right':
                next_obs_ids = self.tokenizer(
                    next_obs,
                    padding='longest',
                    return_tensors='pt',
                    add_special_tokens=False,  # Prevents adding special tokens
                    padding_side='right',
                )['input_ids'].to(torch.int64)
                if next_obs_ids.shape[1] > self.tool_config.max_obs_length:
                    print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.tool_config.max_obs_length}")
                    next_obs_ids = next_obs_ids[:, :self.tool_config.max_obs_length]
            else:
                raise ValueError(f"Invalid truncate_obs_side: {self.tool_config.truncate_obs_side}")
            if self.tool_config.enable_mtrl:
                next_obs = self.tokenizer.batch_decode(
                    next_obs_ids,
                    skip_special_tokens=True
                )
                processed_next_obs = []
                for i in range(len(next_obs)):
                    if finishs[i] or dones[i]:
                        # do action is false
                        assert next_obs[i] == "", f"next_obs should be empty when finishs is True, but got {next_obs[i]}"
                        processed_next_obs.append("")
                    elif valid_action[i]:
                        processed_next_obs.append(mtrl_sep.format(obs=next_obs[i]))
                    else:
                        processed_next_obs.append(mtrl_sep.format(obs="Your action is not valid, please check the format and try again." + next_obs[i]))
                next_obs = processed_next_obs
                next_obs_ids = self.tokenizer(
                    next_obs,
                    padding='longest',
                    return_tensors='pt',
                    add_special_tokens=False,  # Prevents adding special tokens
                )['input_ids'].to(torch.int64)
            next_obs = self.tokenizer.batch_decode(
                next_obs_ids,
                skip_special_tokens=True,
            )
            return next_obs
    
    async def _postprocess_responses(self, outputs: torch.Tensor, action_step: int) -> torch.Tensor:
        """Process responses to stop at python operation or answer operation."""
        active_responses = [outputs.choices[i].text for i in range(len(outputs.choices))]
        active_finish_reasons = [outputs.choices[i].finish_reason for i in range(len(outputs.choices))]
        
        finishes = []
        for i in range(len(active_responses)):
            finish = True
            if active_finish_reasons[i] == "stop" and outputs.choices[i].stop_reason is not None:
                active_responses[i] = active_responses[i] + outputs.choices[i].stop_reason
                if self.tool_config.enable_mtrl:
                    active_responses[i] += self.tool_config.turn_end_token
                finish = False
            if finish and self.tool_config.min_turns > action_step:
                finish = False
                if self.tool_config.enable_mtrl:
                    if self.tool_config.action_stop_tokens:
                        # add action stop tokens
                        active_responses[i] += self.tool_config.action_stop_tokens[0]
                    active_responses[i] += self.tool_config.turn_end_token
            finishes.append(finish)
        return active_responses, finishes, active_finish_reasons
        
    def load_model(self):
        """load the model using VLLM backend"""
        print(f"Loading Model using VLLM: {self.model_config.model}...")
        # start a VLLM server using vllm.serve
        vllm_args = [f"--{k.replace('_', '-')}" for k in self.model_config.__dict__.keys() if k not in ["model", "api_key", "num_models", "host", "port"]]
        vllm_args = []
        for k, v in self.model_config.__dict__.items():
            if k not in ["model", "api_key", "num_models", "host", "port"]:
                    vllm_args.append(f"--{k.replace('_', '-')}")
                    if not isinstance(v, bool):
                        vllm_args.append(str(v))
        
        host = "0.0.0.0"
        num_models = self.model_config.num_models
        ports = random.sample(range(8000, 9000), num_models)
        self.vllm_processes = []
        gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES", ",".join([str(i) for i in range(torch.cuda.device_count())])).split(",")
        tensor_parallel_size = self.model_config.tensor_parallel_size
        gpu_ids_per_model = [gpu_ids[i:i+tensor_parallel_size] for i in range(0, len(gpu_ids), tensor_parallel_size)]
        assert len(gpu_ids) >= num_models * tensor_parallel_size, f"Not enough GPUs available: {len(gpu_ids)} < {num_models * tensor_parallel_size}"
        for i in range(num_models):
            cmd = [
                "vllm", "serve", self.model_config.model, "--api-key", "token-abc123",
                "--host", host, "--port", str(ports[i]), 
                "--disable-uvicorn-access-log", "--disable-log-stats", "--disable-log-requests"
            ] + vllm_args
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids_per_model[i])
            env["VLLM_LOGGING_LEVEL"] = "ERROR"
            vllm_process = subprocess.Popen(cmd, env=env)
            self.vllm_processes.append(vllm_process)
        self.clients = [
            openai.Client(api_key="token-abc123", base_url=f"http://{host}:{ports[i]}/v1") for i in range(num_models)
        ]
        
        # Wait for the service to start (poll the health endpoint)
        max_retries = 60
        retry_interval = 10
        vllm_model_status = [False for _ in range(num_models)]
        for i in range(max_retries):
            for j in range(num_models):
                if vllm_model_status[j]:
                    continue
                try:
                    response = self.clients[j].models.list()
                    vllm_model_status[j] = True
                    print(f"vLLM instance model-{j} status: {response}")
                except Exception as e:
                    # print(f"vLLM instance model-{j} at {host}:{ports[j]} is not ready yet: {str(e)}")
                    continue
            if all(vllm_model_status):
                print(f"✅ vLLM service started successfully with model: {self.model_config.model}")
                return     
            else:
                time.sleep(retry_interval)
        
        # If we get here, the service failed to start
        print("Failed to start one or more vLLM services. Check vLLM logs.")
        for process in self.vllm_processes:
            stderr = process.stderr.read()
            print(f"vLLM stderr: {stderr}")
            process.terminate()
        
        raise RuntimeError("Failed to start vLLM services")
    
    async def send_request(self, client, prompts: List[str], model:str, sampling_params: dict) -> str:
        # Send the request using the client
        sampling_params = sampling_params.copy()
        # Use the async encode method to get tokens
        async with self.encode_lock:
            prompt_lens = [len(self.tokenizer.encode(prompt)) for prompt in prompts]
            max_prompt_tokens = max(prompt_lens)
        
        sampling_params['max_tokens'] = min(max(self.model_config.max_model_len - max_prompt_tokens, 0), sampling_params['max_tokens'])
        # print(f"Sending request to {client.base_url} with sampling params: {sampling_params}")
        
        # Run the API call in an executor to not block the event loop
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.completions.create(
                model=model,
                prompt=prompts,
                echo=False,
                stream=False,
                **sampling_params
            )
        )
        return response
    
    async def generate_with_tools(self, prompts: List[str], sampling_params: dict) -> Tuple[List[str], List[str]]:
        """
        Generate text with tool calls in a multi-turn loop.
        
        Args:
            prompts: Initial prompts for generation
            sampling_params: Sampling parameters for the model
            
        Returns:
            Tuple of (full_responses, finish_reasons)
        """
        client = random.choice(self.clients) # ensure the same trajectory uses the same client for prefix caching
        assert sampling_params.get("n", 1) <= 1, "n > 1 is not supported yet for tool generation"
        contexts = prompts
        final_responses = ["" for _ in range(len(prompts))]
        traj_ids = [str(uuid.uuid4()) for _ in range(len(prompts))]
        active_masks = [True for _ in range(len(prompts))]
        finish_reasons = [None for _ in range(len(prompts))]
        model = self.model_config.model
        
        # keep trying to generate the response until reached the tool-calling limit
        for action_step in range(self.tool_config.max_turns+1):
            # print(f"Action step: {action_step}/{self.tool_config.max_turns}")
            if action_step == self.tool_config.max_turns:
                # last turn, don't stop by action stop tokens
                if "stop" in sampling_params and sampling_params["stop"] is not None:
                    for action_stop_token in self.tool_config.action_stop_tokens:
                        if action_stop_token in sampling_params["stop"]:
                            sampling_params["stop"].remove(action_stop_token)
                
            active_traj_ids = [traj_ids[i] for i in range(len(traj_ids)) if active_masks[i]]
            active_contexts = [contexts[i] for i in range(len(contexts)) if active_masks[i]]
            if len(active_contexts) == 0:
                break
            
            # send request asynchronously
            outputs = await self.send_request(
                client,
                active_contexts,
                model,
                sampling_params
            )
            active_responses, finishes, active_finish_reasons = await self._postprocess_responses(outputs, action_step)
            
            # Use async tool server call if possible
            if hasattr(self, 'call_tool_server_async'):
                tool_responses = await self.call_tool_server_async(
                    active_traj_ids,
                    active_responses,
                    finishes
                )
            else:
                # Fallback to sync version but run in executor
                tool_responses = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    self.call_tool_server,
                    active_traj_ids,
                    active_responses,
                    finishes
                )
                
            # print(f"Active observations (preprocess): {tool_responses['observations']}")
            observations = await self.post_process_observations(tool_responses["observations"], tool_responses["dones"], tool_responses["valids"], finishes)
            dones = tool_responses["dones"]
            valids = tool_responses["valids"]
            
            # print(f"Active step: {action_step}")
            # print(f"Active responses: {active_responses}")
            # print(f"Active observations: {observations}")
            # print(f"Active dones: {dones}")
            # print(f"Active valids: {valids}")
            # print(f"Active traj_ids: {active_traj_ids}")
            # print(f"Active finishs: {finishes}")
            # print(f"Active finish_reasons: {active_finish_reasons}")
            active_idx = 0
            for i in range(len(contexts)):
                if active_masks[i]:
                    contexts[i] += active_responses[active_idx] + observations[active_idx]
                    final_responses[i] += active_responses[active_idx] + observations[active_idx]
                    finish_reasons[i] = active_finish_reasons[active_idx]
                    active_masks[i] = not dones[active_idx]
                    active_idx += 1
            
        return final_responses, finish_reasons
    
    async def chat_completions_async(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """process API request and generate response"""
        # print(f"Received request: {body}")
        
        if "messages" not in body or not body["messages"]:
            raise ValueError("No messages found in the request.")
        if not 'user' in [message["role"] for message in body["messages"]]:
            raise ValueError("No user message found in the request.")
        
        assert body["model"] == self.model_config.model, f"model mismatch: {body['model']} != {self.model_config.model}"
        
        async with self.encode_lock:
            prompt = self.tokenizer.apply_chat_template(body['messages'],
                                                    add_generation_prompt=True,
                                                    tokenize=False)
        if body.get('n', 1) > 1:
            prompts = [prompt for _ in range(body["n"])]
        else:
            prompts = [prompt]

        sampling_params = {
            "temperature": body.get("temperature", 1.0),
            "max_tokens": body.get("max_tokens", body.get("max_completion_tokens", 512)),
            "top_p": body.get("top_p", 1.0),
            "stop": list(set(body.get("stop", []) + self.tool_config.action_stop_tokens)),
        }

        # print(f"Sampling params: {sampling_params}")
        all_responses, finish_reasons = await self.generate_with_tools(prompts, sampling_params)
        
        async with self.encode_lock:
            prompt_tokens = len(self.tokenizer.encode(prompt))
            completion_tokens = 0
            for response in all_responses:
                completion_tokens += len(self.tokenizer.encode(response))
            total_tokens = prompt_tokens + completion_tokens
        
        # format the response into OpenAI-compliant format
        return {
            "id": f"chatcmpl-{str(uuid.uuid4())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_config.model,
            "choices": [
                {
                    "index": i,
                    "message": {
                        "role": "assistant",
                        "content": all_responses[i],
                    },
                    "finish_reason": finish_reasons[i]
                } for i in range(len(all_responses))
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            } 
        }
    
    def chat_completions(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous wrapper for chat_completions"""
        return asyncio.run(self.chat_completions_async(body))
        
    async def completions_async(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """process API request and generate response async"""
        # print(f"Received request: {body}")
        if 'prompt' not in body:
            raise ValueError("No prompt found in the request.")
        assert body["model"] == self.model_config.model, f"model mismatch: {body['model']} != {self.model_config.model}"
        prompt = body['prompt']

        if body.get('n', 1) > 1:
            prompts = [prompt for _ in range(body["n"])]
        else:
            prompts = [prompt]

        sampling_params = {
            "temperature": body.get("temperature", 1.0),
            "max_tokens": body.get("max_tokens", body.get("max_completion_tokens", 512)),
            "top_p": body.get("top_p", 1.0),
            "stop": list(set(body.get("stop", []) + self.tool_config.action_stop_tokens)),
        }

        all_responses, finish_reasons = await self.generate_with_tools(prompts, sampling_params)
        
        async with self.encode_lock:
            prompt_tokens = len(self.tokenizer.encode(prompt))
            completion_tokens = 0
            for response in all_responses:
                completion_tokens += len(self.tokenizer.encode(response))
            total_tokens = prompt_tokens + completion_tokens
        
        # format the response into OpenAI-compliant format
        return {
            "id": f"chatcmpl-{str(uuid.uuid4())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_config.model,
            "choices": [
                {
                    "index": i,
                    "text": all_responses[i],
                    "finish_reason": finish_reasons[i]
                } for i in range(len(all_responses))
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            } 
        }
    
    def completions(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous wrapper for completions_async"""
        return asyncio.run(self.completions_async(body))
        
    async def close(self):
        """Close any resources (like HTTP sessions and processes) when shutting down"""
        # Close HTTP session
        if self.session:
            await self.session.close()
            self.session = None
            
        # Terminate all VLLM processes
        for process in self.vllm_processes:
            if process:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    
        self.vllm_processes = []
        self.clients = []
        
    def __del__(self):
        """Destructor to ensure resources are cleaned up"""
        try:
            asyncio.run(self.close())
        except RuntimeError:
            # Handle "Event loop is closed" error that can happen during shutdown
            pass
