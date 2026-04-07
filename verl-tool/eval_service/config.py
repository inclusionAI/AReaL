import os
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass

@dataclass
class ModelConfig:
    model: str
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    trust_remote_code: bool = True
    num_models: int = 1
    max_model_len: int = 4096
@dataclass
class ToolConfig:
    tool_server_url: str = "http://localhost:30150/get_observation"
    max_turns: int = 5  # max generation turns
    truncate_obs_side: str = "left"  # "left" or "right", which side to truncate when the observation is too long
    action_stop_tokens: str = None
    max_obs_length: int = 512  # maximum length of observation
    enable_mtrl: bool=False
    mtrl_sep: str=None # "\n<|im_start|>system\n{obs}<|im_end|>\n<|im_start|>assistant\n"
    turn_end_token: str="<|im_end|>"
    min_turns: int=0
    
    def post_init(self):
        """
        Post-initialization processing for ToolConfig (will not call automatically)
        """
        # action_stop_tokens can be a string or a file path
        if isinstance(self.action_stop_tokens, str):
            if os.path.exists(self.action_stop_tokens):
                with open(self.action_stop_tokens, "r") as f:
                    self.action_stop_tokens = f.read().split(',')
            else:
                self.action_stop_tokens = self.action_stop_tokens.split(',')
            self.action_stop_tokens = [token.strip('\n ') for token in self.action_stop_tokens]
            self.action_stop_tokens = [token for token in self.action_stop_tokens if token]
        else:
            self.action_stop_tokens = None
        print(f"using action_stop_tokens: {self.action_stop_tokens}")

@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 32
    ws_max_queue: int = 1000
    log_level: str = "error"
    timeout_keep_alive: int = 60