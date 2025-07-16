import requests
import subprocess
import time
import json
from typing import List, Dict, Any, Optional

class VLLMManager:
    def __init__(self, host: str = "localhost", port: int = 8000):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.process = None
    
    def start_server(self, model_name: str, gpu_memory_utilization: float = 0.9):
        """Start VLLM server with specified model"""
        try:
            # Check if already running
            response = requests.get(f"{self.base_url}/v1/models")
            if response.status_code == 200:
                print("VLLM server already running")
                return
        except:
            pass
        
        print(f"Starting VLLM server with model {model_name}...")
        cmd = [
            "vllm", "serve", model_name,
            "--host", self.host,
            "--port", str(self.port),
            "--gpu-memory-utilization", str(gpu_memory_utilization)
        ]
        
        self.process = subprocess.Popen(cmd)
        
        # Wait for server to be ready
        max_attempts = 60
        for attempt in range(max_attempts):
            try:
                response = requests.get(f"{self.base_url}/v1/models")
                if response.status_code == 200:
                    print("VLLM server is ready!")
                    return
            except:
                pass
            time.sleep(1)
        
        raise Exception("Failed to start VLLM server")
    
    def get_model_list(self) -> List[Dict[str, Any]]:
        """Get the list of available models"""
        try:
            response = requests.get(f"{self.base_url}/v1/models")
            if response.status_code == 200:
                return response.json()["data"]
            else:
                return []
        except Exception as e:
            print(f"Error getting model list: {e}")
            return []
    
    def chat(self, model: str, message: str, system_prompt: Optional[str] = None, 
             temperature: float = 0.7, max_tokens: int = 1024) -> str:
        """Send a chat message"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        try:
            response = requests.post(f"{self.base_url}/v1/chat/completions", json=payload)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                raise Exception(f"API request failed: {response.status_code}")
        except Exception as e:
            print(f"Error in chat: {e}")
            return ""
    
    def completion(self, model: str, prompt: str, temperature: float = 0.7, 
                   max_tokens: int = 1024) -> str:
        """Send a completion request"""
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        try:
            response = requests.post(f"{self.base_url}/v1/completions", json=payload)
            if response.status_code == 200:
                return response.json()["choices"][0]["text"]
            else:
                raise Exception(f"API request failed: {response.status_code}")
        except Exception as e:
            print(f"Error in completion: {e}")
            return ""
    
    def stop_server(self):
        """Stop the VLLM server"""
        if self.process:
            self.process.terminate()
            self.process.wait()
            print("VLLM server stopped") 