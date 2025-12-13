from prompt import STEP_MERGE_SYSTEM_PROMPTS


import requests
import json
import time
import tqdm
import logging
from threading import Lock


api_key = 'Your api key here'

TOTAL_PROMPT_TOKENS = 0
TOTAL_COMPLETION_TOKENS = 0
TOTAL_TOKENS = 0

# Lock for thread-safe counter updates
token_lock = Lock()

MODEL_COSTS = {
    "gemini-2.5-flash": dict(input=3/(10**6), output=15/(10**6)),
}

def call_model_claude(user_prompt, system_prompt, model="gemini-2.5-flash", max_retries=3, retry_delay=2, **kwargs):
    """
    Call the model API with retry mechanism.
    
    Args:
        user_prompt: User's prompt
        system_prompt: System prompt
        model: Model name
        max_retries: Maximum number of retry attempts (default: 3)
        retry_delay: Delay between retries in seconds (default: 2)
        **kwargs: Additional arguments for the API
        
    Returns:
        API response or None if all retries fail
    """
    global TOTAL_PROMPT_TOKENS, TOTAL_COMPLETION_TOKENS, TOTAL_TOKENS
    url = 'https://matrixllm.alipay.com/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    data = {
        "stream": False,
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {   
                "role": "user",
                "content": user_prompt,
            }
        ],
        **kwargs,
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)
            if response.status_code == 200:
                result = response.json()
                
                # Thread-safe token counting
                with token_lock:
                    TOTAL_PROMPT_TOKENS += result["usage"]["prompt_tokens"]
                    TOTAL_COMPLETION_TOKENS += result["usage"]["completion_tokens"]
                    TOTAL_TOKENS += result["usage"]["total_tokens"]
                    cost = TOTAL_PROMPT_TOKENS * MODEL_COSTS[model]["input"] + TOTAL_COMPLETION_TOKENS * MODEL_COSTS[model]["output"]
                    print(f"LLM Cost: prompt_tokens={TOTAL_PROMPT_TOKENS}. completion_tokens={TOTAL_COMPLETION_TOKENS}. total_tokens={TOTAL_TOKENS}. cost={cost}", flush=True)
                
                return result
            else:
                print(f"请求失败 (attempt {attempt + 1}/{max_retries}): {response.status_code}")
                print(response.text)
                
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    
        except requests.exceptions.RequestException as e:
            print(f"请求异常 (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
    
    print(f"所有 {max_retries} 次重试均失败")
    return None
if __name__ == "__main__":
    
    print(call_model_claude("hello", STEP_MERGE_SYSTEM_PROMPTS))