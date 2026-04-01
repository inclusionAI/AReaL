import requests
import json
import time
import tqdm
import logging


api_key = ''

TOTAL_PROMPT_TOKENS = 0
TOTAL_COMPLETION_TOKENS = 0
TOTAL_TOKENS = 0

MODEL_COSTS = {
    "claude-3-5-sonnet-20241022": dict(input=3/(10**6), output=15/(10**6)),
}

def call_model_claude(prompt, model="claude-3-5-sonnet-20241022", **kwargs):
    global TOTAL_PROMPT_TOKENS, TOTAL_COMPLETION_TOKENS, TOTAL_TOKENS
    url = ''
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    data = {
        "stream": False,
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        **kwargs,
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        # answer_text = result['choices'][0]['message']['content']
        # print(answer_text)
        # print(result)
        TOTAL_PROMPT_TOKENS += result["usage"]["prompt_tokens"]
        TOTAL_COMPLETION_TOKENS += result["usage"]["completion_tokens"]
        TOTAL_TOKENS += result["usage"]["total_tokens"]

        print("LLM Cost: prompt_tokens={}. completion_tokens={}. total_tokens={}. cost={}".format(TOTAL_PROMPT_TOKENS, TOTAL_COMPLETION_TOKENS, TOTAL_TOKENS, TOTAL_PROMPT_TOKENS * MODEL_COSTS[model]["input"] + TOTAL_COMPLETION_TOKENS * MODEL_COSTS[model]["output"]), flush=True)
    else:
        print(f"请求失败: {response.status_code}")
        print(response.text)
        return None

    return result