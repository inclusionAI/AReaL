import requests
import json


api_key = ''


def call_model(messages, model="deepseek-v3-1-250821", **kwargs):
    url = 'https://matrixllm.alipay.com/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }
    data = {
        "stream": False,
        "model": model,
        "messages": messages,
        **kwargs,
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        answer_text = result['choices'][0]['message']['content']
        # print(answer_text)
        print(result)
    else:
        print(f"请求失败: {response.status_code}")
        print(response.text)
    return answer_text
if __name__ == '__main__':
    messages = [{"role": "user", "content": "你好"}]
    call_model(messages)