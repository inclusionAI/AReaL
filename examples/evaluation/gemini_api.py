import requests
import base64
import os

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')



def call_model(prompt, model="claude-3-7-sonnet-20250219", image_path=None, **kwargs):
    url = 'https://matrixllm.alipay.com/v1/chat/completions'
    api_key = ""  
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }

    content = []
    
    content.append({
        "type": "text",
        "text": prompt
    })

    if image_path:
        ext = os.path.splitext(image_path)[1].lower().replace('.', '')
        mime_type = f"image/{ext}" if ext != 'jpg' else "image/jpeg"
        
        base64_image = encode_image(image_path)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{base64_image}"
            }
        })

    data = {
        "stream": False,
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": content, 
            }
        ],
        "extra_body": {"google": {"thinking_config": {"include_thoughts": True,"thinking_level":"high"}, "thought_tag_marker": "think"}},
        **kwargs,
    }

    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        result = response.json()
        answer_text = result['choices'][0]['message']['content']
        print(result)
        return answer_text
    else:
        print(f"请求失败: {response.status_code}")
        print(response.text)
        return None

if __name__ == "__main__":
    prompt_text = "The road from Anna's to Mary's house is $16 \mathrm{~km}$ long. The road from Mary's to John's house is $20 \mathrm{~km}$ long. The road from the crossing to Mary's house is $9 \mathrm{~km}$ long. How long is the road from Anna's to John's house?"
    call_model(
        prompt=prompt_text, 
        model="gemini-3-pro-preview", 
        image_path="84edit.jpg",
    )