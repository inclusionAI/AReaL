# Mac OS brew install ollama
# Linux: curl -fsSL https://ollama.com/install.sh | sh
# Python sdk: pip install ollama
import ollama

ollama.pull("qwen2.5:1.5b")

# Generate a response
response = ollama.chat(model='qwen2.5:1.5b', messages=[
    {
        'role': 'user',
        'content': 'Why is the sky blue?',
    },
])
print(response['message']['content'])