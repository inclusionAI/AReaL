import os

import dotenv
from litellm import completion
from together import Together

dotenv.load_dotenv()


def test_together_sdk():
    """Test Together AI using their SDK"""
    client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

    stream = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[
            {"role": "user", "content": "What are the top 3 things to do in New York?"}
        ],
        stream=True,
    )

    for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
    print()


def test_litellm():
    """Test Together AI using LiteLLM"""
    response = completion(
        model="together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[
            {"role": "user", "content": "What are the top 3 things to do in New York?"}
        ],
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    print("Testing Together AI SDK:")
    test_together_sdk()
    print("\nTesting LiteLLM:")
    test_litellm()
