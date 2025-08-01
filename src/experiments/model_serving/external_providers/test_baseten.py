import os

import dotenv
from openai import OpenAI

dotenv.load_dotenv()


def test_baseten():
    """Test Baseten model deployment"""
    client = OpenAI(
        api_key=os.environ["BASETEN_API_KEY"], base_url=os.environ["BASETEN_MODEL_URL"]
    )

    response = client.chat.completions.create(
        model="",
        messages=[{"role": "user", "content": "Tell me a fun fact about Python."}],
        temperature=0.3,
        max_tokens=100,
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    test_baseten()
