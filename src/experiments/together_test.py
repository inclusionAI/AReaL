import os

import dotenv
from together import Together
from litellm import completion

dotenv.load_dotenv()

client = Together(api_key=os.getenv("TOGETHER_API_KEY"))


stream = client.chat.completions.create(
  model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
  messages=[{"role": "user", "content": "What are the top 3 things to do in New York?"}],
  stream=True,
)

for chunk in stream:
  print(chunk.choices[0].delta.content or "", end="", flush=True)




response = completion(
  model="together_ai/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
  messages=[{"role": "user", "content": "What are the top 3 things to do in New York?"}],
)

print(response.choices[0].message.content)