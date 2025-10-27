from dotenv import load_dotenv
import os
from openai import OpenAI
# This code loads the OpenAI API key and base URL from environment variables using the dotenv package.
# It ensures that sensitive information is not hardcoded in the script, enhancing security.

from dotenv import load_dotenv
# from problem import problem, problem2, problem3, CoT_solution, CoT_solution_2, CoT_solution_3, problem4, CoT_solution_4, problem5, CoT_solution_5
import os
load_dotenv()
openai_api_key = os.environ.get("INFINI_API_KEY")
openai_base_url = os.environ.get("INFINI_BASE_URL")
silicon_api_key = os.environ.get("SILICON_API_KEY")
silicon_base_url = os.environ.get("SILICON_BASE_URL")
seed = 42
# print(openai_base_url)  
client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)
import re
import numpy as np

from utils import extract_step_descriptions, parse_dependency_output, identify_parallel_steps, extract_code_blocks  
# You can choose a model from the following list
# Or you can log into your Infini-AI or SiliconFlow account, and find an available model you want to use.
# model = "Qwen/QVQ-72B-Preview"
# model="llama-3.3-70b-instruct"
model="deepseek-v3-0324"

# from problem import problem, CoT_solution, problem2, CoT_solution_2
from utils import extract_step_descriptions
model="deepseek-v3-0324"

response = client.chat.completions.create(
    model=model,
    temperature=0,
    seed = seed,
    messages=[
    {"role": "system", "content": """You are a helpful assistant 
        """},
    {"role": "user", "content": "Hello"}
    ]
)
if not response.choices:
    print("Warning: API is not working properly")
# print(response.choices[0].message.content)
print (response)