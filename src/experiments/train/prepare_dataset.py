"""
Goal:
- Load results in Results object
- Generate trajectories in OpenAI chat format corresponding to the Agent's view of the trajectory and including system message. Have this in jsonl format.
- Create a ChatML format dataset for training assistant only.

"""

from transformers import AutoTokenizer

from tau2.data_model.simulation import Results
from tau2.utils.utils import DATA_DIR

results_path = (
    DATA_DIR
    / "exp"
    / "qwen2.5-7b-retries"
    / "ollama_chat"
    / "qwen2.5:7b_telecom_no-user_gpt-4.1-2025-04-14_1trials.json"
)

results = Results.load(results_path)

print(results.info)

print(results.simulations[0])

print(results.simulations[0].reward_info)

print(results.simulations[0].reward_info.reward_basis)


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)
print(
    tokenizer.chat_template
)  # If you see "<|im_start|>" or {% for message in messages %}, it uses ChatML.


## Setting up data in ChatML format.
def format_openai_chat_to_chatml(messages):
    chatml = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        chatml += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    return chatml


def preprocess(example):
    full_text = format_openai_chat_to_chatml(example["messages"])
    return tokenizer(full_text, truncation=True, padding="max_length", max_length=1024)


def preprocess_assistant_only_chatml(example):
    messages = example["messages"]
    assert messages[-1]["role"] == "assistant"

    prompt_messages = messages[:-1]
    target_message = messages[-1]["content"]

    prompt = format_openai_chat_to_chatml(prompt_messages)
    full_text = prompt + f"<|im_start|>assistant\n{target_message}<|im_end|>"

    tokenized = tokenizer(
        full_text, truncation=True, padding="max_length", max_length=1024
    )
    labels = tokenized["input_ids"][:]

    # Mask prompt tokens so model only learns from the assistant reply
    prompt_len = len(tokenizer(prompt)["input_ids"])
    labels[:prompt_len] = [-100] * prompt_len
    tokenized["labels"] = labels
    return tokenized
