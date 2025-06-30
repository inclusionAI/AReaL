from datasets import Dataset
import hashlib
import re
'''
{
"id": ID,
"image": IMAGE_PATH,
"conversations": [{"from": "human", "value": QUESTION},{"from": "gpt", "value": ANSWER}]
}

'''

def process_llava_cot_dataset(dataset: Dataset, processor, reward_mode):
    solution_pattern =r"^(.*?)<CONCLUSION>"
    answer_pattern = r"<CONCLUSION>\n\n(.*?)\n\n</CONCLUSION>"
    def generate_hash(text):
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    def process_example(example):
        # Add query_id column
        example["query_id"] =example["id"]
    
        # the query text used as input (prompt) for the evaluation model
        example["image_hash"] = generate_hash(example["conversations"]+example["image"])
        example["question"] = example["conversations"][0]["value"]
        solution_match = re.search(solution_pattern, example["conversations"][1]["value"], re.DOTALL)
        if solution_match:
            example["solution"] = solution_match.group(1).strip()
        else:
            example["solution"] =None
        answer_match = re.search(answer_pattern, example["conversations"][1]["value"], re.DOTALL)
        if answer_match:
            example["answer"] = answer_match.group(1).strip()

        conversation =  [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": example["image"]},
                    {"type": "text", "text": example["question"]},
                ],
            },
        ]
        example["input"]= processor.apply_chat_template(
            conversation,  return_tensors="pt"
        )
        
        return example

    dataset = dataset.map(
        lambda example: process_example(example),
        batched=True,
    )
    return dataset