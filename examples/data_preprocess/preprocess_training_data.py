import json
import sys
from argparse import ArgumentParser
from typing import Dict, List

# An example of prompt template, please remember to add special tokens, this example is for boba-2 coding dataset
prompt_template = """
<|im_start|>user\n{question}\n/think<|im_end|>\n<|im_start|>assistant\n<think>
"""


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file with validation"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"ERROR: JSONL file not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        print(f"ERROR: JSON parsing failed in {file_path}: {str(e)}")
        raise


def process_code_data(file_path: str) -> List[Dict]:
    """Process code dataset from JSONL file"""
    if not file_path:
        return []

    raw_data = load_jsonl(file_path)
    processed = []

    for item in raw_data:
        # Field extraction and transformation
        input_output = json.loads(item["input_output"])
        processed.append(
            {
                "task": "code",
                "query_id": item["query_id"],
                "prompt": prompt_template.format(question=item["question"]),
                "solutions": item.get("solutions", []),  # nothing for code dataset
                "input_output": json.dumps(
                    {
                        "inputs": input_output.get("inputs", []),
                        "outputs": input_output.get("outputs", []),
                        "fn_name": item.get("metadata", {}).get("fn_name", ""),
                        "remote": False,
                    }
                ),
                "language": item.get("language", "PYTHON"),  # default to python
            }
        )

        case_size = sys.getsizeof(processed[-1]["input_output"])
        assert (
            case_size < 500 * 1024
        ), f"'input_output' exceeds 500KB ({case_size} bytes). Use remote testcase instead."

    return processed


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    processed_data = process_code_data(args.data_path)
    with open(args.output_path, "w") as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
