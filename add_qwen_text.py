import json
import sys

def add_qwen_text(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["qwen_text"] = record["input"] + record["output"]
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Done. Output written to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 add_qwen_text.py <input.jsonl> <output.jsonl>")
        sys.exit(1)
    add_qwen_text(sys.argv[1], sys.argv[2])
