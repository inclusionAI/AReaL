import json
import re

def clean_jsonl_file(input_file, output_file):
    """
    Clean JSONL file by removing the \n:...\n pattern from all text fields
    """
    def clean_text(text):
        if isinstance(text, str):
            return re.sub(r'\n:.*?\n', '\n', text, flags=re.DOTALL)
        return text
    
    def clean_dict(obj):
        if isinstance(obj, dict):
            return {k: clean_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_dict(item) for item in obj]
        elif isinstance(obj, str):
            return clean_text(obj)
        else:
            return obj
    
    cleaned_lines = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                cleaned_data = clean_dict(data)
                cleaned_lines.append(json.dumps(cleaned_data, ensure_ascii=False))
            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} is not valid JSON: {e}")
                continue
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in cleaned_lines:
            f.write(line + '\n')
    
    print(f"Cleaned {len(cleaned_lines)} lines")
    print(f"Output saved to: {output_file}")

# Usage for JSONL
clean_jsonl_file("final_dataset.jsonl", "output_cleaned.jsonl")