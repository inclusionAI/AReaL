import json

input_file = "output.jsonl"
output_file = "filtered_output.jsonl"

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        try:
            obj = json.loads(line)
            text = obj.get("text", "")
            if "<Goal>\n</Goal>" in text:
                continue  # Skip this line
            outfile.write(json.dumps(obj, ensure_ascii=False) + "\n")
        except Exception as e:
            # Optionally handle malformed lines
            continue