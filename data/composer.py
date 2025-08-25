import json

INPUT = 'Werewolf_placeholder.jsonl'
OUTPUT = 'Werewolf_placeholder_3k.jsonl'
TARGET_LINES = 3000

# Read original examples
# with open(INPUT, 'r', encoding='utf-8') as f:
#     originals = [json.loads(line) for line in f if line.strip()]
# if not originals:
#     raise ValueError("Input file is empty or contains only blank lines")

# Write with unique IDs
with open(OUTPUT, 'w', encoding='utf-8') as out:
    total_written = 0
    while total_written < TARGET_LINES:
        new_example = {}
        new_example["id"] = total_written  # assign unique id
        new_example["prompt"] = f"{total_written}"
        out.write(json.dumps(new_example, ensure_ascii=False) + '\n')
        total_written += 1
        if total_written >= TARGET_LINES:
            break

print(f"Wrote {total_written} lines to {OUTPUT}")