import json

input_path = "MathVision_gemini.jsonl"      # 原文件路径
output_path = "MathVision_gemini_clean.jsonl" # 新文件路径

removed_count = 0
kept_count = 0

with open(input_path, "r", encoding="utf-8") as f_in, \
     open(output_path, "w", encoding="utf-8") as f_out:
    
    for line in f_in:
        line = line.strip()
        if not line: continue # 跳过空行
        
        try:
            data = json.loads(line)
            # 核心判断逻辑：如果 len_answer 不为 0，则保留
            if data.get("len_answer", 0) != 0:
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                kept_count += 1
            else:
                removed_count += 1
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON line: {line[:50]}...")

print(f"完成。保留了 {kept_count} 行，移除了 {removed_count} 行。")
print(f"新文件保存为: {output_path}")