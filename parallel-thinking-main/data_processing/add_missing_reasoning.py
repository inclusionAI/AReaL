import json
import re
from typing import List, Tuple, Dict

def normalize_text(text: str) -> str:
    """标准化文本，移除多余的空白字符和换行符"""
    return re.sub(r'\s+', ' ', text.strip())

def extract_content_from_main_thread(main_thread: str) -> List[str]:
    """从main_thread中提取<think>标签内的内容"""
    pattern = r'<think: type = \'[^\']*\'>(.*?)</think: type = \'[^\']*\'>'
    matches = re.findall(pattern, main_thread, re.DOTALL)
    return [match.strip() for match in matches]

def extract_content_from_thread_1(thread_1: str) -> List[str]:
    """从thread_1中提取<thread_processing>标签内的内容"""
    pattern = r'<thread_processing id = \'[^\']*\'>(.*?)</thread_processing>'
    matches = re.findall(pattern, thread_1, re.DOTALL)
    return [match.strip() for match in matches]

def extract_original_cot_content(original_cot: str) -> str:
    """提取original_CoT中<think>和</think>标签之间的内容用于匹配"""
    pattern = r'<think>(.*?)</think>'
    matches = re.findall(pattern, original_cot, re.DOTALL)
    # 将所有匹配的内容连接起来
    return '\n'.join(match.strip() for match in matches)

def find_content_in_original(content: str, original_cot: str) -> bool:
    """检查内容是否在original_CoT中存在，忽略空白字符差异"""
    # 只使用<think>标签内的内容进行匹配
    original_content = extract_original_cot_content(original_cot)
    
    normalized_content = normalize_text(content)
    normalized_original = normalize_text(original_content)
    
    # 首先尝试完整匹配
    if normalized_content in normalized_original:
        return True
    
    # 如果不存在，尝试按换行符分割后逐个查找
    lines = content.split('\n')
    for line in lines:
        normalized_line = normalize_text(line)
        if normalized_line and normalized_line in normalized_original:
            return True
    
    return False

def find_covered_positions(original_cot: str, all_contents: List[str]) -> List[Tuple[int, int]]:
    """找到在original_CoT中被覆盖的位置范围"""
    # 只使用<think>标签内的内容进行匹配
    original_content = extract_original_cot_content(original_cot)
    normalized_original = normalize_text(original_content)
    covered_positions = []
    
    for content in all_contents:
        normalized_content = normalize_text(content)
        if normalized_content:
            start_pos = normalized_original.find(normalized_content)
            if start_pos != -1:
                end_pos = start_pos + len(normalized_content)
                covered_positions.append((start_pos, end_pos))
    
    # 合并重叠的区间
    covered_positions.sort()
    merged_positions = []
    for start, end in covered_positions:
        if merged_positions and start <= merged_positions[-1][1]:
            merged_positions[-1] = (merged_positions[-1][0], max(merged_positions[-1][1], end))
        else:
            merged_positions.append((start, end))
    
    return merged_positions

def find_missing_parts(original_cot: str, main_contents: List[str], thread_1_contents: List[str]) -> List[str]:
    """找到original_CoT中未被覆盖的部分"""
    all_covered_contents = main_contents + thread_1_contents
    
    # 验证提取的内容是否在original中存在
    for i, content in enumerate(all_covered_contents):
        if not find_content_in_original(content, original_cot):
            # 尝试按行分割查找
            lines = content.split('\n')
            found_any = False
            for line in lines:
                if line.strip() and find_content_in_original(line.strip(), original_cot):
                    found_any = True
                    break
            if not found_any:
                print(f"警告: 提取的内容在original_CoT中不存在 (索引 {i}): {content[:100]}...")
    
    # 只使用<think>标签内的内容进行匹配
    original_content = extract_original_cot_content(original_cot)
    normalized_original = normalize_text(original_content)
    covered_positions = find_covered_positions(original_cot, all_covered_contents)
    
    # 找到未覆盖的部分
    missing_parts = []
    last_end = 0
    
    for start, end in covered_positions:
        if start > last_end:
            # 有未覆盖的部分
            missing_text = normalized_original[last_end:start]
            if missing_text.strip():
                missing_parts.append(missing_text.strip())
        last_end = max(last_end, end)
    
    # 检查最后一部分
    if last_end < len(normalized_original):
        missing_text = normalized_original[last_end:]
        if missing_text.strip():
            missing_parts.append(missing_text.strip())
    
    return missing_parts

def find_insertion_position(missing_part: str, original_cot: str, main_contents: List[str], thread_1_contents: List[str]) -> str:
    """确定缺失部分应该插入到main_thread还是thread_1"""
    # 只使用<think>标签内的内容进行匹配
    original_content = extract_original_cot_content(original_cot)
    
    normalized_missing = normalize_text(missing_part)
    normalized_original = normalize_text(original_content)
    
    # 找到missing_part在original中的位置
    missing_pos = normalized_original.find(normalized_missing)
    if missing_pos == -1:
        return "main_thread"  # 默认放入main_thread
    
    # 找到前后相邻的内容
    before_text = normalized_original[:missing_pos].strip()
    after_text = normalized_original[missing_pos + len(normalized_missing):].strip()
    
    # 检查相邻内容是否在main_thread或thread_1中
    main_found = False
    thread_1_found = False
    
    # 检查前后文本的最后/开始部分是否在提取的内容中
    for content in main_contents:
        normalized_content = normalize_text(content)
        if (before_text and len(before_text) > 20 and before_text[-50:] in normalized_content) or \
           (after_text and len(after_text) > 20 and after_text[:50] in normalized_content):
            main_found = True
            break
    
    for content in thread_1_contents:
        normalized_content = normalize_text(content)
        if (before_text and len(before_text) > 20 and before_text[-50:] in normalized_content) or \
           (after_text and len(after_text) > 20 and after_text[:50] in normalized_content):
            thread_1_found = True
            break
    
    if main_found and not thread_1_found:
        return "main_thread"
    elif thread_1_found and not main_found:
        return "thread_1"
    else:
        return "main_thread"  # 默认放入main_thread

def insert_missing_content(target_thread: str, missing_content: str, thread_type: str) -> str:
    """将缺失内容插入到指定线程中"""
    if thread_type == "main_thread":
        # 在最后一个</think>标签前插入
        last_think_start = target_thread.rfind('<think: type =')
        last_think_end = target_thread.rfind('</think: type =')
        if last_think_start != -1 and last_think_end != -1:
            new_content = f"\n\n{missing_content}"
            return target_thread[:last_think_end] + new_content + target_thread[last_think_end:]
        # 如果没找到合适位置，创建新的think块
        new_block = f"\n\n<think: type = 'missing_content'>\n{missing_content}\n</think: type = 'missing_content'>"
        return target_thread + new_block
    
    else:  # thread_1
        # 在最后一个</thread_processing>标签前插入
        last_thread_start = target_thread.rfind('<thread_processing: id=')
        last_thread_end = target_thread.rfind('</thread_processing: id=')
        if last_thread_start != -1 and last_thread_end != -1:
            new_content = f"\n\n{missing_content}"
            return target_thread[:last_thread_end] + new_content + target_thread[last_thread_end:]
        # 如果没找到合适位置，创建新的thread_processing块
        new_block = f"\n\n<thread_processing id= 'missing_content'>\n{missing_content}\n</thread_processing id= 'missing_content'>"
        return target_thread + new_block

def process_single_record(record: Dict, record_index: int) -> Dict:
    """处理单条记录"""
    print(f"\n=== 处理记录 {record_index + 1} ===")
    
    original_cot = record['original_CoT']
    main_thread = record['main_thread']
    thread_1 = record['thread_1']
    
    # 提取内容
    main_contents = extract_content_from_main_thread(main_thread)
    thread_1_contents = extract_content_from_thread_1(thread_1)
    
    print(f"从 main_thread 提取到 {len(main_contents)} 个内容块")
    print(f"从 thread_1 提取到 {len(thread_1_contents)} 个内容块")
    
    # 找到缺失部分
    missing_parts = find_missing_parts(original_cot, main_contents, thread_1_contents)
    
    print(f"发现 {len(missing_parts)} 个未覆盖的部分:")
    for i, missing_part in enumerate(missing_parts):
        print(f"  未覆盖部分 {i + 1}: {missing_part[:100]}{'...' if len(missing_part) > 100 else ''}")
    
    # 将缺失部分插入适当位置
    new_main_thread = main_thread
    new_thread_1 = thread_1
    
    for i, missing_part in enumerate(missing_parts):
        if missing_part.strip():  # 忽略空白内容
            target = find_insertion_position(missing_part, original_cot, main_contents, thread_1_contents)
            print(f"  未覆盖部分 {i + 1} 将插入到: {target}")
            
            if target == "main_thread":
                new_main_thread = insert_missing_content(new_main_thread, missing_part, "main_thread")
            else:
                new_thread_1 = insert_missing_content(new_thread_1, missing_part, "thread_1")
    
    return {
        'original_CoT': original_cot,
        'main_thread': new_main_thread,
        'thread_1': new_thread_1
    }

def process_jsonl_file(input_file: str, output_file: str):
    """处理JSONL文件"""
    processed_count = 0
    error_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                record = json.loads(line.strip())
                processed_record = process_single_record(record, line_num - 1)
                outfile.write(json.dumps(processed_record, ensure_ascii=False) + '\n')
                processed_count += 1
                
            except Exception as e:
                print(f"处理第 {line_num} 行时出错: {e}")
                error_count += 1
                # 写入原始记录
                outfile.write(line)
    
    print(f"\n处理完成！成功处理 {processed_count} 条记录，错误 {error_count} 条")

if __name__ == "__main__":
    # 使用示例
    input_file = "/home/zhangzy/parallel-thinking/data_processing/cleaned_combined_parallel_thinking_data.jsonl"  # 输入文件路径
    output_file = "output_data.jsonl"  # 输出文件路径
    
    print("开始处理数据恢复...")
    process_jsonl_file(input_file, output_file)
    print("数据恢复完成！")