# ...existing code...
import json
import os
import re
def convert_original_cot_simple(input_file, output_file):
    """
    Simple converter that directly puts original_cot as gpt response
    No filtering, no system prompt, no cleaning
    """
    print("Converting data with original_cot as direct gpt response...")
    
    converted_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            
            question = data.get('Problem', '')
            original_cot = data.get('CoT', '')
            
            if question and original_cot:
                conversation = {
                    "conversations": [
                        {
                            "from": "human",
                            "value": question
                        },
                        {
                            "from": "gpt", 
                            "value": original_cot
                        }
                    ]
                }
                converted_data.append(conversation)
    
    print(f"Prepared {len(converted_data)} training examples")
    
    # Save training data
    os.makedirs('data', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Create dataset info
    dataset_info = {
        "original_cot": {
            "file_name": os.path.basename(output_file),
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations"
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value", 
                "user_tag": "human",
                "assistant_tag": "gpt"
            }
        }
    }
    
    # Save dataset info in the same directory as the output file
    output_dir = os.path.dirname(output_file)
    dataset_info_path = os.path.join(output_dir, 'dataset_info_raw.json')
    with open(dataset_info_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    print(f"Dataset info saved to: {dataset_info_path}")
    print(f"Output saved to: {output_file}")

# ...existing code...

if __name__ == "__main__":
    print("\n" + "="*50 + "\n")
    
    # Simple original_cot converter
    print("=== Original CoT Simple Converter ===")
    convert_original_cot_simple(
        "/home/admin/langzhang.zzy/inclusionAI/AReaL/parallel-thinking-main/data_processing/original_cot_simple_large_1.jsonl",
        "/home/zhangzy/parallel-thinking/data_processing/new_dataset/original_cot_simple_large_1.jsonl"
    )
    
    # print("\n" + "="*50 + "\n")
    
    # # Keep existing converters as well
    # print("=== Assistant Filtering Converter (Reasoning Process Only) ===")
    # convert_with_assistant_filtering(
    #     "/home/zhangzy/parallel-thinking/data_processing/output_with_planning_and_conclusions_full.jsonl",
    #     "/home/zhangzy/parallel-thinking/data_processing/new_dataset/main_dialogue_clean_7150930.jsonl"
    # )
    
    # print("\n" + "="*50 + "\n")
    
    # print("=== Thread Dialogue Converter ===")
    # convert_to_thread_dialogue_format(
    #     "/home/zhangzy/parallel-thinking/data_processing/output_with_planning_and_conclusions_full.jsonl",
    #     "/home/zhangzy/parallel-thinking/data_processing/new_dataset/thread_dialogue_clean_7150930.jsonl"
    # )