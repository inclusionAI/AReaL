import json
import re
from typing import Set, List, Dict, Tuple

def extract_tasks_with_ids_from_text(text: str) -> Dict[str, str]:
    """
    Extract all task contents with their thread IDs from thread definitions in the given text.
    Returns a dict mapping normalized task -> thread_id.
    """
    # Pattern to match thread definitions with task content and ID
    thread_pattern = r'<thread id=\'([^\']*?)\'>\s*<task>\s*(.*?)\s*</task>'
    
    task_to_id = {}
    matches = re.findall(thread_pattern, text, re.DOTALL)
    
    for thread_id, task in matches:
        # Clean up the task text (remove extra whitespace, normalize)
        cleaned_task = ' '.join(task.strip().split())
        task_to_id[cleaned_task] = thread_id
    
    return task_to_id

def extract_tasks_from_text(text: str) -> Set[str]:
    """
    Extract all task contents from thread definitions in the given text.
    Returns a set of task strings for comparison.
    """
    # Pattern to match thread definitions with task content
    thread_pattern = r'<thread id=\'[^\']*\'>\s*<task>\s*(.*?)\s*</task>'
    
    tasks = set()
    matches = re.findall(thread_pattern, text, re.DOTALL)
    
    for task in matches:
        # Clean up the task text (remove extra whitespace, normalize)
        cleaned_task = ' '.join(task.strip().split())
        tasks.add(cleaned_task)
    
    return tasks

def update_thread_id_in_block(block: str, old_id: str, new_id: str) -> str:
    """
    Update all occurrences of thread ID in a thread block.
    """
    # Replace thread id in opening tag
    block = re.sub(r'<thread id=\'[^\']*?\'>', f'<thread id=\'{new_id}\'>', block)
    
    # Replace thread id in thread_processing tag
    block = re.sub(r'<thread_processing id\s*=\s*\'[^\']*?\'>', f'<thread_processing id=\'{new_id}\'>', block)
    
    # Replace thread id in thread_result tag
    block = re.sub(r'<thread_result id=\'[^\']*?\'>', f'<thread_result id=\'{new_id}\'>', block)
    
    return block

def remove_undefined_threads_from_thread1(thread1_content: str, main_thread_task_to_id: Dict[str, str]) -> str:
    """
    Remove threads from thread1_content that are not defined in main_thread_tasks,
    and update thread IDs to match those from main_thread.
    """
    # Find all complete thread blocks in thread1
    thread_block_pattern = r'(<thread id=\'([^\']*?)\'>\s*<task>\s*(.*?)\s*</task>\s*<objective>\s*.*?\s*</objective>\s*</thread>\s*<thread_processing id\s*=\s*\'[^\']*?\'>\s*.*?\s*</thread_processing>\s*<thread_result id=\'[^\']*?\'>\s*.*?\s*</thread_result>)'
    
    matches = re.findall(thread_block_pattern, thread1_content, re.DOTALL)
    
    filtered_blocks = []
    
    for full_block, old_thread_id, task in matches:
        # Clean up the task text
        cleaned_task = ' '.join(task.strip().split())
        
        # Check if this task exists in main_thread
        if cleaned_task in main_thread_task_to_id:
            # Get the correct thread ID from main_thread
            correct_thread_id = main_thread_task_to_id[cleaned_task]
            
            # Update the thread ID in the block
            updated_block = update_thread_id_in_block(full_block, old_thread_id, correct_thread_id)
            
            filtered_blocks.append(updated_block)
            print(f"✅ Keeping thread with task: '{cleaned_task}' (ID: {old_thread_id} -> {correct_thread_id})")
        else:
            print(f"❌ Removing thread with task: '{cleaned_task}' (ID: {old_thread_id})")
    
    # Reconstruct the content with filtered blocks
    if filtered_blocks:
        # Extract the header (everything before first thread block)
        header_match = re.search(r'^(.*?)(?=<thread id=)', thread1_content, re.DOTALL)
        header = header_match.group(1) if header_match else ""
        
        # Combine header with filtered blocks
        result = header + '\n'.join(filtered_blocks)
    else:
        # If no blocks remain, return just the header
        header_match = re.search(r'^(.*?)(?=<thread id=)', thread1_content, re.DOTALL)
        result = header_match.group(1) if header_match else thread1_content
    
    return result

def clean_dataset(input_file: str, output_file: str):
    """
    Clean the JSONL dataset by removing threads from thread_1 that are not defined in main_thread,
    and align thread IDs with those from main_thread.
    """
    print(f"🔄 Processing dataset: {input_file}")
    
    cleaned_data = []
    removed_count = 0
    kept_count = 0
    id_updates_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                main_thread = data.get('main_thread', '')
                thread_1 = data.get('thread_1', '')
                
                if not main_thread or not thread_1:
                    print(f"⚠️ Line {line_num}: Missing main_thread or thread_1, keeping as-is")
                    cleaned_data.append(data)
                    continue
                
                # Extract tasks with IDs from main_thread
                main_thread_task_to_id = extract_tasks_with_ids_from_text(main_thread)
                print(f"\n📝 Line {line_num}: Found {len(main_thread_task_to_id)} tasks in main_thread:")
                for task, thread_id in sorted(main_thread_task_to_id.items()):
                    print(f"  - '{task}' (ID: {thread_id})")
                
                # Clean thread_1 and align IDs
                original_thread1_tasks = extract_tasks_from_text(thread_1)
                print(f"🧵 Original thread_1 has {len(original_thread1_tasks)} tasks")
                
                cleaned_thread_1 = remove_undefined_threads_from_thread1(thread_1, main_thread_task_to_id)
                
                # Count changes
                cleaned_thread1_tasks = extract_tasks_from_text(cleaned_thread_1)
                removed_this_line = len(original_thread1_tasks) - len(cleaned_thread1_tasks)
                removed_count += removed_this_line
                kept_count += len(cleaned_thread1_tasks)
                
                # Count ID updates (threads that were kept)
                id_updates_count += len(cleaned_thread1_tasks)
                
                print(f"📊 Removed {removed_this_line} threads, kept {len(cleaned_thread1_tasks)} threads with aligned IDs")
                
                # Update data
                data['thread_1'] = cleaned_thread_1
                cleaned_data.append(data)
                
            except json.JSONDecodeError as e:
                print(f"❌ Error parsing line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"❌ Error processing line {line_num}: {e}")
                continue
    
    # Save cleaned data
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in cleaned_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n🎉 Dataset cleaning complete!")
    print(f"📊 Summary:")
    print(f"  - Processed {len(cleaned_data)} entries")
    print(f"  - Removed {removed_count} threads total")
    print(f"  - Kept {kept_count} threads total")
    print(f"  - Updated {id_updates_count} thread IDs to match main_thread")
    print(f"  - Output saved to: {output_file}")

def main():
    """
    Main function to run the dataset cleaning.
    """
    input_file = input("Enter input JSONL file path: ").strip()
    if not input_file:
        input_file = "output_cleaned.jsonl"  # Default
    
    output_file = input("Enter output JSONL file path: ").strip()
    if not output_file:
        output_file = "output_threads_cleaned.jsonl"  # Default
    
    try:
        clean_dataset(input_file, output_file)
    except FileNotFoundError:
        print(f"❌ Error: Input file '{input_file}' not found.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Example usage with hardcoded paths
    input_file = "/home/zhangzy/parallel-thinking/data_processing/new_760936.jsonl"
    output_file = "/home/zhangzy/parallel-thinking/data_processing/new_761040_cleaned.jsonl"
    
    clean_dataset(input_file, output_file)
    
    # Or use interactive mode
    # main()