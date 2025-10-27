import json
import os
import re

def contains_the_model(content):
    """
    Check if content contains ": The model" anywhere.
    
    Args:
        content (str): The content to check
        
    Returns:
        bool: True if ": The model" is found, False otherwise
    """
    if not content or content.startswith("["):
        # Skip if content is empty or is an error message like "[File not found: ...]"
        return False
    
    return ": The model" in content

def filter_jsonl_file(input_file, output_file):
    """
    Filter the JSONL file by removing entire entries that contain ": The model" 
    in their main_thread or thread_1 content.
    
    Args:
        input_file (str): Path to the input JSONL file
        output_file (str): Path to the output filtered JSONL file
    """
    
    total_stats = {
        'entries_processed': 0,
        'entries_kept': 0,
        'entries_removed': 0,
        'removed_due_to_main_thread': 0,
        'removed_due_to_thread_1': 0,
        'removed_due_to_both': 0
    }
    
    print(f"Filtering JSONL file: {input_file}")
    print(f"Output file: {output_file}")
    print("-" * 50)
    
    try:
        with open(input_file, 'r', encoding='utf-8') as in_f, \
             open(output_file, 'w', encoding='utf-8') as out_f:
            
            for line_num, line in enumerate(in_f, 1):
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line.strip())
                    total_stats['entries_processed'] += 1
                    
                    # Check if main_thread contains ": The model"
                    main_has_model = False
                    if 'main_thread' in data:
                        main_has_model = contains_the_model(data['main_thread'])
                    
                    # Check if thread_1 contains ": The model"
                    thread_has_model = False
                    if 'thread_1' in data:
                        thread_has_model = contains_the_model(data['thread_1'])
                    
                    # If either contains ": The model", skip this entire entry
                    if main_has_model or thread_has_model:
                        total_stats['entries_removed'] += 1
                        
                        # Track which field(s) caused the removal
                        if main_has_model and thread_has_model:
                            total_stats['removed_due_to_both'] += 1
                            print(f"Entry {data.get('problem_index', line_num)}: Removed (both main_thread and thread_1 contain ': The model')")
                        elif main_has_model:
                            total_stats['removed_due_to_main_thread'] += 1
                            print(f"Entry {data.get('problem_index', line_num)}: Removed (main_thread contains ': The model')")
                        else:
                            total_stats['removed_due_to_thread_1'] += 1
                            print(f"Entry {data.get('problem_index', line_num)}: Removed (thread_1 contains ': The model')")
                        
                        continue  # Skip writing this entry to output
                    
                    # If no ": The model" found, keep this entry
                    total_stats['entries_kept'] += 1
                    json.dump(data, out_f, ensure_ascii=False)
                    out_f.write('\n')
                    
                    if total_stats['entries_processed'] % 100 == 0:
                        print(f"Processed {total_stats['entries_processed']} entries... (kept: {total_stats['entries_kept']}, removed: {total_stats['entries_removed']})")
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON at line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"Error processing entry at line {line_num}: {e}")
                    continue
        
        print(f"\n=== FILTERING SUMMARY ===")
        print(f"Total entries processed: {total_stats['entries_processed']}")
        print(f"Entries kept: {total_stats['entries_kept']}")
        print(f"Entries removed: {total_stats['entries_removed']}")
        print(f"  - Removed due to main_thread: {total_stats['removed_due_to_main_thread']}")
        print(f"  - Removed due to thread_1: {total_stats['removed_due_to_thread_1']}")
        print(f"  - Removed due to both: {total_stats['removed_due_to_both']}")
        print(f"Output saved to: {output_file}")
        
        # Calculate percentages
        if total_stats['entries_processed'] > 0:
            kept_pct = (total_stats['entries_kept'] / total_stats['entries_processed']) * 100
            removed_pct = (total_stats['entries_removed'] / total_stats['entries_processed']) * 100
            print(f"Percentage kept: {kept_pct:.1f}%")
            print(f"Percentage removed: {removed_pct:.1f}%")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
        return
    except Exception as e:
        print(f"Error processing file: {e}")
        return

def main():
    # File paths
    input_file = "combined_parallel_thinking_data.jsonl"
    output_file = "filtered_parallel_thinking_data.jsonl"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        print("Please make sure the combined JSONL file exists")
        return
    
    print(f"Filtering entries containing ': The model' from JSONL file...")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print("=" * 60)
    
    filter_jsonl_file(input_file, output_file)
    
    print(f"\nFiltering completed!")
    print(f"Filtered dataset saved as: {output_file}")

if __name__ == "__main__":
    main()