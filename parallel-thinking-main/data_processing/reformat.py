import json
import re

def fix_thread_content(content):
    """Fix a single thread content by moving text after ': The model' from task to objective"""
    if not content:
        return content
    
    # Pattern to match <task>...: The model...<objective></objective>
    pattern = r'(<task>.*?): The model (.*?)</task>\s*<objective>\s*</objective>'
    
    def replace_func(match):
        task_part = match.group(1)  # Everything before ": The model"
        objective_part = match.group(2)  # Everything after ": The model"
        
        # Return the fixed format
        return f"{task_part}\n</task>\n<objective>\n {objective_part}</objective>"
    
    # Apply the fix
    fixed_content = re.sub(pattern, replace_func, content, flags=re.DOTALL)
    
    return fixed_content

def has_empty_thread_processing(line):
    """Check if the line contains empty thread_processing pattern"""
    # Pattern to match both literal newlines and escaped newlines in JSON
    patterns = [
        r'<thread_processing id=\'\d+\'>\s*</thread_processing>',  # Literal whitespace
        r'<thread_processing id=\'\d+\'>\\n</thread_processing>',  # Escaped newline in JSON
        r'<thread_processing id=\'\d+\'>\n</thread_processing>',   # Actual newline
    ]
    
    for pattern in patterns:
        if re.search(pattern, line):
            return True
    return False

def fix_jsonl_file(input_file, output_file):
    """Fix the entire JSONL file and skip lines with empty thread_processing"""
    fixed_count = 0
    total_count = 0
    skipped_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            original_line = line
            line = line.strip()
            if not line:
                continue
            
            # Check if this line should be skipped (contains empty thread_processing)
            if has_empty_thread_processing(line):
                skipped_count += 1
                print(f"Skipped line {line_num} (contains empty thread_processing)")
                continue
                
            try:
                # Parse JSON
                data = json.loads(line)
                total_count += 1
                
                # Check and fix main_thread
                if 'main_thread' in data and data['main_thread']:
                    original_main = data['main_thread']
                    fixed_main = fix_thread_content(data['main_thread'])
                    if fixed_main != original_main:
                        data['main_thread'] = fixed_main
                        fixed_count += 1
                        print(f"Fixed main_thread in line {line_num}")
                
                # Check and fix thread_1
                if 'thread_1' in data and data['thread_1']:
                    original_thread1 = data['thread_1']
                    fixed_thread1 = fix_thread_content(data['thread_1'])
                    if fixed_thread1 != original_thread1:
                        data['thread_1'] = fixed_thread1
                        fixed_count += 1
                        print(f"Fixed thread_1 in line {line_num}")
                
                # Write the (potentially fixed) data
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_num}: {e}")
                # Write the original line if it can't be parsed
                outfile.write(original_line)
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                # Write the original line if there's any other error
                outfile.write(original_line)
    
    print(f"\nProcessing complete!")
    print(f"Total lines read: {line_num}")
    print(f"Lines skipped (empty thread_processing): {skipped_count}")
    print(f"Records processed: {total_count}")
    print(f"Records with fixes applied: {fixed_count}")
    print(f"Output saved to: {output_file}")

def preview_fixes(input_file, max_previews=3):
    """Preview what fixes would be applied without actually fixing the file"""
    preview_count = 0
    skip_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile:
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue
            
            # Check if this line would be skipped
            if has_empty_thread_processing(line):
                skip_count += 1
                if skip_count <= 3:  # Show first few skipped lines
                    print(f"\n=== WOULD SKIP LINE {line_num} ===")
                    print("Reason: Contains empty thread_processing pattern")
                    print(f"Line preview: {line[:200]}...")
                continue
            
            try:
                data = json.loads(line)
                
                # Check main_thread
                if 'main_thread' in data and data['main_thread']:
                    original = data['main_thread']
                    fixed = fix_thread_content(data['main_thread'])
                    if fixed != original:
                        print(f"\n=== PREVIEW FIX {preview_count + 1} (Line {line_num}, main_thread) ===")
                        print("BEFORE:")
                        print(original[:500] + "..." if len(original) > 500 else original)
                        print("\nAFTER:")
                        print(fixed[:500] + "..." if len(fixed) > 500 else fixed)
                        preview_count += 1
                        if preview_count >= max_previews:
                            break
                
                # Check thread_1
                if 'thread_1' in data and data['thread_1']:
                    original = data['thread_1']
                    fixed = fix_thread_content(data['thread_1'])
                    if fixed != original:
                        print(f"\n=== PREVIEW FIX {preview_count + 1} (Line {line_num}, thread_1) ===")
                        print("BEFORE:")
                        print(original[:500] + "..." if len(original) > 500 else original)
                        print("\nAFTER:")
                        print(fixed[:500] + "..." if len(fixed) > 500 else fixed)
                        preview_count += 1
                        if preview_count >= max_previews:
                            break
                            
            except json.JSONDecodeError:
                continue
            except Exception:
                continue
    
    print(f"\nPreview summary:")
    print(f"Lines that would be skipped: {skip_count}")
    print(f"Fixes that would be applied: {preview_count}")

if __name__ == "__main__":
    # Configuration
    input_file = "/home/zhangzy/parallel-thinking/data_processing/output_threads_cleaned_new.jsonl"
    output_file = "output_threads_cleaned_new_format_correct_no_empty.jsonl"
    
    # First, preview what changes would be made
    print("=== PREVIEW MODE ===")
    print("Showing what fixes would be applied...")
    preview_fixes(input_file, max_previews=3)
    
    # Ask for confirmation
    print("\n=== CONFIRMATION ===")
    confirm = input("Do you want to proceed with fixing the file? (y/n): ")
    
    if confirm.lower() in ['y', 'yes']:
        # Apply the fixes
        fix_jsonl_file(input_file, output_file)
    else:
        print("Operation cancelled.")