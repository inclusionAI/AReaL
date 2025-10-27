import json

def read_jsonl_and_print_specific_line(file_path, target_line_num):
    """
    Read a JSONL file and print the main_thread value of a specific line
    
    Args:
        file_path (str): Path to the JSONL file
        target_line_num (int): Line number to print (1-indexed)
    """
    try:
        # Read JSONL file line by line
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                if line_num == target_line_num:
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            # Parse the line as JSON
                            data = json.loads(line)
                            
                            # Print main_thread value if it exists
                            if 'main_thread' in data:
                                print(f"Entry {line_num} main_thread:")
                                print(data['thread_1'])
                            else:
                                print(f"Entry {line_num}: No 'main_thread' key found")
                                
                        except json.JSONDecodeError as e:
                            print(f"Error parsing line {line_num}: {e}")
                    else:
                        print(f"Line {line_num} is empty")
                    return  # Exit after finding the target line
                        
    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"Error reading file: {e}")
    
    print(f"Line {target_line_num} not found in file")

# Main execution
if __name__ == "__main__":
    # Replace with your JSONL file path
    jsonl_file_path = "/home/zhangzy/parallel-thinking/data_processing/combined_parallel_thinking_data.jsonl"
    
    # Set which line you want to print (1-indexed)
    target_line = 514 # Change this to the line number you want
    
    print(f"Reading line {target_line} from JSONL file:")
    read_jsonl_and_print_specific_line(jsonl_file_path, target_line)
