import json
import re
from typing import List, Dict

def extract_conversations(text: str) -> List[Dict[str, str]]:
    """
    Extract conversation pairs from a single line.
    
    For each line:
    1. First conversation: input = content before <Think>, output = content after <Think> (including the token)
    2. For each <Parallel> stage, for each <Path>, create a conversation where:
       - input = content before </Goal> in that stage (including </Goal>) + "\n<Path>\ni. " where i is the path index
       - output = content in that path excluding the index, including </Path>
    """
    conversations = []
    
    # Find the position of <Think>
    think_match = re.search(r'<Think>', text, re.IGNORECASE)
    
    if think_match:
        # First conversation: before <Think> -> after <Think> (including <Think>)
        input_text = text[:think_match.start()].strip()
        output_text = text[think_match.start():].strip()
        
        conversations.append({
            "input": input_text,
            "output": output_text
        })
    
    # Find all <Parallel>...</Parallel> blocks
    parallel_pattern = r'<Parallel>(.*?)</Parallel>'
    parallel_blocks = re.finditer(parallel_pattern, text, re.DOTALL | re.IGNORECASE)
    
    for parallel_block in parallel_blocks:
        parallel_content = parallel_block.group(1)
        parallel_start = parallel_block.start()
        
        # Extract content up to and including </Goal> within this parallel block
        goal_match = re.search(r'(.*?</Goal>)', parallel_content, re.DOTALL | re.IGNORECASE)
        
        if goal_match:
            # Get the position of </Goal> in the original text
            goal_end_in_parallel = goal_match.end()
            goal_end_in_text = parallel_start + len('<Parallel>') + goal_end_in_parallel
            
            # Content from the beginning of the text up to and including </Goal> of this stage
            content_before_goal = text[:goal_end_in_text].strip()
            
            # Find all <Path>...</Path> within this parallel block
            path_pattern = r'<Path>\s*\d+\.\s*(.*?)</Path>'
            paths = re.finditer(path_pattern, parallel_content, re.DOTALL | re.IGNORECASE)
            
            path_index = 1
            for path_match in paths:
                # Input: all content from start to </Goal> + "\n<Path>\ni. " where i is the path index
                input_text = f"{content_before_goal}\n<Path>\n{path_index}. "
                
                # Output: content in the path excluding the index, including </Path>
                path_content = path_match.group(1).strip()
                output_text = f"{path_content}\n</Path>"
                
                conversations.append({
                    "input": input_text,
                    "output": output_text
                })
                
                path_index += 1
    
    return conversations


def process_jsonl(input_file: str, output_file: str):
    """
    Process the input JSONL file and create conversation pairs.
    Each line in the output will be a conversation with 'input' and 'output' keys.
    """
    total_lines = 0
    total_conversations = 0
    
    with open(input_file, 'r', encoding='utf-8') as in_f, \
         open(output_file, 'w', encoding='utf-8') as out_f:
        
        for line_num, line in enumerate(in_f, 1):
            if line_num % 100 == 0:
                print(f"Processing line {line_num}...")
            
            try:
                data = json.loads(line)
                text = data.get('text', '')
                
                # Extract conversations from this line
                conversations = extract_conversations(text)
                
                # Write each conversation as a separate line
                for conv in conversations:
                    out_f.write(json.dumps(conv, ensure_ascii=False) + '\n')
                    total_conversations += 1
                
                total_lines += 1
                
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON at line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                continue
    
    print(f"\nProcessing complete!")
    print(f"Total input lines processed: {total_lines}")
    print(f"Total conversations created: {total_conversations}")
    print(f"Average conversations per line: {total_conversations / total_lines:.2f}" if total_lines > 0 else "N/A")


if __name__ == "__main__":
    input_file = "filtered_output.jsonl"
    output_file = "conversations_output.jsonl"
    
    print(f"Converting {input_file} to conversation pairs...")
    print(f"Output will be saved to {output_file}\n")
    
    process_jsonl(input_file, output_file)
