import json

def process_jsonl_file(input_file, output_file):
    """
    Process JSONL file to move human's content after '<think>\n' to the start of GPT response.
    """
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            line = line.strip()
            if not line:
                continue
                
            try:
                # Parse JSON line
                data = json.loads(line)
                
                # Process conversations
                if 'conversations' in data:
                    conversations = data['conversations']
                    
                    # Find human and gpt messages
                    for i in range(len(conversations) - 1):
                        current_msg = conversations[i]
                        next_msg = conversations[i + 1]
                        
                        # Check if current is human and next is gpt/assistant
                        if (current_msg.get('from') == 'human' and 
                            next_msg.get('from') in ['gpt', 'assistant']):
                            
                            human_content = current_msg.get('value', '')
                            gpt_content = next_msg.get('value', '')
                            
                            # Find <think>\n in human content
                            think_token = '<think>\n'
                            if think_token in human_content:
                                # Split human content at <think>\n
                                parts = human_content.split(think_token, 1)
                                before_think = parts[0]
                                after_think = think_token + parts[1] if len(parts) > 1 else ''
                                
                                # Update human content (remove after <think>\n)
                                current_msg['value'] = before_think + "<think>"
                                
                                # Update gpt content (prepend after <think>\n)
                                if after_think:
                                    next_msg['value'] = after_think.replace("<think>", "").strip() + gpt_content
                
                # Write processed line to output
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON line: {e}")
                # Write original line if parsing fails
                outfile.write(line + '\n')

# Usage
if __name__ == "__main__":
    input_file = "/home/zhangzy/parallel-thinking/SFT/LLaMA-Factory/data/mixed_dataset_end_token_force add_820914.jsonl"  # Replace with your input file path
    output_file = "/home/zhangzy/parallel-thinking/SFT/LLaMA-Factory/data/mixed_dataset_end_token_force added_reasoning_back.jsonl"  # Replace with your output file path
    
    process_jsonl_file(input_file, output_file)
    print(f"Processing complete. Output saved to {output_file}")