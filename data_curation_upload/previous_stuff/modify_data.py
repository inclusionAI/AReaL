import json

def modify_jsonl_with_system_prompt(input_file, output_file):
    """
    Add a system prompt to each conversation in a JSONL file.
    
    Args:
        input_file: Path to the input JSONL file
        output_file: Path to the output JSONL file
    """
    system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            # Parse the JSON line
            data = json.loads(line.strip())
            
            # Check if 'conversations' key exists
            if 'conversations' in data:
                # Create new conversation list with system prompt at the beginning
                new_conversations = [
                    {
                        "from": "system",
                        "value": system_prompt
                    }
                ]
                # Add existing conversations
                new_conversations.extend(data['conversations'])
                
                # Update the conversations
                data['conversations'] = new_conversations
            
            # Write the modified line to output file
            outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print(f"Modified JSONL file saved to: {output_file}")

if __name__ == "__main__":
    # Example usage
    input_file = "example.jsonl"
    output_file = "example_modified.jsonl"
    
    modify_jsonl_with_system_prompt(input_file, output_file)
