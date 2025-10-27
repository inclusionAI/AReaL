import json
import re

def extract_last_objective(text):
    """Extract the last objective enclosed in <objective>...</objective> tags"""
    # Find all objectives in the text
    objectives = re.findall(r'<objective>(.*?)</objective>', text, re.DOTALL)
    if objectives:
        # Return the last objective found
        return objectives[-1].strip()
    return None

def process_jsonl_file(input_file, output_file):
    """Process JSONL file to extract objectives and modify GPT responses"""
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            line = line.strip()
            if not line:
                continue
                
            try:
                data = json.loads(line)
                
                # Check if this line has conversations
                if 'conversations' not in data:
                    # Write unchanged data to output file
                    json.dump(data, outfile, ensure_ascii=False)
                    outfile.write('\n')
                    continue
                
                # Look for user message with objective
                user_message = None
                gpt_response = None
                user_index = -1
                gpt_index = -1
                
                for i, conv in enumerate(data['conversations']):
                    if conv.get('from') == 'human' and '<objective>' in conv.get('value', ''):
                        user_message = conv['value']
                        user_index = i
                    elif conv.get('from') == 'gpt' and user_index != -1 and i == user_index + 1:
                        gpt_response = conv['value']
                        gpt_index = i
                        break
                
                # If no objective found, write unchanged data
                if not user_message:
                    json.dump(data, outfile, ensure_ascii=False)
                    outfile.write('\n')
                    continue
                
                # Extract the last objective
                last_objective = extract_last_objective(user_message)
                if not last_objective:
                    # Write unchanged data if no valid objective
                    json.dump(data, outfile, ensure_ascii=False)
                    outfile.write('\n')
                    continue
                
                # Modify GPT response if found
                if gpt_response is not None:
                    # Add objective to the start of GPT response
                    last_objective = last_objective.replace('s ',' ') # remove plural 's '
                    modified_response = f"Okay, so I need to {last_objective} {gpt_response}"
                    data['conversations'][gpt_index]['value'] = modified_response
                
                # Write modified data to output file
                json.dump(data, outfile, ensure_ascii=False)
                outfile.write('\n')
                
            except json.JSONDecodeError:
                # Write malformed JSON lines as-is (or skip them - your choice)
                outfile.write(line + '\n')
                continue
# Example usage
if __name__ == "__main__":
    input_file = "/home/admin/langzhang.zzy/inclusionAI/AReaL/parallel-thinking-main/data_processing/xml_output_new_no_first_step_parallel_large_conclusion_copy/converted_data_new_cleaned_with_parallel_tags_with_planning_conclusion_main_mixed.jsonl"
    output_file = "/home/admin/langzhang.zzy/inclusionAI/AReaL/parallel-thinking-main/data_processing/xml_output_new_no_first_step_parallel_large_conclusion_copy/converted_data_new_cleaned_with_parallel_tags_with_planning_conclusion_main_mixed_add_okay.jsonl"
    
    process_jsonl_file(input_file, output_file)
    print(f"Processing complete. Output saved to {output_file}")