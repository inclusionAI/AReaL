import json
import re
import os
from typing import List, Dict
import openai
from openai import OpenAI
from datetime import datetime
from dotenv import load_dotenv
# from problem import problem, problem2, problem3, CoT_solution, CoT_solution_2, CoT_solution_3, problem4, CoT_solution_4, problem5, CoT_solution_5
import os
load_dotenv()
openai_api_key = os.environ.get("INFINI_API_KEY")
openai_base_url = os.environ.get("INFINI_BASE_URL")
silicon_api_key = os.environ.get("SILICON_API_KEY")
silicon_base_url = os.environ.get("SILICON_BASE_URL")
seed = 42
# Configure your LLM API (using OpenAI as example, adjust as needed)
# openai.api_key = "your-api-key-here"
client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)
def extract_parallel_processing_sections(text: str) -> List[Dict]:
    """
    Extract all parallel processing sections from the main thread
    """
    pattern = r'<parallel_processing>(.*?)</parallel_processing>'
    matches = re.finditer(pattern, text, re.DOTALL)
    
    sections = []
    for match in matches:
        sections.append({
            'content': match.group(1).strip(),
            'start_pos': match.start(),
            'end_pos': match.end(),
            'full_match': match.group(0)
        })
    
    return sections

def generate_planning_prompt(parallel_content: str, problem_context: str = "", section_index: int = 0) -> str:
    """
    Create a prompt for the LLM to generate planning content
    """
    prompt = f"""You are an expert problem solver. Given a parallel processing section where multiple threads work on different aspects of a problem, generate a clear planning process that should come before this parallel execution.

This is parallel processing section #{section_index + 1} in the reasoning process.

The planning should analyze what needs to be done based on the parallel threads in THIS specific section.

IMPORTANT: Do NOT present it as separate threads or tasks, but as a coherent planning section that logically precedes the parallel processing. This planning should be a whole paragraph, rather than a list of tasks.

IMPORTANT: Your generated planning should be as concise as possible but still cover the necessary details for the parallel processing section presented.

IMPORTANT: Do NOT mention parallel process, just naturally integrate the planning into the context of the problem. For example, you can say "we need to consider the following aspects..." or "Okay, we shall start by ..." rather than "In this parallel processing section, we will...".

Problem context: {problem_context}

Parallel processing section #{section_index + 1}:
{parallel_content}

Generate a planning section that would logically precede THIS SPECIFIC parallel processing step. 


Planning:"""
    
    return prompt

def call_llm_for_planning(prompt: str, section_index: int = 0) -> str:
    """
    Call LLM API to generate planning content
    """
    try:
        model = "deepseek-v3-0324"

        response = client.chat.completions.create(
            model=model,
            temperature=0.3,  # Increased temperature for more variety
            seed=seed + section_index,  # Different seed for each section
            max_tokens=800,
            messages=[
                {"role": "system", "content": f"""You are an expert at generating strategic planning content for problem-solving processes. 
                You are currently working on planning section #{section_index + 1}. Make sure your planning is specific to this particular parallel processing step and different from other sections."""},
                {"role": "user", "content": prompt}
            ]
        )
        
        content = response.choices[0].message.content.strip()
        
        # Clean up any potential XML tags in the response
        content = re.sub(r'</?planning>', '', content)
        content = content.strip()
        
        return content
    
    except Exception as e:
        print(f"Error calling LLM API: {e}")
        # Fallback planning text with section number
        return f"Planning for section #{section_index + 1}: Before proceeding with this parallel processing step, let me analyze the specific problem structure and plan the approach for coordinating these particular reasoning threads."

def insert_planning_before_parallel(text: str, planning_text: str, parallel_start_pos: int) -> str:
    """
    Insert planning text before the parallel processing section
    """
    # Insert planning with proper formatting and ensure proper closing
    planning_section = f"\n<planning>\n{planning_text}\n</planning>\n"
    
    # Insert before the parallel processing section
    modified_text = text[:parallel_start_pos] + planning_section + text[parallel_start_pos:]
    
    return modified_text

def process_jsonl_with_planning(input_file: str, output_file: str):
    """
    Process JSONL file to add planning sections before parallel processing
    """
    print(f"Processing {input_file} to add planning sections...")
    
    processed_count = 0
    skipped_count = 0
    total_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            total_count += 1
            
            try:
                data = json.loads(line.strip())
                main_thread = data.get('main_thread', '')
                original_problem = data.get('original_problem', '')
                
                if not main_thread:
                    print(f"Line {line_num}: No main_thread found, skipping")
                    outfile.write(line)
                    skipped_count += 1
                    continue
                
                # Extract parallel processing sections
                parallel_sections = extract_parallel_processing_sections(main_thread)
                
                if not parallel_sections:
                    print(f"Line {line_num}: No parallel processing sections found, skipping")
                    outfile.write(line)
                    skipped_count += 1
                    continue
                
                print(f"Line {line_num}: Found {len(parallel_sections)} parallel processing sections")
                
                # Process each parallel section and collect insertions
                modified_main_thread = main_thread
                insertions = []  # Store all insertions to apply them correctly
                
                for i, section in enumerate(parallel_sections):
                    print(f"  Processing section {i + 1}/{len(parallel_sections)}")
                    print(f"  Section content preview: {section['content'][:100]}...")
                    
                    # Generate planning content with section-specific context
                    prompt = generate_planning_prompt(section['content'], original_problem, i)
                    planning_content = call_llm_for_planning(prompt, i)
                    
                    print(f"  Generated planning preview: {planning_content[:100]}...")
                    
                    # Store the insertion info
                    insertions.append({
                        'position': section['start_pos'],
                        'content': f"\n<planning>\n{planning_content}\n</planning>\n",
                        'section_index': i
                    })
                
                # Apply all insertions in reverse order to maintain correct positions
                for insertion in reversed(insertions):
                    pos = insertion['position']
                    content = insertion['content']
                    section_idx = insertion['section_index']
                    
                    print(f"  Inserting planning for section {section_idx + 1} at position {pos}")
                    modified_main_thread = modified_main_thread[:pos] + content + modified_main_thread[pos:]
                
                # Update the data with modified main_thread
                data['main_thread'] = modified_main_thread
                
                # Write modified data
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                processed_count += 1
                
                print(f"  ✅ Successfully processed line {line_num}")
                
            except json.JSONDecodeError as e:
                print(f"Line {line_num}: JSON decode error - {e}")
                outfile.write(line)  # Write original line
                skipped_count += 1
                
            except Exception as e:
                print(f"Line {line_num}: Processing error - {e}")
                outfile.write(line)  # Write original line
                skipped_count += 1
    
    print(f"\n📊 Processing Summary:")
    print(f"Total entries: {total_count}")
    print(f"Successfully processed: {processed_count}")
    print(f"Skipped/errors: {skipped_count}")
    print(f"Output saved to: {output_file}")

def process_single_entry_for_testing(text: str, problem: str = "") -> str:
    """
    Test function to process a single entry
    """
    print("Testing single entry processing...")
    
    parallel_sections = extract_parallel_processing_sections(text)
    print(f"Found {len(parallel_sections)} parallel processing sections")
    
    if not parallel_sections:
        return text
    
    modified_text = text
    offset = 0
    
    for i, section in enumerate(reversed(parallel_sections)):
        print(f"Processing section {len(parallel_sections) - i}")
        print(f"Section content preview: {section['content'][:100]}...")
        
        prompt = generate_planning_prompt(section['content'], problem)
        planning_content = call_llm_for_planning(prompt)
        
        print(f"Generated planning: {planning_content[:100]}...")
        
        insertion_pos = section['start_pos'] + offset
        modified_text = insert_planning_before_parallel(
            modified_text, 
            planning_content, 
            insertion_pos
        )
        
        planning_section_length = len(f"\n\n<planning>\n{planning_content}\n</planning>\n\n")
        offset += planning_section_length
    
    return modified_text

def main():
    """
    Main function to process JSONL files
    """
    # Configuration
    input_file = "/home/zhangzy/parallel-thinking/data_processing/add_planning_test_small.jsonl"  # Update path
    output_file = "/home/zhangzy/parallel-thinking/data_processing/output_with_planning.jsonl"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Process the file
    process_jsonl_with_planning(input_file, output_file)
    
    print("\n🎉 Processing complete!")

if __name__ == "__main__":
    # For testing with a single entry
    test_mode = False
    
    if test_mode:
        # Test with a sample text
        sample_text = """
        Question: Test problem
        Assistant: <reasoning_process>
        Let me think about this...
        
        <parallel_processing>
        <launch_threads>
        <thread id='1'>
        <task>Analyze part A</task>
        <objective>Find the solution for A</objective>
        </thread>
        </launch_threads>
        </parallel_processing>
        
        <answer>Final answer</answer>
        """
        
        result = process_single_entry_for_testing(sample_text, "Test problem")
        print("Result:")
        print(result)
    else:
        main()