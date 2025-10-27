import json
import re
from typing import Dict, Any, Callable
from openai import OpenAI
from dotenv import load_dotenv
# from problem import problem, problem2, problem3, CoT_solution, CoT_solution_2, CoT_solution_3, problem4, CoT_solution_4, problem5, CoT_solution_5
import os
load_dotenv()
openai_api_key = os.environ.get("INFINI_API_KEY")
openai_base_url = os.environ.get("INFINI_BASE_URL")
silicon_api_key = os.environ.get("SILICON_API_KEY")
silicon_base_url = os.environ.get("SILICON_BASE_URL")
def process_objective_content(content: str) -> str:
    """
    This is the function you need to implement.
    It takes the content inside <objective></objective> tags and returns a processed string.
    
    Args:
        content (str): The content extracted from <objective></objective> tags
        
    Returns:
        str: The processed content to replace the original objective content
    """
    # TODO: Implement your processing logic here
    # For now, this is just a placeholder that returns the content unchanged
    # call llm api
    client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)
    response = client.chat.completions.create(
      model="deepseek-v3-0324",
      temperature=0,
      messages=[
        {"role": "system", "content": """You are a helpful assistant that inspect objective of a step in math problem. The user will give you an objective of a step. If you think it contains result of the step, you should return 'True', otherwise, you should return 'False'. For example, if the objective is 'calculates the length of the line segment AB.', you should return 'False', because it only contains the task of the step, not the result. Instead, you should return 'True' if the objective is 'Recall the formula of the area of a triangle as 0.5 * base * height' for it contains the result of the step: '0.5 * base * height'.
         """,
         "role": "user", "content": f"{content}"}
      ]
    )
    if response.choices[0].message.content == "True":
        response = client.chat.completions.create(
            model="deepseek-v3-0324",
            temperature=0,
            messages=[
                {"role": "system", "content": """You are a helpful assistant that fix objective of a step in math problem. The user will give you an objective of a step. The objective is too aggressive, you should fix it to be more accurate. You need to remove the result of the step from the objective, for example, if the objective is 'Recall the formula of the area of a triangle as 0.5 * base * height', you should return 'Recall the formula of the area of a triangle', so that it only contains the task of the step rather than the result.
                """,
                "role": "user", "content": f"{content}"}
            ]
        )
        return response.choices[0].message.content
    else:
        return content



def extract_and_process_objectives(text: str, processor_func: Callable[[str], str]) -> str:
    """
    Extract content from <objective></objective> tags, process it, and replace it back.
    
    Args:
        text (str): Input text containing <objective></objective> tags
        processor_func (Callable): Function to process the extracted content
        
    Returns:
        str: Text with processed objective content
    """
    # Pattern to match <objective>content</objective>
    pattern = r'<objective>(.*?)</objective>'
    
    def replace_objective(match):
        objective_content = match.group(1)
        processed_content = processor_func(objective_content)
        return f'<objective>{processed_content}</objective>'
    
    # Replace all occurrences of <objective></objective> with processed content
    processed_text = re.sub(pattern, replace_objective, text, flags=re.DOTALL)
    
    return processed_text


def process_jsonl_file(input_file: str, output_file: str, processor_func: Callable[[str], str]):
    """
    Process a JSONL file by extracting and processing content from <objective></objective> tags.
    
    Args:
        input_file (str): Path to input JSONL file
        output_file (str): Path to output JSONL file
        processor_func (Callable): Function to process the extracted objective content
    """
    processed_lines = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Parse JSON line
                    data = json.loads(line)
                    
                    # Process all string values in the JSON object
                    processed_data = process_json_values(data, processor_func)
                    
                    # Add processed line to results
                    processed_lines.append(json.dumps(processed_data, ensure_ascii=False))
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line {line_num}: {e}")
                    # Keep original line if JSON parsing fails
                    processed_lines.append(line)
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    # Keep original line if processing fails
                    processed_lines.append(line)
        
        # Write processed data to output file
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for processed_line in processed_lines:
                outfile.write(processed_line + '\n')
        
        print(f"Successfully processed {len(processed_lines)} lines.")
        print(f"Output written to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except Exception as e:
        print(f"Error processing file: {e}")


def process_json_values(obj: Any, processor_func: Callable[[str], str]) -> Any:
    """
    Recursively process all string values in a JSON object to extract and process objectives.
    
    Args:
        obj: JSON object (dict, list, or primitive value)
        processor_func: Function to process the extracted objective content
        
    Returns:
        Processed JSON object
    """
    if isinstance(obj, dict):
        return {key: process_json_values(value, processor_func) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [process_json_values(item, processor_func) for item in obj]
    elif isinstance(obj, str):
        # Process string values that might contain <objective></objective> tags
        return extract_and_process_objectives(obj, processor_func)
    else:
        # Return primitive values unchanged
        return obj


def main():
    """
    Main function to demonstrate usage.
    """
    # Example usage
    input_file = "input.jsonl"  # Change this to your input file path
    output_file = "output_processed.jsonl"  # Change this to your desired output file path
    
    # Process the JSONL file
    process_jsonl_file(input_file, output_file, process_objective_content)


if __name__ == "__main__":
    # Example of how to use the script
    
    # Test with the provided example
    test_data = {
        "problem_index": 22,
        "original_problem": "In a clothing store, a sweater that was originally priced at $73 was marked down and sold for $54.75. What was the percent amount of the markdown?",
        "main_thread": "Question: In a clothing store...<objective>verifies the calculation by checking if 25% of the original price equals the markdown amount.</objective>...",
        "thread_1": "Problem: In a clothing store...<objective>confirms the steps and the final result by ensuring the sale price matches when the markdown is applied to the original price.</objective>..."
    }
    
    # Test the processing function
    print("Testing objective extraction and processing...")
    
    # Process the test data
    processed_data = process_json_values(test_data, process_objective_content)
    
    print("Original data (main_thread excerpt):")
    if len(test_data.get("main_thread", "")) > 100:
        print(test_data["main_thread"][:100] + "...")
    
    print("\nProcessed data (main_thread excerpt):")
    if len(processed_data.get("main_thread", "")) > 100:
        print(processed_data["main_thread"][:100] + "...")
    
    print("\nTo use this script with your own data:")
    print("1. Implement the 'process_objective_content' function with your logic")
    print("2. Call process_jsonl_file(input_file, output_file, process_objective_content)")
    print("3. Or modify the main() function to suit your needs")