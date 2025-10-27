# import json
# import re

# def parse_jsonl_file(filename):
#     """
#     Parse a JSONL file and extract problem data.
#     Handles multi-line JSON objects and triple-quoted strings with LaTeX.
    
#     Args:
#         filename (str): Path to the JSONL file
        
#     Returns:
#         list: List of dictionaries containing problem data
#     """
#     problems = []
    
#     try:
#         with open(filename, 'r', encoding='utf-8') as file:
#             content = file.read()
            
#         # Try to parse as complete multi-line JSON objects
#         problems = parse_multiline_json(content)
        
#         # Print basic info about each problem
#         for i, problem_data in enumerate(problems, 1):
#             print(f"Problem {i}:")
#             if 'problem' in problem_data:
#                 print(f"  Question: {problem_data['problem'][:100]}...")
#             elif 'Problem' in problem_data:
#                 print(f"  Question: {problem_data['Problem'][:100]}...")
            
#             if 'answer' in problem_data:
#                 print(f"  Answer: {problem_data['answer']}")
#             elif 'Answer' in problem_data:
#                 print(f"  Answer: {problem_data['Answer'][:50]}...")
            
#             print("-" * 50)
                    
#     except FileNotFoundError:
#         print(f"File '{filename}' not found.")
#         return []
#     except Exception as e:
#         print(f"Error reading file: {e}")
#         return []
    
#     return problems

# def fix_triple_quotes(content):
#     """
#     Convert triple quotes to valid JSON format, preserving LaTeX escapes.
    
#     Args:
#         content (str): Raw content with potential triple quotes
        
#     Returns:
#         str: Content with triple quotes converted to valid JSON
#     """
#     def replace_triple_quotes(match):
#         inner_content = match.group(1)
        
#         # Escape quotes and control characters for JSON, but leave backslashes as-is
#         inner_content = inner_content.replace('\\', '\\\\')  # Escape backslashes first
#         inner_content = inner_content.replace('"', '\\"')    # Escape quotes
#         inner_content = inner_content.replace('\n', '\\n')   # Escape newlines
#         inner_content = inner_content.replace('\r', '\\r')   # Escape carriage returns
#         inner_content = inner_content.replace('\t', '\\t')   # Escape tabs
        
#         return f'"{inner_content}"'
    
#     # Pattern to find triple-quoted strings
#     pattern = r'"""(.*?)"""'
    
#     # Replace triple quotes with properly formatted JSON strings
#     fixed_content = re.sub(pattern, replace_triple_quotes, content, flags=re.DOTALL)
    
#     return fixed_content

# def parse_multiline_json(content):
#     """
#     Parse content that may contain multi-line JSON objects with triple quotes.
    
#     Args:
#         content (str): File content as string
        
#     Returns:
#         list: List of parsed JSON objects
#     """
#     problems = []
    
#     # First, fix the triple quotes
#     try:
#         fixed_content = fix_triple_quotes(content)
#     except Exception as e:
#         print(f"Error fixing triple quotes: {e}")
#         return problems
    
#     # Split into potential JSON objects
#     json_objects = extract_json_objects_robust(fixed_content)
    
#     for i, json_str in enumerate(json_objects, 1):
#         try:
#             problem_data = json.loads(json_str)
#             problems.append(problem_data)
#         except json.JSONDecodeError as e:
#             print(f"Error parsing JSON object {i}: {e}")
#             print(f"JSON content: {json_str[:200]}...")
            
#             # Try to fix common issues and retry
#             try:
#                 fixed_json = fix_json_issues(json_str)
#                 problem_data = json.loads(fixed_json)
#                 problems.append(problem_data)
#                 print(f"  -> Fixed and parsed successfully!")
#             except:
#                 print(f"  -> Could not fix JSON object {i}")
    
#     return problems

# def extract_json_objects_robust(content):
#     """
#     Extract JSON objects from content using robust brace matching.
    
#     Args:
#         content (str): Content containing JSON objects
        
#     Returns:
#         list: List of JSON strings
#     """
#     json_objects = []
#     brace_count = 0
#     current_object = ""
#     in_string = False
#     escape_next = False
#     i = 0
    
#     while i < len(content):
#         char = content[i]
        
#         if escape_next:
#             current_object += char
#             escape_next = False
#             i += 1
#             continue
            
#         if char == '\\' and in_string:
#             current_object += char
#             escape_next = True
#             i += 1
#             continue
            
#         if char == '"':
#             in_string = not in_string
#             current_object += char
#         elif not in_string:
#             if char == '{':
#                 if brace_count == 0:
#                     current_object = "{"
#                 else:
#                     current_object += char
#                 brace_count += 1
#             elif char == '}':
#                 current_object += char
#                 brace_count -= 1
#                 if brace_count == 0 and current_object.strip():
#                     json_objects.append(current_object.strip())
#                     current_object = ""
#             elif brace_count > 0:
#                 current_object += char
#         else:
#             current_object += char
            
#         i += 1
    
#     return json_objects

# def fix_json_issues(json_str):
#     """
#     Try to fix common JSON formatting issues.
    
#     Args:
#         json_str (str): Potentially malformed JSON string
        
#     Returns:
#         str: Fixed JSON string
#     """
#     # Remove any trailing commas before closing braces
#     json_str = re.sub(r',\s*}', '}', json_str)
#     json_str = re.sub(r',\s*]', ']', json_str)
    
#     return json_str

# def parse_jsonl_file_alternative(filename):
#     """
#     Alternative parsing method that reads line by line and accumulates.
    
#     Args:
#         filename (str): Path to the JSONL file
        
#     Returns:
#         list: List of dictionaries containing problem data
#     """
#     problems = []
    
#     try:
#         with open(filename, 'r', encoding='utf-8') as file:
#             lines = file.readlines()
        
#         current_json = ""
#         brace_count = 0
#         in_triple_quotes = False
        
#         for line_num, line in enumerate(lines, 1):
#             original_line = line
#             line = line.rstrip('\n\r')
            
#             # Check for triple quotes
#             if '"""' in line:
#                 triple_quote_count = line.count('"""')
#                 if triple_quote_count % 2 == 1:  # Odd number means toggle state
#                     in_triple_quotes = not in_triple_quotes
            
#             # If we're in triple quotes, just accumulate the line
#             if in_triple_quotes:
#                 current_json += line + "\\n"  # Add escaped newline
#                 continue
            
#             # Count braces when not in triple quotes
#             in_string = False
#             escape_next = False
            
#             for char in line:
#                 if escape_next:
#                     escape_next = False
#                     continue
#                 if char == '\\':
#                     escape_next = True
#                     continue
#                 if char == '"' and not escape_next:
#                     in_string = not in_string
#                     continue
#                 if not in_string:
#                     if char == '{':
#                         brace_count += 1
#                     elif char == '}':
#                         brace_count -= 1
            
#             current_json += line
            
#             # If braces are balanced and we have content, try to parse
#             if brace_count == 0 and current_json.strip():
#                 try:
#                     # Fix triple quotes in the accumulated JSON
#                     fixed_json = fix_triple_quotes(current_json)
#                     problem_data = json.loads(fixed_json)
#                     problems.append(problem_data)
#                     print(f"Problem {len(problems)} parsed successfully (lines up to {line_num})")
#                     print_problem_info(problem_data)
#                     current_json = ""
#                 except json.JSONDecodeError as e:
#                     print(f"Error parsing JSON ending at line {line_num}: {e}")
#                     print(f"JSON content: {current_json[:200]}...")
#                     current_json = ""
#                 except Exception as e:
#                     print(f"Unexpected error at line {line_num}: {e}")
#                     current_json = ""
                    
#     except FileNotFoundError:
#         print(f"File '{filename}' not found.")
#         return []
#     except Exception as e:
#         print(f"Error reading file: {e}")
#         return []
    
#     return problems

# def print_problem_info(problem_data):
#     """Helper function to print problem information"""
#     if 'problem' in problem_data:
#         print(f"  Question: {problem_data['problem'][:100]}...")
#     elif 'Problem' in problem_data:
#         print(f"  Question: {problem_data['Problem'][:100]}...")
    
#     if 'answer' in problem_data:
#         print(f"  Answer: {problem_data['answer']}")
#     elif 'Answer' in problem_data:
#         print(f"  Answer: {problem_data['Answer'][:50]}...")
    
#     print("-" * 50)

# def extract_field(problems, field_name):
#     """
#     Extract a specific field from all problems.
    
#     Args:
#         problems (list): List of problem dictionaries
#         field_name (str): Name of the field to extract
        
#     Returns:
#         list: List of values for the specified field
#     """
#     values = []
#     for problem in problems:
#         if field_name in problem:
#             values.append(problem[field_name])
#         # Handle case variations (Problem vs problem, etc.)
#         elif field_name.lower() in [k.lower() for k in problem.keys()]:
#             for k, v in problem.items():
#                 if k.lower() == field_name.lower():
#                     values.append(v)
#                     break
#     return values

# def filter_problems_by_criteria(problems, criteria_func):
#     """
#     Filter problems based on a custom criteria function.
    
#     Args:
#         problems (list): List of problem dictionaries
#         criteria_func (function): Function that takes a problem dict and returns bool
        
#     Returns:
#         list: Filtered list of problems
#     """
#     return [problem for problem in problems if criteria_func(problem)]

# def main():
#     # Parse the JSONL file using the main method
#     filename = "problem.jsonl"
#     print("Parsing with main method...")
#     problems = parse_jsonl_file(filename)
    
#     # If main method doesn't work well, try alternative
#     if len(problems) < 5:  # Assuming you should have more than 5 problems
#         print("\nTrying alternative parsing method...")
#         problems = parse_jsonl_file_alternative(filename)
    
#     print(f"\nTotal problems loaded: {len(problems)}")
    
#     if problems:
#         # Example usage: Extract all answers
#         answers = extract_field(problems, "answer")
#         print(f"\nExtracted {len(answers)} answers:")
#         for i, answer in enumerate(answers[:5], 1):  # Show first 5
#             print(f"  {i}. {answer}")
        
#         # Example usage: Extract all CoT (Chain of Thought)
#         cot_responses = extract_field(problems, "CoT")
#         print(f"\nExtracted {len(cot_responses)} CoT responses")
        
#         # Example usage: Filter problems by length of question
#         short_problems = filter_problems_by_criteria(
#             problems, 
#             lambda p: len(p.get('problem', p.get('Problem', ''))) < 100
#         )
#         print(f"\nFound {len(short_problems)} short problems (< 100 chars)")
        
#         # Example usage: Access individual problem data
#         if problems:
#             print(f"\nFirst problem details:")
#             first_problem = problems[0]
#             for key, value in first_problem.items():
#                 if isinstance(value, str) and len(value) > 100:
#                     print(f"  {key}: {value[:97]}...")
#                 else:
#                     print(f"  {key}: {value}")

# if __name__ == "__main__":
#     main()
import json
import re

def extract_json_objects_simple(filename):
    """
    Extract JSON objects from a JSONL file by finding complete bracket pairs.
    Does not attempt to parse or validate the JSON content.
    
    Args:
        filename (str): Path to the JSONL file
        
    Returns:
        list: List of raw JSON strings (one per complete bracket pair)
    """
    json_objects = []
    
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
        
        brace_count = 0
        current_object = ""
        in_string = False
        escape_next = False
        i = 0
        
        while i < len(content):
            char = content[i]
            
            # Handle escape sequences
            if escape_next:
                current_object += char
                escape_next = False
                i += 1
                continue
                
            if char == '\\' and in_string:
                current_object += char
                escape_next = True
                i += 1
                continue
                
            # Track if we're inside a string
            if char == '"':
                in_string = not in_string
                current_object += char
            elif not in_string:
                if char == '{':
                    if brace_count == 0:
                        current_object = "{"
                    else:
                        current_object += char
                    brace_count += 1
                elif char == '}':
                    current_object += char
                    brace_count -= 1
                    if brace_count == 0 and current_object.strip():
                        json_objects.append(current_object.strip())
                        current_object = ""
                elif brace_count > 0:
                    current_object += char
            else:
                # Inside a string - escape control characters
                if char == '\n':
                    current_object += '\\n'
                elif char == '\r':
                    current_object += '\\r'
                elif char == '\t':
                    current_object += '\\t'
                else:
                    current_object += char
            
            i += 1
            
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []
    
    return json_objects

def fix_json_control_chars(json_str):
    """
    Fix unescaped control characters in JSON strings.
    
    Args:
        json_str (str): JSON string with potential control character issues
        
    Returns:
        str: JSON string with properly escaped control characters
    """
    # This is a more targeted approach - only fix obvious control chars in string values
    def escape_string_content(match):
        content = match.group(1)
        # Escape control characters
        content = content.replace('\n', '\\n')
        content = content.replace('\r', '\\r') 
        content = content.replace('\t', '\\t')
        content = content.replace('\b', '\\b')
        content = content.replace('\f', '\\f')
        return f'"{content}"'
    
    # Find string values and escape control characters in them
    # This regex matches quoted strings, being careful about escaped quotes
    pattern = r'"((?:[^"\\]|\\.)*)(?<!\\)"'
    fixed_json = re.sub(pattern, escape_string_content, json_str)
    
    return fixed_json

def jsonLTOdict(filename="/home/admin/langzhang.zzy/inclusionAI/AReaL/out_open_math_reasoning_small.jsonl", length_limit = 10):
    #filename = "problem.jsonl"
    
    # Extract all JSON objects
    json_objects = extract_json_objects_simple(filename)
    
    # Convert JSON strings to Python dictionaries
    parsed_objects = []
    for i, json_str in enumerate(json_objects, 1):
        try:
            parsed_dict = json.loads(json_str)
            parsed_objects.append(parsed_dict)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON object {i}: {e}")
            # Try to fix control character issues
            try:
                fixed_json = fix_json_control_chars(json_str)
                parsed_dict = json.loads(fixed_json)
                parsed_objects.append(parsed_dict)
                print(f"  -> Fixed control characters and parsed successfully!")
            except json.JSONDecodeError as e2:
                print(f"  -> Still failed after fixing: {e2}")
                # Save the problematic JSON for debugging
                with open(f"debug_object_{i}.json", 'w', encoding='utf-8') as f:
                    f.write(json_str)
                print(f"  -> Saved problematic JSON to debug_object_{i}.json")
                continue
        if i == length_limit:
            print(f"Reached length limit of {length_limit} objects, stopping early.")
            break
    
    print(f"\nTotal objects extracted: {len(json_objects)}")
    print(f"Total objects successfully parsed: {len(parsed_objects)}")
    
    # Print the last parsed dictionary
    if parsed_objects:
        print(f"\n--- Last Object (Dictionary) ---")
        last_dict = parsed_objects[-1]
        for key, value in last_dict.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"{key}: {value[:97]}...")
            else:
                print(f"{key}: {value}")
    else:
        print("No objects were successfully parsed.")
    return parsed_objects
if __name__ == "__main__":
    jsonLTOdict()