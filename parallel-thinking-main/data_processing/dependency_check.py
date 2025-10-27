import xml.etree.ElementTree as ET
import re
import json
from openai import OpenAI
from dotenv import load_dotenv
import os
import itertools

# Load environment variables
load_dotenv()
openai_api_key = os.environ.get("INFINI_API_KEY")
openai_base_url = os.environ.get("INFINI_BASE_URL")

client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)
model = "deepseek-v3-0324"

class ThreadInfo:
    def __init__(self, thread_id, task, objective, content, result):
        self.thread_id = thread_id
        self.task = task
        self.objective = objective
        self.content = content
        self.result = result
    
    def __str__(self):
        return f"Thread {self.thread_id}: {self.task} - {self.objective}"

def load_xml_from_jsonl(file_path, target_line_num):
    """
    Load XML content from a JSONL file using the "thread_1" key.
    
    Args:
        file_path (str): Path to the JSONL file
        target_line_num (int): Line number to read (1-indexed)
        
    Returns:
        str: XML content from the "thread_1" key, or None if not found
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
                            
                            # Return thread_1 value if it exists
                            if 'thread_1' in data:
                                print(f"Successfully loaded XML from entry {line_num}")
                                return data['thread_1']
                            else:
                                print(f"Entry {line_num}: No 'thread_1' key found")
                                return None
                                
                        except json.JSONDecodeError as e:
                            print(f"Error parsing line {line_num}: {e}")
                            return None
                    else:
                        print(f"Line {line_num} is empty")
                        return None
                        
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    print(f"Line {target_line_num} not found in file")
    return None

def parse_xml_content(xml_content):
    """
    Parse XML content and extract thread information.
    
    Args:
        xml_content (str): XML content as string
        
    Returns:
        list: List of ThreadInfo objects
    """
    threads = []
    
    try:
        # Extract thread information using regex since the XML structure is not standard
        thread_pattern = r'<thread id=\'(\d+)\'>(.*?)</thread>'
        processing_pattern = r'<thread_processing id = \'(\d+)\'>(.*?)</thread_processing>'
        result_pattern = r'<thread_result id=\'(\d+)\'>(.*?)</thread_result>'
        
        # Find all threads
        thread_matches = re.findall(thread_pattern, xml_content, re.DOTALL)
        processing_matches = re.findall(processing_pattern, xml_content, re.DOTALL)
        result_matches = re.findall(result_pattern, xml_content, re.DOTALL)
        
        # Create dictionaries for easy lookup
        processing_dict = {int(match[0]): match[1].strip() for match in processing_matches}
        result_dict = {int(match[0]): match[1].strip() for match in result_matches}
        
        for thread_id, thread_content in thread_matches:
            thread_id = int(thread_id)
            
            # Extract task and objective from thread content
            task_match = re.search(r'<task>(.*?)</task>', thread_content, re.DOTALL)
            objective_match = re.search(r'<objective>(.*?)</objective>', thread_content, re.DOTALL)
            
            task = task_match.group(1).strip() if task_match else ""
            objective = objective_match.group(1).strip() if objective_match else ""
            
            # Get processing content and result
            processing_content = processing_dict.get(thread_id, "")
            result_content = result_dict.get(thread_id, "")
            
            thread_info = ThreadInfo(thread_id, task, objective, processing_content, result_content)
            threads.append(thread_info)
        
        # Sort threads by ID
        threads.sort(key=lambda x: x.thread_id)
        
    except Exception as e:
        print(f"Error parsing XML content: {e}")
    
    return threads

def parse_xml_file(file_path):
    """
    Parse the XML file and extract thread information.
    (Kept for backward compatibility)
    
    Args:
        file_path (str): Path to the XML file
        
    Returns:
        list: List of ThreadInfo objects
    """
    threads = []
    
    try:
        # Read the file content
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        return parse_xml_content(content)
        
    except Exception as e:
        print(f"Error parsing XML file: {e}")
    
    return threads

def analyze_all_dependencies_together(threads):
    """
    Analyze dependencies between all threads in a single LLM call.
    
    Args:
        threads (list): List of ThreadInfo objects
        
    Returns:
        list: List of dependency tuples (dependent_thread_id, dependency_thread_id)
    """
    if len(threads) <= 1:
        return []
    
    # Build the prompt with all threads
    threads_info = ""
    for thread in threads:
        threads_info += f"""
Thread {thread.thread_id}:
- Task: {thread.task}
- Objective: {thread.objective}
- Content: {thread.content}
- Result: {thread.result}
"""
    
    prompt = f"""
You are analyzing dependencies between mathematical reasoning steps that will be executed in parallel threads.
Determine which threads depend on other threads, meaning a thread needs another thread to be completed BEFORE it can start its reasoning process. 

IMPORTANT RULES:
1. All thread results are visible to all other threads, so dependencies based ONLY on using another thread's task, objective, result do NOT count as dependencies.
2. Only consider TRUE REASONING DEPENDENCIES where one thread needs another thread's reasoning process or intermediate steps to begin its own reasoning. We can assume all the task, objective, and result information is available globally.
3. Do NOT consider result usage as dependency since all tasks, objectives, and results are available globally.

Here are all the threads:
{threads_info}

For each thread, determine if it depends on any other threads. Consider ONLY these factors:
1. Does the thread need the REASONING PROCESS or INTERMEDIATE STEPS from another thread (not just the final result)?
2. Does the thread build upon CONCEPTS or METHODS established during another thread's **reasoning** process? Do NOT consider result usage.
3. Would the thread be unable to START its reasoning without another thread's **reasoning** process being completed first? We can assume all results are available globally.

For example, if a step calculates the maximum of x + y when x^2 + y^2 = 1, by assuming x = cos(theta) and y = sin(theta), it depends on the reasoning process of the step if the step is using theta, but NOT if it is using x + y, x, y, or the result of the step directly

Output format: List all dependencies as pairs in the format "(dependent_thread_id, source_thread_id)" where the first thread depends on the second thread.
If there are no dependencies, output "NO_DEPENDENCIES".

Examples:
- If Thread 2 depends on Thread 0, output: (2, 0)
- If Thread 3 depends on Thread 1 and Thread 2 depends on Thread 0, output: (3, 1), (2, 0)
- If no dependencies exist, output: NO_DEPENDENCIES

Only output the dependency pairs or "NO_DEPENDENCIES", nothing else.
"""

    try:
        print(f"Analyzing dependencies for {len(threads)} threads in a single call...")
        
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": "You are a mathematical reasoning dependency analyzer. Focus on reasoning process dependencies, not result usage. Output only dependency pairs or NO_DEPENDENCIES."},
                {"role": "user", "content": prompt}
            ]
        )
        
        result = response.choices[0].message.content.strip()
        print(f"LLM Response: {result}")
        
        # Parse the result
        dependencies = []
        if result.upper() == "NO_DEPENDENCIES":
            print("No dependencies found by LLM")
            return dependencies
        
        # Extract dependency pairs using regex
        pattern = r'\((\d+),\s*(\d+)\)'
        matches = re.findall(pattern, result)
        
        for match in matches:
            dependent_id = int(match[0])
            source_id = int(match[1])
            dependencies.append((dependent_id, source_id))
            print(f"Found dependency: Thread {dependent_id} depends on Thread {source_id}")
        
        return dependencies
        
    except Exception as e:
        print(f"Error analyzing dependencies: {e}")
        return []

def print_dependency_summary(threads, dependencies):
    """
    Print a summary of threads and their dependencies.
    
    Args:
        threads (list): List of ThreadInfo objects
        dependencies (list): List of dependency tuples
    """
    print("\n" + "="*60)
    print("THREAD SUMMARY")
    print("="*60)
    
    for thread in threads:
        print(f"Thread {thread.thread_id}: {thread.task}")
        print(f"  Objective: {thread.objective}")
        print(f"  Result: {thread.result}")
        print()
    
    print("="*60)
    print("DEPENDENCY ANALYSIS")
    print("="*60)
    
    if dependencies:
        print("Found dependencies:")
        for dep_thread, source_thread in dependencies:
            print(f"  Thread {dep_thread} depends on Thread {source_thread}")
    else:
        print("No dependencies found - all threads can run in parallel!")
    
    print(f"\nTotal threads: {len(threads)}")
    print(f"Total dependencies: {len(dependencies)}")

def main():
    # Configuration - choose your input method
    use_jsonl = True  # Set to True to use JSONL, False to use XML file
    
    if use_jsonl:
        # JSONL configuration
        jsonl_file_path = "output_submit_all_1000_new.jsonl"  # Change to your JSONL file
        target_line = 3  # Change to the line number you want (1-indexed)
        
        print(f"Loading XML from JSONL file: {jsonl_file_path}, line {target_line}")
        xml_content = load_xml_from_jsonl(jsonl_file_path, target_line)
        
        if xml_content is None:
            print("Failed to load XML content from JSONL file.")
            return
        
        # Parse XML content
        threads = parse_xml_content(xml_content)
    else:
        # XML file configuration (backward compatibility)
        xml_file_path = "temp_thread1_3.xml"
        print(f"Loading XML from file: {xml_file_path}")
        threads = parse_xml_file(xml_file_path)
    
    if not threads:
        print("No threads found in the XML content.")
        return
    
    print(f"Found {len(threads)} threads")
    
    # Analyze dependencies using single LLM call
    dependencies = analyze_all_dependencies_together(threads)
    
    # Print summary
    print_dependency_summary(threads, dependencies)
    
    # Optional: Create adjacency matrix
    if threads:
        n = len(threads)
        adj_matrix = [[0 for _ in range(n)] for _ in range(n)]
        
        # Map thread IDs to indices
        id_to_index = {thread.thread_id: i for i, thread in enumerate(threads)}
        
        for dep_thread, source_thread in dependencies:
            if dep_thread in id_to_index and source_thread in id_to_index:
                dep_idx = id_to_index[dep_thread]
                source_idx = id_to_index[source_thread]
                adj_matrix[dep_idx][source_idx] = 1
        
        print("\nAdjacency Matrix (rows depend on columns):")
        print("   ", end="")
        for thread in threads:
            print(f"{thread.thread_id:2}", end=" ")
        print()
        
        for i, thread in enumerate(threads):
            print(f"{thread.thread_id:2}:", end="")
            for j in range(n):
                print(f"{adj_matrix[i][j]:2}", end=" ")
            print()

if __name__ == "__main__":
    main()