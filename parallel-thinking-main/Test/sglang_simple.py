import sglang as sgl
from sglang import function, gen, set_default_backend, RuntimeEndpoint
import requests
import time
import subprocess
import re
import asyncio
from typing import List, Dict, Tuple
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import signal
import os
import json
from datetime import datetime

# Define system prompts (same as original)
MAIN_SYSTEM_PROMPT = """You are a helpful assistant that solve math problems. 
The input is a math problem and you need to solve it step by step. If you think you need to launch a thread, you can do so by using the '<launch_threads>' tag. Each thread will have its own task and objective, put the whole thread_launching process in the following format:

Write a code that use sglang server to generate the answer. This neeed to have the same propmt and structure (delaing with parallel processing) as the code in sglang_inference.py, but you need to stop the server after each generation, also process the thread one by one so as to avoid deadlock
```
<launch_threads>
<thread id='0'>
<task>
[Task Name]
</task>
<objective>
[Objective of the thread]
</objective>
</thread>
<thread id='1'>
<task>
[Task Name]
</task>
<objective>
[Objective of the thread]
</objective>
</thread>
</launch_threads>
```
You should complete the whole reasoning process of the original problem, rather than just a partial step in main mode. If you are in the main mode, start the reasoning process with the special tag '<think>'
"""
THREAD_SYSTEM_PROMPT = """You are a helpful assistant that solve math problems.  

The input is a math problem and reasoning process before this thread is launched. Also the task and objective of this thread will be provided in the end of the input. You should complete the task and objective of this thread, start your processing with <thread_processing id = 'i'> where i is the index of the thread. End your processing with '</thread_processing>'. After this, put the result of this step between '<thread_result id='i'>' and '</thread_result>'. DO NOT output the special tag '<think>'  DO NOT output the special tag '<think>', what you need to do is to finish the reasoning of <thread_processing id='i'> and output its result, you only need to solve this partial step not the full problem
 Stop reasoning when you reach the end of the thread processing and then output the result in the format of '<thread_result id='i'>result</thread_result>'.
 NEVER solve the whole problem, you MUST STOP after the objective of this step is reached. Also, think for a while before you output the result, put the reasoning process in <thread_processing id='i'> ... </thread_processing> tag, where 'i' is the id of this thread. Put the result of **THIS STEP** (not the whole problem) in the <thread_result id='i'> ... </thread_result> tag"""

# Format function (same)
def format_chat_template(system_prompt: str, user_content: str, assistant_start: str = "") -> str:
    formatted = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
{assistant_start}"""
    return formatted

# SGL functions (same)
@sgl.function
def main_generation_until_threads(s, problem: str, first_token: str = "<think>", max_tokens: int = 16384):
    user_content = problem
    formatted_prompt = format_chat_template(MAIN_SYSTEM_PROMPT, user_content, first_token)
    
    s += formatted_prompt
    s += gen(
        "main_reasoning",
        max_tokens=max_tokens,
        temperature=0.6,
        top_p=0.9,
        stop=["</launch_threads>", "<|im_end|>", "<|endoftext|>", "</s>"]
    )

@sgl.function
def continue_main_generation(s, problem: str, accumulated_context: str, max_tokens: int = 16384):
    user_content = problem
    assistant_marker = "<|im_start|>assistant\n"
    if assistant_marker in accumulated_context:
        assistant_response = accumulated_context.split(assistant_marker, 1)[1]
    else:
        assistant_response = accumulated_context
    
    formatted_prompt = format_chat_template(MAIN_SYSTEM_PROMPT, user_content, assistant_response)
    
    s += formatted_prompt
    s += gen(
        "final_reasoning",
        max_tokens=max_tokens,
        temperature=0.6,
        top_p=0.9,
        stop=["</launch_threads>", "<|im_end|>", "<|endoftext|>", "</s>"]
    )

@sgl.function
def thread_processing(s, problem: str, context: str, thread_id: str, task: str, objective: str, max_tokens: int = 16384):
    user_content = f"""Problem: {problem}
{context}
<thread id='{thread_id}'>
<task>{task}</task><objective>{objective}</objective></thread>"""
    
    assistant_start = f"<thread_processing id='{thread_id}'>"
    formatted_prompt = format_chat_template(THREAD_SYSTEM_PROMPT, user_content, assistant_start)
    
    s += formatted_prompt
    s += gen(
        "thread_response",
        max_tokens=max_tokens,
        temperature=0.6,
        top_p=0.9,
        stop=["</thread_result>", "<|im_end|>", "<|endoftext|>", "</s>"]
    )

# ActivityMonitor class (simplified, kept for timeout detection)
class ActivityMonitor:
    def __init__(self, timeout_seconds=600):
        self.timeout_seconds = timeout_seconds
        self.last_activity = time.time()
        self.lock = threading.Lock()
        self.active = True
        
    def update_activity(self):
        with self.lock:
            self.last_activity = time.time()
    
    def check_timeout(self):
        with self.lock:
            return time.time() - self.last_activity > self.timeout_seconds
    
    def deactivate(self):
        with self.lock:
            self.active = False
    
    def is_active(self):
        with self.lock:
            return self.active

# Modified processor class - process threads sequentially, remove parallel and restart inside
class SequentialThreadProcessor:
    def __init__(self, host="127.0.0.1", port=30002):
        self.host = host
        self.port = port
        self.monitor = ActivityMonitor()

    def extract_thread_launch_info(self, text: str) -> list:
        # Same as original
        self.monitor.update_activity()
        launch_pattern = r'<launch_threads>(.*?)</launch_threads>'
        launch_matches = re.findall(launch_pattern, text, re.DOTALL)
        
        if not launch_matches:
            incomplete_pattern = r'<launch_threads>(.*?)$'
            incomplete_matches = re.findall(incomplete_pattern, text, re.DOTALL)
            if incomplete_matches:
                launch_matches = incomplete_matches
            else:
                return []
        
        result_pattern = r'<thread_result id=\'(\d+)\'>'
        existing_results = re.findall(result_pattern, text)
        existing_thread_ids = set(existing_results)
        
        thread_info = []
        for launch_content in launch_matches:
            thread_pattern = r"<thread id='(\d+)'>\s*<task>\s*(.*?)\s*</task>\s*<objective>\s*(.*?)\s*</objective>\s*</thread>"
            threads = re.findall(thread_pattern, launch_content, re.DOTALL)
            for thread_id, task, objective in threads:
                thread_id = thread_id.strip()
                task = task.strip()
                objective = objective.strip()
                if thread_id not in existing_thread_ids:
                    thread_info.append({
                        'id': thread_id,
                        'task': task,
                        'objective': objective
                    })
        return thread_info
    
    def extract_thread_result(self, thread_response: str, thread_id: str) -> str:
        # Same as original
        self.monitor.update_activity()
        result_start_pattern = rf'<thread_result id=\'{thread_id}\'>'
        result_start_match = re.search(result_start_pattern, thread_response)
        if result_start_match:
            start_pos = result_start_match.start()
            result_content = thread_response[start_pos:].strip()
            result_content = result_content + "\n</thread_result>"
            return result_content
        else:
            return f"<thread_result id='{thread_id}'>\n{thread_response.strip()}\n</thread_result>"
        
    def process_single_thread(self, problem: str, context: str, thread_info: dict) -> tuple:
        # Same as original
        try:
            self.monitor.update_activity()
            state = thread_processing.run(
                problem=problem,
                context=context,
                thread_id=thread_info['id'],
                task=thread_info['task'],
                objective=thread_info['objective'],
                max_tokens=16384
            )
            self.monitor.update_activity()
            thread_response = state['thread_response']
            thread_result = self.extract_thread_result(thread_response, thread_info['id'])
            self.monitor.update_activity()
            return thread_info['id'], thread_result
        except Exception as e:
            self.monitor.update_activity()
            error_result = f"<thread_result id='{thread_info['id']}'>Error: {str(e)}</thread_result>"
            return thread_info['id'], error_result
    
    def process_threads_sequential(self, problem: str, context: str, thread_infos: list) -> dict:
        results = {}
        self.monitor.update_activity()
        for thread_info in thread_infos:
            thread_id, result_content = self.process_single_thread(problem, context, thread_info)
            results[thread_id] = result_content
            self.monitor.update_activity()
        return results
    
    def generate_with_sequential_threads(self, problem: str, first_token: str = "<think>") -> str:
        self.monitor.update_activity()
        main_state = main_generation_until_threads.run(
            problem=problem,
            first_token=first_token,
            max_tokens=16384
        )
        self.monitor.update_activity()
        main_reasoning = main_state['main_reasoning']
        if not main_reasoning.endswith("</launch_threads>"):
            main_reasoning += "</launch_threads>\n"
        
        current_context = format_chat_template(MAIN_SYSTEM_PROMPT, problem, first_token + main_reasoning)
        assistant_marker = "<|im_start|>assistant\n"
        if assistant_marker in current_context:
            reasoning_context = current_context.split(assistant_marker, 1)[1]
        else:
            reasoning_context = first_token + main_reasoning
        
        main_turn_count = 1
        while True:
            self.monitor.update_activity()
            thread_turn_count = 1
            while True:
                self.monitor.update_activity()
                thread_infos = self.extract_thread_launch_info(reasoning_context)
                if not thread_infos:
                    break
                
                thread_results = self.process_threads_sequential(problem, reasoning_context, thread_infos)
                combined_results = ""
                for thread_id in sorted(thread_results.keys()):
                    combined_results += thread_results[thread_id] + "\n"
                
                reasoning_context = reasoning_context + "<step_resolution>\n" + combined_results + "</step_resolution>\n"
                thread_turn_count += 1
                if thread_turn_count > 5:
                    break
            
            self.monitor.update_activity()
            final_state = continue_main_generation.run(
                problem=problem,
                accumulated_context=reasoning_context,
                max_tokens=16384
            )
            self.monitor.update_activity()
            final_reasoning = final_state['final_reasoning']
            if not final_reasoning.endswith("</launch_threads>") and "<launch_threads>" in final_reasoning:
                final_reasoning += "</launch_threads>\n"
            
            reasoning_context = reasoning_context + final_reasoning
            
            if "<launch_threads>" not in final_reasoning:
                break
            
            main_turn_count += 1
            if main_turn_count > 10:
                break
                
        self.monitor.update_activity()
        return reasoning_context

# Helper functions (same)
def wait_for_server(host="127.0.0.1", port=30002, timeout=300):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http:#{host}:{port}/health", timeout=5)
            if response.status_code == 200:
                return True
        except:
            time.sleep(10)
    return False

def terminate_server_process(server_process):
    if server_process is None:
        return
    try:
        server_process.terminate()
        server_process.wait(timeout=10)
    except:
        server_process.kill()
        server_process.wait()

def save_progress(completed_problems, progress_file="progress_simple.json"):
    with open(progress_file, 'w') as f:
        json.dump({
            'completed_problems': completed_problems,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

def load_progress(progress_file="progress_simple.json"):
    try:
        with open(progress_file, 'r') as f:
            data = json.load(f)
            return set(data.get('completed_problems', []))
    except:
        return set()

# Modified process function - start server once per problem, terminate after
def process_problem_with_timeout(problem_text, problem_index, max_retries=3):
    for attempt in range(max_retries):
        model_path = "/nvme1n1/zzy/parallel_7230838/deepseek-parallel-thinking"
        host = "127.0.0.1"
        port = 30002
        
        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model-path", model_path,
            "--host", host,
            "--port", str(port),
            "--tp-size", "1",
            "--max-total-tokens", "16384",
            "--mem-fraction-static", "0.85",
            "--disable-cuda-graph",
            "--chunked-prefill-size", "1024",
        ]
        
        server_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        try:
            if not wait_for_server(host, port, timeout=300):
                terminate_server_process(server_process)
                continue
                
            backend = RuntimeEndpoint(f"http:#{host}:{port}")
            sgl.set_default_backend(backend)
            
            processor = SequentialThreadProcessor(host, port)
            
            def timeout_monitor():
                while processor.monitor.is_active():
                    time.sleep(10)
                    if processor.monitor.check_timeout():
                        terminate_server_process(server_process)
                        break
            
            monitor_thread = threading.Thread(target=timeout_monitor, daemon=True)
            monitor_thread.start()
            
            full_response = processor.generate_with_sequential_threads(problem_text)
            
            processor.monitor.deactivate()
            
            base_directory = time.strftime("%Y%m%d_%H%M%S")
            if not os.path.exists(base_directory):
                os.makedirs(base_directory)
            output_file = os.path.join(base_directory, f"answer_simple_problem_{problem_index}.txt")
            with open(output_file, "w") as f:
                f.write(f"Problem {problem_index}:\n{problem_text}\n\nSolution:\n{full_response}\n\nCompleted at: {datetime.now().isoformat()}\n")
            
            terminate_server_process(server_process)
            return True
            
        except Exception as e:
            processor.monitor.deactivate()
            terminate_server_process(server_process)
            if attempt < max_retries - 1:
                time.sleep(30)
            else:
                return False
    
    return False

# Main function (same, with progress file changed)
def main():
    problems = [
        # (same list of problems as in the original code)
        r"""Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.""",
        # ... (all other problems here, omitted for brevity in this edit block)
    ]
    completed_problems = load_progress()
    
    successful_count = 0
    failed_count = 0
    
    for i, problem in enumerate(problems):
        if i in completed_problems:
            successful_count += 1
            continue
        
        success = process_problem_with_timeout(problem, i)
        
        if success:
            successful_count += 1
            completed_problems.add(i)
            save_progress(list(completed_problems))
        else:
            failed_count += 1
        
        if i < len(problems) - 1:
            time.sleep(10)

if __name__ == "__main__":
    main()

