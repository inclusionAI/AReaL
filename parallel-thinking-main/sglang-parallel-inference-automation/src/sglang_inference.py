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

@sgl.function
def main_generation_until_threads(s, prompt_with_first_token: str, max_tokens: int = 16384):
    s += prompt_with_first_token
    s += gen(
        "main_reasoning",
        max_tokens=max_tokens,
        temperature=0.2,
        top_p=0.9,
        stop=["</launch_threads>", "<|im_end|>", "<|endoftext|>", "</s>"]
    )

@sgl.function
def continue_main_generation(s, context: str, thread_results: str, max_tokens: int = 16384):
    s += context + thread_results
    s += gen(
        "final_reasoning",
        max_tokens=max_tokens,
        temperature=0.2,
        top_p=0.9,
        stop=["</launch_threads>", "<|im_end|>", "<|endoftext|>", "</s>"]
    )

@sgl.function
def thread_processing(s, problem: str, context: str, thread_id: str, task: str, objective: str, max_tokens: int = 16384):
    thread_prompt = f"""Problem: {problem}
{context}
<thread id='{thread_id}'>
<task>{task}</task><objective>{objective}</objective></thread>\n
<thread_processing id = '{thread_id}'>
"""
    
    s += thread_prompt
    s += gen(
        "thread_response",
        max_tokens=max_tokens,
        temperature=0.2,
        top_p=0.9,
        stop=["</thread_result>", "<|im_end|>", "<|endoftext|>", "</s>"]
    )

class ActivityMonitor:
    def __init__(self, timeout_seconds=300):
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

class ParallelThreadProcessor:
    def __init__(self, host="127.0.0.1", port=30001):
        self.host = host
        self.port = port
        self.backend = None
        self.monitor = ActivityMonitor()
        
    def extract_thread_launch_info(self, text: str) -> List[Dict[str, str]]:
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
        
        for i, launch_content in enumerate(launch_matches):
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
        
    def process_single_thread(self, problem: str, context: str, thread_info: Dict[str, str]) -> Tuple[str, str]:
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
            return thread_info['id'], thread_result
            
        except Exception as e:
            error_result = f"<thread_result id='{thread_info['id']}'>Error: {str(e)}</thread_result>"
            return thread_info['id'], error_result
    
    def process_threads_parallel(self, problem: str, context: str, thread_infos: List[Dict[str, str]]) -> Dict[str, str]:
        results = {}
        
        self.monitor.update_activity()
        
        with ThreadPoolExecutor(max_workers=min(len(thread_infos), 4)) as executor:
            future_to_thread = {
                executor.submit(self.process_single_thread, problem, context, thread_info): thread_info['id']
                for thread_info in thread_infos
            }
            
            for future in as_completed(future_to_thread):
                thread_id = future_to_thread[future]
                try:
                    result_thread_id, result_content = future.result()
                    results[result_thread_id] = result_content
                    self.monitor.update_activity()
                except Exception as e:
                    results[thread_id] = f"<thread_result id='{thread_id}'>Error: {str(e)}</thread_result>"
                    self.monitor.update_activity()
        
        return results
    
    def generate_with_parallel_threads(self, problem: str, first_token: str = "<reasoning_process>") -> str:
        self.monitor.update_activity()
        
        prompt_with_first_token = problem + first_token
        
        main_state = main_generation_until_threads.run(
            prompt_with_first_token=prompt_with_first_token,
            max_tokens=16384
        )
        
        self.monitor.update_activity()
        main_reasoning = main_state['main_reasoning']
        
        if not main_reasoning.endswith("</launch_threads>"):
            main_reasoning += "</launch_threads>\n"
        
        current_context = prompt_with_first_token + main_reasoning
        
        main_turn_count = 1
        while True:
            self.monitor.update_activity()
            
            thread_turn_count = 1
            while True:
                self.monitor.update_activity()
                
                thread_infos = self.extract_thread_launch_info(current_context)
                
                if not thread_infos:
                    break
                
                thread_results = self.process_threads_parallel(problem, current_context, thread_infos)
                
                combined_results = ""
                for thread_id in sorted(thread_results.keys()):
                    combined_results += thread_results[thread_id] + "\n"
                
                current_context =  current_context + "<step_resolution>\n" + combined_results + "</step_resolution>\n" + "</parallel_processing>\n"
                
                thread_turn_count += 1
                if thread_turn_count > 5:
                    break
            
            self.monitor.update_activity()
            final_state = continue_main_generation.run(
                context=current_context,
                thread_results="",
                max_tokens=16384
            )
            
            self.monitor.update_activity()
            final_reasoning = final_state['final_reasoning']
            if not final_reasoning.endswith("</launch_threads>") and "<launch_threads>" in final_reasoning:
                final_reasoning += "</launch_threads>\n"
            
            current_context = current_context + final_reasoning
            
            if "<launch_threads>" not in final_reasoning:
                break
            
            main_turn_count += 1
            if main_turn_count > 10:
                break
        
        if "<answer>" not in current_context:
            current_context = current_context + "\n<answer>\n"
            answer_state = continue_main_generation.run(
                context=current_context,
                thread_results="",
                max_tokens=8192
            )
            self.monitor.update_activity()
            answer_reasoning = answer_state['final_reasoning']
            current_context = current_context + answer_reasoning
            
        self.monitor.update_activity()
        return current_context

def wait_for_server(host="127.0.0.1", port=30001, timeout=300):
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://{host}:{port}/health", timeout=5)
            if response.status_code == 200:
                return True
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(10)
    
    return False

def terminate_server_process(server_process):
    if server_process is None:
        return
    
    try:
        server_process.terminate()
        server_process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server_process.kill()
        server_process.wait()

def save_progress(completed_problems, progress_file="progress.json"):
    with open(progress_file, 'w') as f:
        json.dump({
            'completed_problems': completed_problems,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

def load_progress(progress_file="progress.json"):
    try:
        with open(progress_file, 'r') as f:
            data = json.load(f)
            return set(data.get('completed_problems', []))
    except FileNotFoundError:
        return set()
    except Exception as e:
        return set()

def process_problem_with_timeout(problem_text, problem_index, max_retries=3):
    for attempt in range(max_retries):
        model_path = "/nvme0n1/zzy_model/fresh_mixed_761514/deepseek-parallel-thinking"
        host = "127.0.0.1"
        port = 30001
        
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
                
            backend = RuntimeEndpoint(f"http://{host}:{port}")
            sgl.set_default_backend(backend)
            
            processor = ParallelThreadProcessor(host, port)
            
            first_token = "<reasoning_process>\n<parallel_processing>\n"
            
            full_response = processor.generate_with_parallel_threads(problem_text, first_token)
            
            terminate_server_process(server_process)
            return True
            
        except Exception as e:
            terminate_server_process(server_process)
            if attempt < max_retries - 1:
                time.sleep(30)
            else:
                return False
    
    return False

def main():
    problems = [
        # List of problems to process
    ]
    
    completed_problems = load_progress()
    
    total_problems = len(problems)
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
    
    print(f"Successful: {successful_count}, Failed: {failed_count}")

if __name__ == "__main__":
    main()