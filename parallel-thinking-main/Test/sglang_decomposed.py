import sglang as sgl
from sglang import function, gen, set_default_backend, RuntimeEndpoint
import subprocess
import time
import requests
import re
from typing import List, Dict
import os
import signal
SYSTEM_PROMPT_MAIN = """You are a helpful assistant that solve math problems. 
The input is a math problem and you need to solve it step by step. If you think you need to launch a thread, you can do so by using the '<launch_threads>' tag. Each thread will have its own task and objective, put the whole thread_launching process in the following format:
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
You should complete the whole reasoning process of the original problem, rather than just a partial step in main mode. If you are in the main mode, start the reasoning process with the special tag '<think>'"""
SYSTEM_PROMPT_THREAD = """You are a helpful assistant that solve math problems.  

The input is a math problem and reasoning process before this thread is launched. Also the task and objective of this thread will be provided in the end of the input. You should complete the task and objective of this thread, start your processing with <thread_processing id = 'i'> where i is the index of the thread. End your processing with '</thread_processing>'. After this, put the result of this step between '<thread_result id='i'>' and '</thread_result>'. DO NOT output the special tag '<think>'  DO NOT output the special tag '<think>', what you need to do is to finish the reasoning of <thread_processing id='i'> and output its result, you only need to solve this partial step not the full problem
 Stop reasoning when you reach the end of the thread processing and then output the result in the format of '<thread_result id='i'> ... </thread_result>'. NEVER solve the whole problem, you MUST STOP after the objective of this step is reached. Also, think for a while before you output the result, put the reasoning process in <thread_processing id='i'> ... </thread_processing> tag, where 'i' is the id of this thread. Put the result of **THIS STEP** (not the whole problem) in the <thread_result id='i'> ... </thread_result> tag"""
def wait_for_server(host="127.0.0.1", port=11451, timeout=300):
    """Wait for server to be ready"""
    print("⏳ Waiting for server to start...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://{host}:{port}/health", timeout=5)
            if response.status_code == 200:
                print("✅ Server is ready!")
                return True
        except (requests.ConnectionError, requests.Timeout):
            elapsed = int(time.time() - start_time)
            print(f"⏳ Server not ready yet, waiting... ({elapsed}s)")
            time.sleep(10)
    
    print(f"❌ Server failed to start within {timeout} seconds")
    return False

def format_chat_template(system_prompt: str, user_content: str, assistant_start: str = "") -> str:
    """Format content using ChatML template"""
    formatted = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
{assistant_start}"""
    return formatted

@sgl.function
def simple_generation(s, formatted_prompt: str, mode , max_tokens: int = 24000):
    """Simple generation function"""
    
    s += formatted_prompt
    if mode == "main":
        s += gen(
            "response",
            max_tokens=max_tokens,
            temperature=0.6,
            top_p=0.9,
            stop=["<|im_end|>", "<|endoftext|>", "</s>", "</launch_threads>"]
        )
    if mode == "thread":
        # For thread generation, we might want to allow more tokens
        s += gen(
            "response",
            max_tokens=max_tokens ,
            temperature=0.6,
            top_p=0.9,
            # add "</thread_result>
            stop=["<|im_end|>", "<|endoftext|>", "</s>", r"</thread_result>"]
        )
    # s += gen(
    #     "response",
    #     max_tokens=max_tokens,
    #     temperature=0.6,
    #     top_p=0.9,
    #     stop=["<|im_end|>", "<|endoftext|>", "</s>"]
    # )

# Global variable to track server process
global_server_process = None

def sglang_gen(sys_prompt, user_prompt, model_path, mode, host='localhost', port=11451):
    """
    Generate a response using SGLang server， return generated response.
    """
    global global_server_process
    
    try:
        # Start SGLang server if not already running
        server_process = None
        try:
            # Check if server is already running
            response = requests.get(f"http://{host}:{port}/health", timeout=2)
            if response.status_code != 200:
                raise requests.ConnectionError("Server not ready")
            print("✅ Server already running")
            
            # If server is running but we don't have the process, try to find it
            if global_server_process is None:
                try:
                    import psutil
                    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                        try:
                            if proc.info['cmdline'] and any('sglang.launch_server' in str(cmd) for cmd in proc.info['cmdline']):
                                global_server_process = proc
                                print(f"✅ Found existing server process (PID: {proc.pid})")
                                break
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue
                except ImportError:
                    print("⚠️ psutil not available, cannot track existing server process")
                    
        except (requests.ConnectionError, requests.Timeout):
            # Start server
            print(f"🚀 Starting SGLang server...")
            cmd = [
                "python", "-m", "sglang.launch_server",
                "--model-path", model_path,
                "--host", host,
                "--port", str(port),
                "--tp-size", "1",
                "--max-total-tokens", "8192",
                "--mem-fraction-static", "0.85",
                "--disable-cuda-graph",
                "--chunked-prefill-size", "1024",
            ]
            
            server_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            global_server_process = server_process  # Track the server process globally
            
            # Wait for server to be ready
            if not wait_for_server(host, port, timeout=300):
                if server_process:
                    server_process.terminate()
                    global_server_process = None
                raise RuntimeError("Failed to start SGLang server")
        
        # Connect to server
        backend = RuntimeEndpoint(f"http://{host}:{port}")
        sgl.set_default_backend(backend)
        print(f"✅ Connected to SGLang server at {host}:{port}")
        
        # Format prompt using ChatML template
        formatted_prompt = format_chat_template(sys_prompt, user_prompt)
        
        # Generate response
        state = simple_generation.run(
            formatted_prompt=formatted_prompt,
            mode = mode,
            max_tokens=24000
        )
        
        response = state['response']
        print (f"🔍 Generated response:\n{response}")  # Show first 500 chars for debugging
        print(f"✅ Generation completed")
        
        return response
        
    except Exception as e:
        print(f"❌ Error in sglang_gen: {e}")
        if server_process:
            server_process.terminate()
            global_server_process = None
        return None

def terminate_sglang_server():
    """Terminate the SGLang server process"""
    global global_server_process
    
    # First try to terminate using the stored process
    if global_server_process is not None:
        try:
            print("🛑 Terminating SGLang server...")
            # Try graceful termination first
            global_server_process.terminate()
            try:
                global_server_process.wait(timeout=10)
                print("✅ Server terminated gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if graceful termination fails
                print("⚠️ Graceful termination timeout, force killing...")
                global_server_process.kill()
                global_server_process.wait()
                print("✅ Server force killed")
        except Exception as e:
            print(f"❌ Error terminating server: {e}")
        finally:
            global_server_process = None
    else:
        # If no stored process, try to find and kill by port
        print("🔍 No stored server process, searching for server by port...")
        try:
            # Find processes using port 11451
            import psutil
            killed = False
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    # Look for sglang server processes
                    if proc.info['cmdline'] and any('sglang.launch_server' in str(cmd) for cmd in proc.info['cmdline']):
                        print(f"🛑 Found SGLang server process (PID: {proc.info['pid']}), terminating...")
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                            print("✅ Server terminated gracefully")
                            killed = True
                        except psutil.TimeoutExpired:
                            print("⚠️ Graceful termination timeout, force killing...")
                            proc.kill()
                            proc.wait()
                            print("✅ Server force killed")
                            killed = True
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if not killed:
                # Fallback: try to find by port using lsof
                try:
                    result = subprocess.run(['lsof', '-ti:11451'], capture_output=True, text=True)
                    if result.returncode == 0 and result.stdout.strip():
                        pid = result.stdout.strip()
                        print(f"🛑 Found process using port 11451 (PID: {pid}), terminating...")
                        subprocess.run(['kill', '-TERM', pid])
                        time.sleep(2)
                        # Check if still running, force kill if needed
                        result = subprocess.run(['kill', '-0', pid], capture_output=True)
                        if result.returncode == 0:
                            print("⚠️ Graceful termination timeout, force killing...")
                            subprocess.run(['kill', '-KILL', pid])
                        print("✅ Server terminated")
                        killed = True
                except subprocess.CalledProcessError:
                    pass
            
            if not killed:
                print("🔍 No SGLang server process found to terminate")
                
        except ImportError:
            # If psutil not available, try lsof approach only
            try:
                result = subprocess.run(['lsof', '-ti:11451'], capture_output=True, text=True)
                if result.returncode == 0 and result.stdout.strip():
                    pid = result.stdout.strip()
                    print(f"🛑 Found process using port 11451 (PID: {pid}), terminating...")
                    subprocess.run(['kill', '-TERM', pid])
                    time.sleep(2)
                    # Check if still running, force kill if needed
                    result = subprocess.run(['kill', '-0', pid], capture_output=True)
                    if result.returncode == 0:
                        print("⚠️ Graceful termination timeout, force killing...")
                        subprocess.run(['kill', '-KILL', pid])
                    print("✅ Server terminated")
                else:
                    print("🔍 No process found using port 11451")
            except subprocess.CalledProcessError as e:
                print(f"❌ Error finding/killing server process: {e}")
        except Exception as e:
            print(f"❌ Error searching for server process: {e}")

def extract_thread_launch_info(text: str) -> List[Dict[str, str]]:
        """Extract thread launch information from text - only unprocessed threads"""
        
        print(f"\n🔍 DEBUG: Full text being analyzed:")
        print(f"{'='*60}")
        print(text[-1000:])  # Show last 1000 chars for debugging
        print(f"{'='*60}")
        
        # Find all launch_threads blocks
        launch_pattern = r'<launch_threads>(.*?)</launch_threads>'
        launch_matches = re.findall(launch_pattern, text, re.DOTALL)
        
        print(f"🔍 Found {len(launch_matches)} launch_threads blocks")
        
        # If no complete launch_threads blocks found, try to find incomplete ones
        if not launch_matches:
            # Look for launch_threads that might not be closed yet
            incomplete_pattern = r'<launch_threads>(.*?)$'
            incomplete_matches = re.findall(incomplete_pattern, text, re.DOTALL)
            if incomplete_matches:
                print(f"🔍 Found {len(incomplete_matches)} incomplete launch_threads blocks")
                launch_matches = incomplete_matches
            else:
                print("🔍 No launch_threads blocks found at all")
                return []
        
        # Get all existing thread results to avoid reprocessing
        result_pattern = r'<thread_result id=\'(\d+)\'>'
        existing_results = re.findall(result_pattern, text)
        existing_thread_ids = set(existing_results)
        
        print(f"🔍 Found existing thread results for IDs: {existing_thread_ids}")
        
        thread_info = []
        
        # Process all launch_threads blocks to find unprocessed threads
        for i, launch_content in enumerate(launch_matches):
            print(f"\n🔍 Processing launch_threads block {i+1}:")
            print(f"Content: {launch_content[:200]}..." if len(launch_content) > 200 else f"Content: {launch_content}")
            
            # Updated regex pattern to handle multiline task and objective content
            thread_pattern = r"<thread id='(\d+)'>\s*<task>\s*(.*?)\s*</task>\s*<objective>\s*(.*?)\s*</objective>\s*</thread>"
            threads = re.findall(thread_pattern, launch_content, re.DOTALL)
            
            print(f"🔍 Found {len(threads)} thread definitions in this block")
            
            for thread_id, task, objective in threads:
                thread_id = thread_id.strip()
                task = task.strip()
                objective = objective.strip()
                
                print(f"🔍 Processing thread {thread_id}")
                print(f"  Task: {task}")
                print(f"  Objective: {objective}")
                
                # Only add threads that don't have results yet
                if thread_id not in existing_thread_ids:
                    print(f"✅ Found unprocessed thread {thread_id}: {task}")
                    thread_info.append({
                        'id': thread_id,
                        'task': task,
                        'objective': objective
                    })
                else:
                    print(f"⏭️  Skipping already processed thread {thread_id}")
        
        print(f"\n🔍 Final result: {len(thread_info)} unprocessed threads found")
        return thread_info
def generate_main(problem, sys_prompt, prev_reasoning, model_path, host='localhost', port=11451):
    """Generation in main thread with reasoning context."""
    # Fix: call sglang_gen with correct parameters
    return sglang_gen(sys_prompt, problem + "\n" + prev_reasoning, model_path, mode="main", host=host, port=port)
def generate_thread(prev_reasoning, model_path, host='localhost', port=11451):
    # Extract the content before the last <launch_threads> tag
    thread_info = extract_thread_launch_info(prev_reasoning)
    # Find the content before the last <launch_threads> tag
    last_launch_idx = prev_reasoning.rfind("<launch_threads>")
    if last_launch_idx != -1:
        prev_reasoning_before_launch = prev_reasoning[:last_launch_idx]
    else:
        prev_reasoning_before_launch = prev_reasoning
    thread_results = "<step_resolution>"
    for thread in thread_info:
        thread_prompt = f"<thread id='{thread['id']}'>\n<task>\n{thread['task']}\n</task>\n<objective>\n{thread['objective']}\n</objective>\n</thread>"
        full_prompt = prev_reasoning_before_launch + "\n" + thread_prompt
        response = sglang_gen(SYSTEM_PROMPT_THREAD, full_prompt, model_path, mode="thread", host=host, port=port)
        
        # Add null check for response
        if response is None:
            print(f"❌ Failed to generate response for thread {thread['id']}")
            continue
            
        # extract pattern from response
        match = re.search(rf"<thread_result id='{thread['id']}'>\s*(.*?)\s*</thread_result>", response, re.DOTALL)
        if match:
            response_content = match.group(1).strip()
        else:
            response_content = response.strip()
        thread_results += f"<thread_result id='{thread['id']}'>\n{response_content}\n</thread_result>"
    thread_results += "</step_resolution>"
    return thread_results

def solve_problem(problem, model_path, host='localhost', port=11451):
    """
    Solve a problem using SGLang server.
    """
    try:
        sys_prompt = SYSTEM_PROMPT_MAIN  # Use the proper system prompt
        current_context = ""
        
        while True:
            response = generate_main(problem, sys_prompt, current_context, model_path, host, port)
            
            # Add null check for response
            if response is None:
                print("❌ Failed to generate main response")
                break
                
            current_context += response
            
            if "<launch_threads>" not in response:
                break
            else:
                # Fix: use the existing extract_thread_launch_info function
                thread_results = generate_thread(current_context, model_path, host, port)
                
                # Add null check for thread_results
                if thread_results is None:
                    print("❌ Failed to generate thread results")
                    break
                    
                current_context += thread_results
        
        print("🎉 Problem solving completed!")
        return current_context
        
    except Exception as e:
        print(f"❌ Error solving problem: {e}")
        return None
    finally:
        # Always terminate server when done
        terminate_sglang_server()

if __name__ == "__main__":
    model_path = "/nvme0n1/zzy_model/fresh_mixed_7131740/deepseek-parallel-thinking/checkpoint-1035"
    
    result = solve_problem(r"""Let ABCDEF be a convex equilateral hexagon in which all pairs of opposite sides are parallel. The triangle whose sides are extensions of segments AB, CD, and EF has side lengths 200, 240, and 300. Find the side length of the hexagon.""", model_path)
    
    