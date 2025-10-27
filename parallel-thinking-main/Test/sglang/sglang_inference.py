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
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
@sgl.function
def main_generation_until_threads(s, prompt_with_first_token: str, max_tokens: int = 16384):
    """Generate main thread until thread launching point"""
    s += prompt_with_first_token
    s += gen(
        "main_reasoning",
        max_tokens=max_tokens,
        temperature=0.9,
        top_p=0.9,
        stop=["</launch_threads>", "<|im_end|>", "<|endoftext|>", "</s>"]
    )

@sgl.function
def continue_main_generation(s, context: str, thread_results: str, max_tokens: int = 16384):
    """Continue main generation after thread results are available"""
    s += context + thread_results
    s += gen(
        "final_reasoning",
        max_tokens=max_tokens,
        temperature=0.9,
        top_p=0.9,
        stop=["</launch_threads>", "<|im_end|>", "<|endoftext|>", "</s>"]
    )

@sgl.function
def thread_processing(s, problem: str, context: str, thread_id: str, task: str, objective: str, max_tokens: int = 16384):
    """Process individual thread"""
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
        temperature=0.9,
        top_p=0.9,
        stop=["</thread_result>", "<|im_end|>", "<|endoftext|>", "</s>"]
    )

class ParallelThreadProcessor:
    def __init__(self, host="127.0.0.1", port=30001):
        self.host = host
        self.port = port
        self.backend = None
        # CUDA_VISIBLE_DEVICES = "0"  # Set to your desired GPU device
        
    def extract_thread_launch_info(self, text: str) -> List[Dict[str, str]]:
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
    
    def extract_thread_result(self, thread_response: str, thread_id: str) -> str:
        """Extract thread result from <thread_result> tag to end of output"""
        # Look for thread_result opening tag
        result_start_pattern = rf'<thread_result id=\'{thread_id}\'>'
        result_start_match = re.search(result_start_pattern, thread_response)
        
        if result_start_match:
            # Find the start position of the thread_result tag
            start_pos = result_start_match.start()
            # Take everything from the thread_result tag to the end
            result_content = thread_response[start_pos:].strip()
            result_content = result_content + "\n</thread_result>"  # Ensure it closes properly
            return result_content
        else:
            # If no thread_result tag found, wrap the entire response
            print(f"⚠️ No thread_result tag found for thread {thread_id}, wrapping full response")
            return f"<thread_result id='{thread_id}'>\n{thread_response.strip()}\n</thread_result>"
        
    def process_single_thread(self, problem: str, context: str, thread_info: Dict[str, str]) -> Tuple[str, str]:
        """Process a single thread and return its result"""
        try:
            print(f"\n{'='*60}")
            print(f"🚀 STARTING THREAD {thread_info['id']}")
            print(f"{'='*60}")
            print(f"Task: {thread_info['task']}")
            print(f"Objective: {thread_info['objective']}")
            print(f"{'='*60}")
            print (f"Context before thread {thread_info['id']} 🚀 processing:", context)
            state = thread_processing.run(
                problem=problem,
                context=context,
                thread_id=thread_info['id'],
                task=thread_info['task'],
                objective=thread_info['objective'],
                max_tokens=16384
            )
            
            thread_response = state['thread_response']
            
            print(f"\n📝 THREAD {thread_info['id']} RAW OUTPUT:")
            print(f"{'-'*50}")
            print(thread_response)
            print(f"{'-'*50}")
            
            thread_result = self.extract_thread_result(thread_response, thread_info['id'])
            
            print(f"\n✅ THREAD {thread_info['id']} EXTRACTED RESULT:")
            print(f"{'-'*50}")
            print(thread_result)
            print(f"{'-'*50}")
            
            print(f"✅ Thread {thread_info['id']} completed successfully!")
            return thread_info['id'], thread_result
            
        except Exception as e:
            print(f"❌ Error processing thread {thread_info['id']}: {e}")
            error_result = f"<thread_result id='{thread_info['id']}'>Error: {str(e)}</thread_result>"
            return thread_info['id'], error_result
    
    def process_threads_parallel(self, problem: str, context: str, thread_infos: List[Dict[str, str]]) -> Dict[str, str]:
        """Process multiple threads in parallel"""
        results = {}
        
        print(f"\n🔄 STARTING PARALLEL PROCESSING OF {len(thread_infos)} THREADS")
        print(f"{'='*80}")
        
        with ThreadPoolExecutor(max_workers=min(len(thread_infos), 4)) as executor:
            # Submit all thread processing tasks
            future_to_thread = {
                executor.submit(self.process_single_thread, problem, context, thread_info): thread_info['id']
                for thread_info in thread_infos
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_thread):
                thread_id = future_to_thread[future]
                try:
                    result_thread_id, result_content = future.result()
                    results[result_thread_id] = result_content
                    print(f"✅ Collected result from thread {result_thread_id}")
                except Exception as e:
                    print(f"❌ Thread {thread_id} generated an exception: {e}")
                    results[thread_id] = f"<thread_result id='{thread_id}'>Error: {str(e)}</thread_result>"
        
        print(f"\n🎉 ALL THREADS COMPLETED!")
        print(f"{'='*80}")
        
        return results
    
    def generate_with_parallel_threads(self, problem: str, first_token: str = "<reasoning_process>") -> str:
        """Main function to generate with parallel thread processing - supports multiple turns"""
        print("🚀 STARTING PARALLEL THREAD GENERATION...")
        print(f"{'='*80}")
        
        # Step 1: Generate main reasoning until thread launch
        prompt_with_first_token = problem + first_token
        
        print("📝 STEP 1: Generating main reasoning until thread launch...")
        main_state = main_generation_until_threads.run(
            prompt_with_first_token=prompt_with_first_token,
            max_tokens=16384
        )
        
        main_reasoning = main_state['main_reasoning']
        
        # Check if we need to add the closing tag
        if not main_reasoning.endswith("</launch_threads>"):
            main_reasoning += "</launch_threads>\n"
        
        current_context = prompt_with_first_token + main_reasoning
        
        print(f"\n📋 MAIN THREAD CONTEXT BEFORE THREAD LAUNCH:")
        print(f"{'='*80}")
        print(current_context)
        print(f"{'='*80}")
        
        # Outer loop for main thread turns
        main_turn_count = 1
        while True:
            print(f"\n🔄 MAIN TURN {main_turn_count}: Starting main thread processing...")
            
            # Inner loop for subthread processing within this main turn
            thread_turn_count = 1
            while True:
                print(f"\n🔄 MAIN TURN {main_turn_count}, THREAD TURN {thread_turn_count}: Looking for threads to process...")
                
                # Extract thread information from current context
                thread_infos = self.extract_thread_launch_info(current_context)
                
                if not thread_infos:
                    print(f"❌ No threads found in main turn {main_turn_count}, thread turn {thread_turn_count}")
                    break  # Break inner loop, continue main thread
                
                print(f"\n🔍 Found {len(thread_infos)} threads to process:")
                for thread_info in thread_infos:
                    print(f"  • Thread {thread_info['id']}: {thread_info['task']}")
                    print(f"    Objective: {thread_info['objective']}")
                
                # Process threads in parallel
                print(f"\n🔄 Processing threads in parallel...")
                thread_results = self.process_threads_parallel(problem, current_context, thread_infos)
                
                # Combine thread results
                print(f"\n📊 Combining thread results...")
                combined_results = ""
                for thread_id in sorted(thread_results.keys()):
                    combined_results += thread_results[thread_id] + "\n"
                
                print(f"\n📋 COMBINED THREAD RESULTS:")
                print(f"{'='*80}")
                print(combined_results)
                print(f"{'='*80}")
                
                # Update context with thread results
                current_context =  current_context + "<step_resolution>\n" + combined_results + "</step_resolution>\n" + "</parallel_processing>\n"
                
                thread_turn_count += 1
                if thread_turn_count > 5:  # Safety limit for thread turns
                    print(f"⚠️ Reached maximum thread turn limit (5)")
                    break
            
            # After all threads in this turn are processed, continue main generation
            print(f"\n📝 MAIN TURN {main_turn_count}: Continuing main generation...")
            current_context = current_context + "<think: type = ''>"
            final_state = continue_main_generation.run(
                context=current_context,
                thread_results="",  # Results already added to context
                max_tokens=16384
            )
            
            final_reasoning = final_state['final_reasoning']
            if not final_reasoning.endswith("</launch_threads>") and "<launch_threads>" in final_reasoning:
                final_reasoning += "</launch_threads>\n"
            print(f"\n📝 MAIN TURN {main_turn_count} CONTINUATION:")
            print(f"{'='*80}")
            print(final_reasoning)
            print(f"{'='*80}")
            
            # Update current context for next main turn
            current_context = current_context + final_reasoning
            
            print(f"\n📋 UPDATED CONTEXT AFTER MAIN TURN {main_turn_count}:")
            print(f"{'='*80}")
            print(current_context[-500:])  # Show last 500 chars
            print(f"{'='*80}")
            
            # Check if the new continuation contains more thread launches
            if "<launch_threads>" not in final_reasoning:
                print(f"✅ No more thread launches found after main turn {main_turn_count}, ending generation")
                break  # Break outer loop, end generation
            
            main_turn_count += 1
            if main_turn_count > 10:  # Safety limit for main turns
                print(f"⚠️ Reached maximum main turn limit (10)")
                break
        
        print(f"\n🎉 MULTI-TURN GENERATION COMPLETE! Main turns: {main_turn_count}")
        return current_context

def wait_for_server(host="127.0.0.1", port=30001, timeout=300):
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

def main():
    model_path = "/nvme0n1/zzy_model/fresh_mixed_741808/deepseek-parallel-thinking/"
    host = "127.0.0.1"
    port = 30001
    
    # Start server
    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--host", host,
        "--port", str(port),
        "--tp-size", "1",
        "--max-total-tokens", "16384",  # Increased for parallel processing
        "--mem-fraction-static", "0.85",
        "--disable-cuda-graph",
        "--chunked-prefill-size", "1024",
    ]
    
    print(f"🚀 Starting SGLang server...")
    server_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    try:
        # Wait for server
        if not wait_for_server(host, port, timeout=300):
            print("❌ Failed to start server")
            return
            
        # Connect to server
        backend = RuntimeEndpoint(f"http://{host}:{port}")
        sgl.set_default_backend(backend)
        print(f"✅ Connected to SGLang server at {host}:{port}")
        
        # Initialize processor
        processor = ParallelThreadProcessor(host, port)
        
        # Test problem
        math_problem = """The 27 cells of a $3\times9$ grid are filled in using the numbers 1 through 9 so that each row contains 9 different numbers, and each of the three $3\times3$ blocks heavily outlined in the example below contains 9 different numbers, as in the first three rows of a Sudoku puzzle. | 4 | 2 | 8 | 9 | 6 | 3 | 1 | 7 | 5 | | 3 | 7 | 9 | 5 | 2 | 1 | 6 | 8 | 4 | | 5 | 6 | 1 | 8 | 4 | 7 | 9 | 2 | 3 | The number of different ways to fill such a grid can be written as $p^a\cdot q^b\cdot r^c\cdot s^d$, where $p,q,r,$ and $s$ are distinct prime numbers and $a,b,c,$ and $d$ are positive integers. Find $p\cdot a+q\cdot b+r\cdot c+s\cdot d$.\n"""
        
        first_token = "<reasoning_process>\n<parallel_processing>\n"
        
        # Generate with parallel threads
        print(f"\n🎯 PROBLEM TO SOLVE:")
        print(f"{'='*80}")
        print(math_problem)
        print(f"{'='*80}")
        
        full_response = processor.generate_with_parallel_threads(math_problem, first_token)
        
        print(f"\n🎉 PARALLEL THREAD GENERATION COMPLETE!")
        print(f"{'='*80}")
        print(f"📋 FINAL COMPLETE RESPONSE:")
        print(f"{'='*80}")
        print(full_response)
        print(f"{'='*80}")
        
    except KeyboardInterrupt:
        print("\n⛔ Stopping...")
    finally:
        if server_process:
            server_process.terminate()
            print("🛑 Server stopped")

if __name__ == "__main__":
    main()