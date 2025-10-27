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
from transformers import AutoTokenizer
import sys
sys.path.append("parallel-thinking-main/Test")
from auto_grading import auto_grading

# Define system prompts
MAIN_SYSTEM_PROMPT = """You are a helpful assistant that solve math problems step by step."""

THREAD_SYSTEM_PROMPT = """You are a helpful assistant that solve math problems. 
Continue the reasoning process based on the context provided. Explore this reasoning path thoroughly."""


def format_chat_template(system_prompt: str, user_content: str, assistant_start: str = "") -> str:
    """Format content using chat template"""
    tokenizer = AutoTokenizer.from_pretrained("/storage/openpsi/users/zzy/model/parallel_1_5b/nvidia-parallel-thinking_1_5B_lr5/checkpoint-267")
    if assistant_start == "":
        messages = [{"role": "user", "content": user_content}]
        formatted = tokenizer.apply_chat_template(messages, tokenize = False)
        return formatted
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_start}
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize = False)
    formatted = ''.join(formatted.rsplit('<|im_end|>', 1))
    return formatted


@sgl.function
def main_generation(s, problem: str, context: str = "", max_tokens: int = 32768, turn: int = 1):
    """Generate main thread until 'Alternatively' or natural completion"""
    
    user_content = f"""Solve the following math problem. Make sure to put your answer (and only answer) inside 
    \\boxed{{}}.
    
    {problem}
    """
    
    formatted_prompt = format_chat_template(MAIN_SYSTEM_PROMPT, user_content, context)
    print (f"[DEBUG] FORMATTED PROMPT FOR MAIN: {formatted_prompt}")
    
    s += formatted_prompt 
    if turn != 1:
        s += "Alternatively"
    s += gen(
        "main_response",
        max_tokens=max_tokens,
        temperature=0.6,
        top_p=0.9,
        stop=["Alternatively", "<|im_end|>", "<|endoftext|>", "</s>"]
    )


@sgl.function
def thread_generation(s, problem: str, context: str, thread_id: int, max_tokens: int = 32768):
    """Generate for a thread based on context"""
    user_content = f"""Solve the following math problem. Make sure to put your answer (and only answer) inside 
    \\boxed{{}}.
    
    {problem}
    """
    
    formatted_prompt = format_chat_template(THREAD_SYSTEM_PROMPT, user_content, context)
    print (f"[DEBUG] FORMATTED PROMPT FOR THREAD: {formatted_prompt}")
    s += formatted_prompt + "Alternatively"
    s += gen(
        "thread_response",
        max_tokens=max_tokens,
        temperature=0.6,
        top_p=0.9,
        stop=["Alternatively", "<|im_end|>", "<|endoftext|>", "</s>"]
    )


class AlternativelyParallelProcessor:
    def __init__(self, host="127.0.0.1", port=30000, num_threads=1):
        self.host = host
        self.port = port
        self.num_threads = num_threads
        self.backend = None
        
    def remove_trailing_alternatively(self, text: str) -> str:
        """Remove 'Alternatively' from the end of text if present"""
        # Check if text ends with "Alternatively" (with possible whitespace)
        text = text.rstrip()
        if text.endswith("Alternatively"):
            text = text[:-len("Alternatively")].rstrip()
        return text
    
    def has_boxed_answer(self, text: str) -> bool:
        """Check if text contains \\boxed{ pattern"""
        return "\\boxed{" in text or "boxed{" in text
    
    def extract_boxed_answer(self, text: str) -> str:
        """Extract the answer from \\boxed{answer} pattern"""
        # Try to find boxed patterns
        patterns = [
            r'\\boxed\{([^}]+)\}',  # \boxed{answer}
            r'boxed\{([^}]+)\}',     # boxed{answer}
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            if matches:
                # Take the last match (most likely the final answer)
                answer = matches[-1].strip()
                return answer
        
        return None
    
    def process_single_thread(self, problem: str, context: str, thread_id: int) -> Tuple[int, str, int, bool]:
        """Process a single thread and return (thread_id, content, length, has_answer)"""
        try:
            print(f"\n{'='*60}")
            print(f"🚀 STARTING THREAD {thread_id}")
            print(f"{'='*60}")
            
            state = thread_generation.run(
                problem=problem,
                context=context,
                thread_id=thread_id,
                max_tokens=32768
            )
            
            thread_response = state['thread_response']
            
            # Check if thread has boxed answer before removing "Alternatively"
            has_answer = self.has_boxed_answer(thread_response)
            
            # Remove trailing "Alternatively" if present
            thread_response = self.remove_trailing_alternatively(thread_response)
            thread_response = "Alternatively" + thread_response
            print(f"\n📝 THREAD {thread_id} OUTPUT ({len(thread_response)} chars):")
            print(f"{'-'*50}")
            print(thread_response[:200] + "..." if len(thread_response) > 200 else thread_response)
            print(f"{'-'*50}")
            
            if has_answer:
                answer = self.extract_boxed_answer(thread_response)
                print(f"🎯 THREAD {thread_id} FOUND ANSWER: {answer}")
            
            print(f"✅ Thread {thread_id} completed successfully!")
            return thread_id, thread_response, len(thread_response), has_answer
            
        except Exception as e:
            print(f"❌ Error processing thread {thread_id}: {e}")
            return thread_id, f"[Error in thread {thread_id}: {str(e)}]", 0, False
    
    def process_threads_parallel(self, problem: str, context: str) -> Tuple[List[Tuple[int, str]], bool]:
        """Process multiple threads in parallel and return (sorted results, has_answer)"""
        results = []
        any_has_answer = False
        
        print(f"\n🔄 STARTING PARALLEL PROCESSING OF {self.num_threads} THREADS")
        print(f"{'='*80}")
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit all thread processing tasks
            futures = [
                executor.submit(self.process_single_thread, problem, context, i)
                for i in range(self.num_threads)
            ]
            
            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    thread_id, content, length, has_answer = future.result()
                    results.append((thread_id, content, length))
                    if has_answer:
                        any_has_answer = True
                    print(f"✅ Collected result from thread {thread_id} (length: {length}, has_answer: {has_answer})")
                except Exception as e:
                    print(f"❌ Thread generated an exception: {e}")
        
        # Sort by length (shorter first)
        results.sort(key=lambda x: x[2])
        
        print(f"\n🎉 ALL THREADS COMPLETED!")
        print(f"Thread order by length: {[f'Thread {r[0]} ({r[2]} chars)' for r in results]}")
        if any_has_answer:
            print(f"🎯 At least one thread found an answer!")
        print(f"{'='*80}")
        
        # Return (thread_id, content) tuples in sorted order, and whether any has answer
        return [(r[0], r[1]) for r in results], any_has_answer
    
    def save_turn(self, output_dir: str, problem_idx: int, turn: int, 
                  main_content: str, thread_results: List[Tuple[int, str]] = None,
                  answer: str = None):
        """Save a single turn's generation"""
        turn_dir = os.path.join(output_dir, f"problem_{problem_idx:02d}", f"turn_{turn:02d}")
        os.makedirs(turn_dir, exist_ok=True)
        
        # Save main content
        main_file = os.path.join(turn_dir, "main.txt")
        with open(main_file, "w", encoding='utf-8') as f:
            f.write(main_content)
        
        # Save thread results if any
        if thread_results:
            for thread_id, content in thread_results:
                thread_file = os.path.join(turn_dir, f"thread_{thread_id:02d}.txt")
                with open(thread_file, "w", encoding='utf-8') as f:
                    f.write(content)
            
            # Save order information
            order_file = os.path.join(turn_dir, "thread_order.json")
            with open(order_file, "w", encoding='utf-8') as f:
                json.dump({
                    "order": [tid for tid, _ in thread_results],
                    "lengths": [len(content) for _, content in thread_results]
                }, f, indent=2)
        
        # Save answer if found
        if answer:
            answer_file = os.path.join(turn_dir, "answer.txt")
            with open(answer_file, "w", encoding='utf-8') as f:
                f.write(answer)
        
        print(f"💾 Turn {turn} saved to {turn_dir}")
    
    def generate_with_alternatively_parallel(self, problem: str, output_dir: str, problem_idx: int) -> Tuple[str, str]:
        """Main function to generate with 'Alternatively' based parallel processing
        Returns (accumulated_context, answer)"""
        print("🚀 STARTING ALTERNATIVELY-BASED PARALLEL GENERATION...")
        print(f"{'='*80}")
        
        accumulated_context = ""
        turn = 0
        final_answer = None
        
        while True:
            turn += 1
            if turn ==1:
                print(f"\n{'='*100}")
                print(f"🔄 TURN {turn}: Main Generation")
                print(f"{'='*100}")
                
                # Step 1: Generate main reasoning until "Alternatively" or completion
                print("📝 Generating main reasoning...")
                main_state = main_generation.run(
                    problem=problem,
                    context=accumulated_context,
                    max_tokens=32768, 
                    turn = turn
                )
                
                main_response =  main_state['main_response']
                if turn != 1:
                    main_response = "Alternatively" + main_response 
                
                print(f"\n📋 MAIN GENERATION OUTPUT:")
                print(f"{'='*80}")
                print(main_response)
                print(f"{'='*80}")
                
                # Check if main has boxed answer
                main_has_answer = self.has_boxed_answer(main_response)
                if main_has_answer:
                    final_answer = self.extract_boxed_answer(main_response)
                    print(f"\n🎯 MAIN GENERATION FOUND ANSWER: {final_answer}")
                    accumulated_context += main_response
                    self.save_turn(output_dir, problem_idx, turn, main_response, None, final_answer)
                    break
                
                # Check if generation stopped at "Alternatively"
                stopped_at_alternatively = not main_response.endswith(("</s>", "<|im_end|>", "<|endoftext|>"))
                
                if not stopped_at_alternatively:
                    # Natural completion - save and end
                    print(f"\n✅ Main generation completed naturally (no 'Alternatively' found)")
                    accumulated_context += main_response
                    self.save_turn(output_dir, problem_idx, turn, main_response, None, None)
                    break
                
                # Stopped at "Alternatively" - launch threads
                print(f"\n🔀 Main generation stopped at 'Alternatively', launching {self.num_threads} threads...")
                
                # Update context for thread launching
                thread_context = accumulated_context + main_response
                
            # Step 2: Process threads in parallel
            thread_results, threads_have_answer = self.process_threads_parallel(problem, thread_context)
            
            # Check if any thread has answer
            if threads_have_answer:
                # Find the answer from threads
                for thread_id, content in thread_results:
                    if self.has_boxed_answer(content):
                        final_answer = self.extract_boxed_answer(content)
                        print(f"\n🎯 THREAD {thread_id} FOUND ANSWER: {final_answer}")
                        break
                
                # Save and end
                accumulated_context = thread_context
                # for thread_id, content in thread_results:
                #     accumulated_context += f"\n--- Thread {thread_id} ---\n{content}\n"
                self.save_turn(output_dir, problem_idx, turn, main_response, thread_results, final_answer)
                break
            
            # Step 3: Concatenate thread results (already sorted by length)
            print(f"\n📊 Concatenating thread results (sorted by length)...")
            combined_threads = ""
            for thread_id, content in thread_results:
                combined_threads += f"{content}"
            
            # Save thread results
            self.save_turn(output_dir, problem_idx, turn, main_response, thread_results, None)
            
            # Step 4: Update accumulated context
            accumulated_context = thread_context + combined_threads
            
            print(f"\n📋 ACCUMULATED CONTEXT LENGTH: {len(accumulated_context)} chars")
            
            # Safety check
            if turn >= 20:
                print(f"⚠️ Reached maximum turn limit (20)")
                break
        
        print(f"\n🎉 GENERATION COMPLETE! Total turns: {turn}")
        if final_answer:
            print(f"🎯 FINAL ANSWER: {final_answer}")
        
        # Save final complete result
        problem_dir = os.path.join(output_dir, f"problem_{problem_idx:02d}")
        final_file = os.path.join(problem_dir, "final_result.txt")
        with open(final_file, "w", encoding='utf-8') as f:
            f.write(accumulated_context)
        
        # Save final answer
        if final_answer:
            answer_file = os.path.join(problem_dir, "final_answer.txt")
            with open(answer_file, "w", encoding='utf-8') as f:
                f.write(final_answer)
        
        return accumulated_context, final_answer


def wait_for_server(host="127.0.0.1", port=30000, timeout=300):
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


def terminate_server_process(server_process):
    """Terminate server process and its children"""
    if server_process is None:
        return
    
    try:
        server_process.terminate()
        try:
            server_process.wait(timeout=10)
            print("🛑 Server terminated gracefully")
        except subprocess.TimeoutExpired:
            server_process.kill()
            server_process.wait()
            print("🛑 Server force killed")
    except Exception as e:
        print(f"⚠️ Error terminating server: {e}")


def process_single_problem(problem_text: str, problem_idx: int, output_dir: str, 
                          use_existing_server: bool = False, host: str = "127.0.0.1", 
                          port: int = 30000, num_threads: int = 4):
    """Process a single problem"""
    print(f"\n{'='*100}")
    print(f"🎯 PROCESSING PROBLEM {problem_idx}")
    print(f"{'='*100}")
    print(f"Problem: {problem_text[:200]}...")
    
    # Create problem directory
    problem_dir = os.path.join(output_dir, f"problem_{problem_idx:02d}")
    os.makedirs(problem_dir, exist_ok=True)
    
    model_path = "/nvme1n1/zzy/parallel_7261733/deepseek-parallel-thinking"
    server_process = None
    
    if not use_existing_server:
        # Start server
        cmd = [
            "python", "-m", "sglang.launch_server",
            "--model-path", model_path,
            "--host", host,
            "--port", str(port),
            "--tp-size", "1",
            "--max-total-tokens", "65536",
            "--mem-fraction-static", "0.6",
            "--disable-cuda-graph",
            "--chunked-prefill-size", "1024",
        ]
        
        print(f"🚀 Starting SGLang server...")
        server_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if not wait_for_server(host, port, timeout=300):
            print("❌ Failed to start server")
            terminate_server_process(server_process)
            return False
    else:
        print(f"🔗 Using existing server at {host}:{port}")
        if not wait_for_server(host, port, timeout=30):
            print("❌ Cannot connect to existing server")
            return False
    
    try:
        # Connect to server
        backend = RuntimeEndpoint(f"http://{host}:{port}")
        sgl.set_default_backend(backend)
        print(f"✅ Connected to SGLang server at {host}:{port}")
        
        # Initialize processor
        processor = AlternativelyParallelProcessor(host, port, num_threads)
        
        # Process the problem
        result, answer = processor.generate_with_alternatively_parallel(problem_text, output_dir, problem_idx)
        
        # Save the whole reasoning context to main_answer.txt (for auto_grading compatibility)
        main_answer_file = os.path.join(problem_dir, "main_answer.txt")
        with open(main_answer_file, "w", encoding='utf-8') as f:
            f.write(f"Problem {problem_idx}:\n")
            f.write(f"{problem_text}\n\n")
            f.write(f"Solution:\n")
            f.write(result)
            f.write(f"\n\nCompleted at: {datetime.now().isoformat()}\n")
        
        print(f"\n✅ Problem {problem_idx} completed successfully!")
        if answer:
            print(f"🎯 Answer: {answer}")
        
        if not use_existing_server and server_process:
            terminate_server_process(server_process)
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing problem {problem_idx}: {e}")
        if not use_existing_server and server_process:
            terminate_server_process(server_process)
        return False


def load_problems_from_jsonl(jsonl_path: str) -> Tuple[List[str], List[str]]:
    """Load problems from JSONL file
    Returns (problems, answers) where answers may contain None for conversation format
    """
    problems = []
    answers = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "conversations" in obj:
                # Extract problem from conversation format
                for conv in obj["conversations"]:
                    if conv["from"] == "human":
                        problems.append(conv["value"])
                        # Try to extract answer from gpt response if available
                        answer = None
                        for conv2 in obj["conversations"]:
                            if conv2["from"] == "gpt":
                                # Try to extract boxed answer
                                gpt_response = conv2["value"]
                                boxed_match = re.search(r'\\boxed\{([^}]+)\}', gpt_response)
                                if boxed_match:
                                    answer = boxed_match.group(1).strip()
                                break
                        answers.append(answer)
                        break
            else:
                # Original format
                problems.append(obj.get("problem", ""))
                answers.append(obj.get("answer", None))
    return problems, answers


def main(output_dir: str = None, use_existing_server: bool = False, 
         host: str = "127.0.0.1", port: int = 30000, 
         jsonl_path: str = None, num_threads: int = 4):
    """Main function to process all problems"""
    
    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join("/storage/openpsi/experiments/logs/admin/zzy_test/result", 
                                  f"alternatively_parallel_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    # Load problems and answers
    real_answers = None
    if jsonl_path:
        print(f"📖 Loading problems from: {jsonl_path}")
        problems, real_answers = load_problems_from_jsonl(jsonl_path)
    else:
        # Default test problem
        problems = [
            r"""Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop."""
        ]
        real_answers = None
    
    print(f"📊 Total problems to process: {len(problems)}")
    
    successful_count = 0
    failed_count = 0
    
    for i, problem in enumerate(problems):
        print(f"\n{'='*120}")
        print(f"📈 OVERALL PROGRESS: {i}/{len(problems)}")
        print(f"✅ Successful: {successful_count}")
        print(f"❌ Failed: {failed_count}")
        print(f"{'='*120}")
        
        success = process_single_problem(
            problem, i, output_dir, 
            use_existing_server, host, port, num_threads
        )
        
        if success:
            successful_count += 1
        else:
            failed_count += 1
        
        # Pause between problems
        if i < len(problems) - 1:
            print(f"⏸️  Pausing 10 seconds before next problem...")
            time.sleep(10)
    
    print(f"\n{'='*120}")
    print(f"🎉 ALL PROBLEMS PROCESSING COMPLETE!")
    print(f"{'='*120}")
    print(f"✅ Successful: {successful_count}")
    print(f"❌ Failed: {failed_count}")
    if successful_count + failed_count > 0:
        print(f"📊 Success rate: {successful_count/(successful_count+failed_count)*100:.1f}%")
    print(f"{'='*120}")
    
    # Run auto-grading if we have real answers
    if real_answers and any(ans is not None for ans in real_answers):
        print(f"\n{'='*120}")
        print(f"🎓 RUNNING AUTO-GRADING...")
        print(f"{'='*120}")
        try:
            grading_results = auto_grading(output_dir, real_answers)
            print(f"\n✅ Auto-grading completed!")
            print(f"Grading report saved to: {output_dir}/grading_report.txt")
        except Exception as e:
            print(f"❌ Error during auto-grading: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n⚠️ No real answers available, skipping auto-grading")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run alternatively-based parallel thinking inference")
    parser.add_argument("--use-existing-server", action="store_true", 
                       help="Use an existing SGLang server instead of starting a new one")
    parser.add_argument("--host", default="127.0.0.1", 
                       help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=30000, 
                       help="Server port (default: 30000)")
    parser.add_argument("--output-dir", 
                       help="Output directory (default: timestamp-based)")
    parser.add_argument("--jsonl-path", 
                       help="Path to JSONL file with problems")
    parser.add_argument("--num-threads", type=int, default=4,
                       help="Number of parallel threads to launch (default: 4)")
    
    args = parser.parse_args()
    
    if args.use_existing_server:
        print(f"🔗 Using existing server at {args.host}:{args.port}")
        print("Make sure you have started the SGLang server with:")
        print(f"python -m sglang.launch_server --model-path /nvme1n1/zzy/parallel_7261733/deepseek-parallel-thinking --host {args.host} --port {args.port} --tp-size 1 --max-total-tokens 65536 --mem-fraction-static 0.6 --disable-cuda-graph --chunked-prefill-size 1024")
    
    main(
        output_dir=args.output_dir,
        use_existing_server=args.use_existing_server,
        host=args.host,
        port=args.port,
        jsonl_path=args.jsonl_path,
        num_threads=args.num_threads
    )
