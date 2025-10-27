# Set CUDA device FIRST, before any other imports
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import csv
from datetime import datetime
import pandas as pd
from sglang_inference import ParallelThreadProcessor
import sglang as sgl
from sglang import RuntimeEndpoint
import subprocess
import time
import requests

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

def process_problems_from_csv(csv_file_path, output_file_path, problem_column="problem", start_index=0, max_problems=None):
    """
    Process problems from CSV file and save results to text file
    
    Args:
        csv_file_path: Path to CSV file containing problems
        output_file_path: Path to output text file
        problem_column: Name of the column containing problems (default: "problem")
        start_index: Index to start processing from (default: 0)
        max_problems: Maximum number of problems to process (default: None for all)
    """
    
    # Model and server configuration
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
        "--max-total-tokens", "16384",
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
        
        # Read CSV file
        print(f"📖 Reading problems from: {csv_file_path}")
        
        try:
            df = pd.read_csv(csv_file_path)
        except Exception as e:
            print(f"❌ Error reading CSV file: {e}")
            return
        
        if problem_column not in df.columns:
            print(f"❌ Column '{problem_column}' not found in CSV. Available columns: {list(df.columns)}")
            return
        
        # Filter problems
        problems = df[problem_column].dropna().tolist()
        total_problems = len(problems)
        
        # Apply start_index and max_problems
        if start_index > 0:
            problems = problems[start_index:]
            print(f"📝 Starting from index {start_index}")
        
        if max_problems is not None:
            problems = problems[:max_problems]
            print(f"📝 Processing maximum {max_problems} problems")
        
        print(f"📊 Found {len(problems)} problems to process (out of {total_problems} total)")
        
        # Prepare output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not output_file_path.endswith('.txt'):
            output_file_path = output_file_path + '.txt'
        
        print(f"📝 Results will be saved to: {output_file_path}")
        
        # Process problems
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            # Write header
            output_file.write(f"Parallel Thinking Results\n")
            output_file.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            output_file.write(f"Model: {model_path}\n")
            output_file.write(f"Total problems processed: {len(problems)}\n")
            output_file.write(f"{'='*100}\n\n")
            
            successful_count = 0
            failed_count = 0
            
            for i, problem in enumerate(problems, start=1):
                actual_index = start_index + i - 1
                print(f"\n{'='*80}")
                print(f"🔄 Processing Problem {i}/{len(problems)} (Index: {actual_index})")
                print(f"{'='*80}")
                
                # Write problem header to output file
                output_file.write(f"PROBLEM {i} (Index: {actual_index})\n")
                output_file.write(f"{'='*80}\n")
                output_file.write(f"ORIGINAL PROBLEM:\n")
                output_file.write(f"{problem}\n\n")
                
                try:
                    # Generate response
                    first_token = "<reasoning_process>\n<parallel_processing>\n"
                    
                    print(f"📋 Problem preview: {problem[:200]}{'...' if len(problem) > 200 else ''}")
                    
                    full_response = processor.generate_with_parallel_threads(problem, first_token)
                    
                    # Write result to output file
                    output_file.write(f"GENERATED RESPONSE:\n")
                    output_file.write(f"{'-'*80}\n")
                    output_file.write(f"{full_response}\n")
                    output_file.write(f"{'-'*80}\n\n")
                    
                    successful_count += 1
                    print(f"✅ Problem {i} completed successfully!")
                    
                except Exception as e:
                    error_msg = f"❌ Error processing problem {i}: {str(e)}"
                    print(error_msg)
                    
                    # Write error to output file
                    output_file.write(f"ERROR:\n")
                    output_file.write(f"{error_msg}\n\n")
                    
                    failed_count += 1
                
                # Flush output file after each problem
                output_file.flush()
                
                # Add separator
                output_file.write(f"\n{'='*100}\n\n")
            
            # Write summary
            summary = f"""
PROCESSING SUMMARY:
Total problems: {len(problems)}
Successful: {successful_count}
Failed: {failed_count}
Success rate: {successful_count/len(problems)*100:.1f}%
"""
            
            output_file.write(summary)
            print(summary)
            
        print(f"\n🎉 All problems processed!")
        print(f"📁 Results saved to: {output_file_path}")
        
    except KeyboardInterrupt:
        print("\n⛔ Stopping...")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()  # This will show the full error stack trace
    finally:
        if server_process:
            server_process.terminate()
            print("🛑 Server stopped")

def main():
    """Main function with example usage"""
    # CUDA device already set at the top of the file
    
    # Configuration - modify these paths as needed
    csv_file_path = "/home/zhangzy/parallel-thinking/Test/b7a85281-4307-44c5-a3b1-f6ba6da1d1e2.csv"  # Path to your CSV file
    output_file_path = f"parallel_thinking_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    # CSV configuration
    problem_column = "Problem"  # Name of column containing problems
    start_index = 0  # Start from first problem
    max_problems = 30  # Process maximum problems
    
    # Check if CSV file exists
    if not os.path.exists(csv_file_path):
        print(f"❌ CSV file not found: {csv_file_path}")
        print("📝 Please create a CSV file with problems or update the path")
        
        # Create example CSV file
        example_problems = [
            "What is 2 + 2?",
            "Solve the equation: 3x + 5 = 14",
            "Find the derivative of f(x) = x² + 3x + 2",
            "What is the capital of France?",
            "Explain the concept of parallel processing in AI models"
        ]
        
        example_df = pd.DataFrame({problem_column: example_problems})
        example_df.to_csv("example_problems.csv", index=False)
        print(f"📁 Created example CSV file: example_problems.csv")
        return
    
    # Process problems
    process_problems_from_csv(
        csv_file_path=csv_file_path,
        output_file_path=output_file_path,
        problem_column=problem_column,
        start_index=start_index,
        max_problems=max_problems
    )

if __name__ == "__main__":
    main()