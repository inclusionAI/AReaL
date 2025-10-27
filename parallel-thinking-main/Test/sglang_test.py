import sglang as sgl
from sglang import gen, set_default_backend, RuntimeEndpoint
import requests
import time
import subprocess

def wait_for_server(host="127.0.0.1", port=30002, timeout=300):
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

@sgl.function
def simple_generation(s, prompt: str, max_tokens: int = 100):
    """Simple generation function using gen"""
    s += prompt
    s += gen("response", max_tokens=max_tokens, temperature=0.7, top_p=0.9, stop=["</thread_result>", "<|im_end|>", "<|endoftext|>", "</s>"])

def test_gen_function():
    """Test the gen function with a simple prompt"""
    
    # Model path - adjust this to your model
    model_path = "/nvme1n1/zzy/parallel_7230838/deepseek-parallel-thinking"
    host = "127.0.0.1"
    port = 30002
    
    # Start SGLang server
    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--host", host,
        "--port", str(port),
        "--tp-size", "1",
        "--max-total-tokens", "2048",
        "--mem-fraction-static", "0.85",
        "--disable-cuda-graph",
    ]
    
    print("🚀 Starting SGLang server...")
    server_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    try:
        # Wait for server to be ready
        if not wait_for_server(host, port, timeout=300):
            print("❌ Failed to start server")
            return
            
        # Connect to server
        backend = RuntimeEndpoint(f"http://{host}:{port}")
        sgl.set_default_backend(backend)
        print(f"✅ Connected to SGLang server at {host}:{port}")
        
        # Test prompt
        prompt = "output the word 'hello' one hundred times "
        
        print("\n📝 Testing gen function...")
        print(f"Prompt: {prompt}")
        
        # Generate response using the gen function
        state = simple_generation.run(prompt=prompt, max_tokens=10)
        response = state['response']
        
        print("\n✅ Generation completed!")
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        # Clean up server
        if server_process:
            server_process.terminate()
            server_process.wait()
            print("🛑 Server terminated")

if __name__ == "__main__":
    test_gen_function()
