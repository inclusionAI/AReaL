import asyncio
import time
from openai import AsyncOpenAI

system_prompt = """A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. User: Please integrate natural language reasoning with programs to solve the problem above. If you want to test any python code, writing it inside ```python and ``` tags following with "```output". Put your final answer within \\boxed{}.:
"""

math_problem = """Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.
"""

# Different variations of the math problem to simulate diverse requests
math_problems = [
    math_problem,
    math_problem.replace("9-kilometer", "10-kilometer").replace("4 hours", "5 hours").replace("2 hours and 24 minutes", "3 hours"),
    math_problem.replace("9-kilometer", "8-kilometer").replace("4 hours", "3 hours").replace("2 hours and 24 minutes", "1 hour and 48 minutes"),
    math_problem.replace("s+\\frac{1}{2}", "s+\\frac{2}{3}"),
    math_problem.replace("s+\\frac{1}{2}", "s+1")
]

async def send_request(client, problem_text, request_id):
    """Send a single request and measure the time it takes"""
    start_time = time.time()
    print(f"Starting request {request_id}...")
    
    try:
        completion = await client.chat.completions.create(
            model="GAIR/ToRL-1.5B",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": problem_text
                }
            ],
            temperature=0,
            max_tokens=2048,
            top_p=1,
            n=1,
        )
        
        end_time = time.time()
        print(f"Request {request_id} completed in {end_time - start_time:.2f} seconds")
        
        # Print a shortened version of the response for verification
        response_content = completion.choices[0].message.content
        print(f"Request {request_id} response (truncated): {response_content}...\n")
        
        return {
            "request_id": request_id,
            "duration": end_time - start_time,
            "response": response_content
        }
    except Exception as e:
        end_time = time.time()
        print(f"Request {request_id} failed after {end_time - start_time:.2f} seconds: {str(e)}")
        return {
            "request_id": request_id,
            "duration": end_time - start_time,
            "error": str(e)
        }

async def run_concurrent_test(num_concurrent=5, num_total=10):
    """Run multiple concurrent requests to test server performance"""
    client = AsyncOpenAI(api_key="sk-proj-1234567890", base_url="http://0.0.0.0:5000")
    
    print(f"Starting concurrent test with {num_concurrent} concurrent requests, {num_total} total requests")
    start_time = time.time()
    
    # Create tasks for all requests
    tasks = []
    for i in range(num_total):
        problem = math_problems[i % len(math_problems)]
        tasks.append(send_request(client, problem, i+1))
    
    # Run requests in batches of num_concurrent
    results = []
    for i in range(0, len(tasks), num_concurrent):
        batch = tasks[i:i+num_concurrent]
        batch_results = await asyncio.gather(*batch)
        results.extend(batch_results)
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Calculate statistics
    successful_requests = [r for r in results if "error" not in r]
    failed_requests = [r for r in results if "error" in r]
    
    if successful_requests:
        avg_request_time = sum(r["duration"] for r in successful_requests) / len(successful_requests)
    else:
        avg_request_time = 0
    
    # Print summary
    print("\n===== TEST RESULTS =====")
    print(f"Total test duration: {total_duration:.2f} seconds")
    print(f"Total requests: {num_total}")
    print(f"Successful requests: {len(successful_requests)}")
    print(f"Failed requests: {len(failed_requests)}")
    print(f"Average request time: {avg_request_time:.2f} seconds")
    print(f"Requests per second: {num_total / total_duration:.2f}")
    
    if failed_requests:
        print("\nFailed requests:")
        for req in failed_requests:
            print(f"  Request {req['request_id']}: {req['error']}")

async def sequential_test_for_comparison(num_requests=5):
    """Run sequential requests as a baseline for comparison"""
    client = AsyncOpenAI(api_key="sk-proj-1234567890", base_url="http://0.0.0.0:5000")
    
    print(f"\nStarting sequential test with {num_requests} requests for comparison")
    start_time = time.time()
    
    results = []
    for i in range(num_requests):
        problem = math_problems[i % len(math_problems)]
        result = await send_request(client, problem, f"seq-{i+1}")
        results.append(result)
    
    end_time = time.time()
    total_duration = end_time - start_time
    
    # Calculate statistics
    successful_requests = [r for r in results if "error" not in r]
    
    if successful_requests:
        avg_request_time = sum(r["duration"] for r in successful_requests) / len(successful_requests)
    else:
        avg_request_time = 0
    
    # Print summary
    print("\n===== SEQUENTIAL TEST RESULTS =====")
    print(f"Total test duration: {total_duration:.2f} seconds")
    print(f"Average request time: {avg_request_time:.2f} seconds")
    print(f"Requests per second: {num_requests / total_duration:.2f}")

async def main():
    # Run both tests
    await run_concurrent_test(num_concurrent=3, num_total=6)
    await sequential_test_for_comparison(num_requests=3)

if __name__ == "__main__":
    asyncio.run(main())