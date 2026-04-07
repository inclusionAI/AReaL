#!/usr/bin/env python
"""
Simple IPython Server Stress Test - All-in-One Script
Just run it and see the results!

Usage:
    python simple_stress_test.py --url=http://localhost:5000/get_observation
    
Optional arguments:
    --url=URL                Server URL (required)
    --requests=100          Number of requests to send (default: 100)
    --concurrency=10        Concurrent workers (default: 10)
"""

import json
import requests
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List

# Simple color output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.END}")

@dataclass
class TestResult:
    success: bool
    response_time: float
    trajectory_id: str
    error: str = None

class SimpleStressTest:
    def __init__(self, url: str):
        self.url = url
    
    def run_test(self, trajectory_id: str, action: str) -> TestResult:
        """Run a single test"""
        start = time.time()
        
        try:
            payload = {
                "trajectory_ids": [trajectory_id],
                "actions": [action],
                "extra_fields": [{}]
            }
            
            response = requests.post(self.url, json=payload, timeout=30)
            elapsed = time.time() - start
            
            return TestResult(
                success=response.status_code == 200,
                response_time=elapsed,
                trajectory_id=trajectory_id,
                error=None if response.status_code == 200 else response.text
            )
        except Exception as e:
            return TestResult(
                success=False,
                response_time=time.time() - start,
                trajectory_id=trajectory_id,
                error=str(e)
            )
    
    def cleanup(self, trajectory_id: str):
        """Clean up a trajectory"""
        try:
            payload = {
                "trajectory_ids": [trajectory_id],
                "actions": [""],
                "extra_fields": [{}],
                "finish": [True]
            }
            requests.post(self.url, json=payload, timeout=10)
        except:
            pass  # Silent fail on cleanup
    
    def stress_test(self, num_requests: int = 100, concurrency: int = 10):
        """Run the stress test"""
        
        print_header("IPython Server Stress Test")
        
        print(f"Server URL:     {self.url}")
        print(f"Total Requests: {num_requests}")
        print(f"Concurrency:    {concurrency}")
        print()
        
        # Test simple execution
        action = """<python>print('Hello from stress test!')</python>"""
        
        print("Starting test...")
        results = []
        trajectory_ids = []
        
        start_time = time.time()
        
        def run_test_and_close(trajectory_id: str, action: str) -> TestResult:
            result = self.run_test(trajectory_id, action)
            self.cleanup(trajectory_id)
            return result
        
        # Run tests concurrently
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []
            for i in range(num_requests):
                trajectory_id = f"stress-test-{i:06d}"
                trajectory_ids.append(trajectory_id)
                future = executor.submit(run_test_and_close, trajectory_id, action)
                futures.append(future)
            
            # Collect results with progress
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                completed += 1
                
                if completed % max(1, num_requests // 10) == 0:
                    print(f"  Progress: {completed}/{num_requests} requests completed...")
        
        total_time = time.time() - start_time
        
        # # Clean up all trajectories
        # print("\nCleaning up resources...")
        # with ThreadPoolExecutor(max_workers=concurrency) as executor:
        #     for tid in trajectory_ids:
        #         executor.submit(self.cleanup, tid)
        
        # Calculate statistics
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        if not successful:
            print_error("All requests failed!")
            return
        
        response_times = sorted([r.response_time for r in successful])
        
        avg_time = sum(response_times) / len(response_times)
        min_time = min(response_times)
        max_time = max(response_times)
        median_time = response_times[len(response_times) // 2]
        p95_time = response_times[int(len(response_times) * 0.95)]
        rps = len(successful) / total_time
        
        # Print results
        print_header("Test Results")
        
        print(f"{Colors.BOLD}Summary:{Colors.END}")
        print(f"  Total Requests:     {num_requests}")
        if len(successful) == num_requests:
            print_success(f"Successful:         {len(successful)} (100%)")
        elif len(successful) > num_requests * 0.95:
            print_warning(f"Successful:         {len(successful)} ({len(successful)/num_requests*100:.1f}%)")
        else:
            print_error(f"Successful:         {len(successful)} ({len(successful)/num_requests*100:.1f}%)")
        
        if failed:
            print_error(f"Failed:             {len(failed)}")
        
        print()
        print(f"{Colors.BOLD}Performance:{Colors.END}")
        print(f"  Total Time:         {total_time:.2f}s")
        print(f"  Throughput:         {rps:.2f} requests/second")
        
        print()
        print(f"{Colors.BOLD}Response Times:{Colors.END}")
        print(f"  Average:            {avg_time:.3f}s")
        print(f"  Median:             {median_time:.3f}s")
        print(f"  Min:                {min_time:.3f}s")
        print(f"  Max:                {max_time:.3f}s")
        print(f"  95th Percentile:    {p95_time:.3f}s")
        
        # Health assessment
        print()
        print(f"{Colors.BOLD}Health Assessment:{Colors.END}")
        
        success_rate = len(successful) / num_requests
        if success_rate >= 0.99:
            print_success(f"Success Rate: {success_rate*100:.1f}% - Excellent!")
        elif success_rate >= 0.95:
            print_warning(f"Success Rate: {success_rate*100:.1f}% - Good, but some failures")
        else:
            print_error(f"Success Rate: {success_rate*100:.1f}% - Poor, investigate issues")
        
        if p95_time < avg_time * 2:
            print_success(f"95th Percentile: {p95_time/avg_time:.2f}x average - Consistent performance")
        else:
            print_warning(f"95th Percentile: {p95_time/avg_time:.2f}x average - High variance")
        
        if rps >= 10:
            print_success(f"Throughput: {rps:.1f} req/s - Good")
        elif rps >= 5:
            print_warning(f"Throughput: {rps:.1f} req/s - Moderate")
        else:
            print_error(f"Throughput: {rps:.1f} req/s - Low")
        
        print()
        print_success("Test completed! All resources cleaned up.")
        print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")

def main():
    """Main function"""
    # Parse simple command line arguments
    url = None
    num_requests = 100
    concurrency = 10
    
    for arg in sys.argv[1:]:
        if arg.startswith('--url='):
            url = arg.split('=', 1)[1]
        elif arg.startswith('--requests='):
            num_requests = int(arg.split('=', 1)[1])
        elif arg.startswith('--concurrency='):
            concurrency = int(arg.split('=', 1)[1])
        elif arg in ['-h', '--help']:
            print(__doc__)
            sys.exit(0)
    
    if not url:
        print_error("Error: --url is required!")
        print("\nUsage:")
        print("  python simple_stress_test.py --url=http://localhost:5000/get_observation")
        print("\nOptional:")
        print("  --requests=100       Number of requests (default: 100)")
        print("  --concurrency=10     Concurrent workers (default: 10)")
        print("  --help               Show this help message")
        sys.exit(1)
    
    # Run the test
    tester = SimpleStressTest(url)
    tester.stress_test(num_requests, concurrency)

if __name__ == "__main__":
    main()