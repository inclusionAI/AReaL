#!/usr/bin/env python
import json
import requests
import fire
import logging
import sys
import os

# Add parent directory to path to import SandboxFusionTool
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.sandbox_fusion import SandboxFusionTool

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_sandbox_fusion(
    url: str = None,
    trajectory_id: str = "test-sandbox-fusion-001",
):
    """Test SandboxFusion code execution with multiple languages"""
    
    # Test Python execution
    print("--- Testing Python 1: Basic execution ---")
    action = """<python>print('Hello from Python via SandboxFusion!')</python>"""
    print(_send_test_request(url, trajectory_id, action, "Python Basic"))
    
    print("--- Testing Python 2: Computation ---")
    action = """```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

for i in range(10):
    print(f"Fibonacci({i}) = {fibonacci(i)}")
```"""
    print(_send_test_request(url, trajectory_id, action, "Python Fibonacci"))
    
    # Test JavaScript execution
    print("--- Testing JavaScript 1: Basic execution ---")
    action = """<javascript>console.log('Hello from JavaScript via SandboxFusion!');</javascript>"""
    print(_send_test_request(url, trajectory_id, action, "JavaScript Basic"))
    
    print("--- Testing JavaScript 2: Array operations ---")
    action = """```javascript
const numbers = [1, 2, 3, 4, 5];
const doubled = numbers.map(n => n * 2);
console.log('Original:', numbers);
console.log('Doubled:', doubled);
console.log('Sum:', numbers.reduce((a, b) => a + b, 0));
```"""
    print(_send_test_request(url, trajectory_id, action, "JavaScript Arrays"))
    
    # Test C++ execution
    print("--- Testing C++ ---")
    action = """<cpp>
#include <iostream>
#include <vector>

int main() {
    std::cout << "Hello from C++ via SandboxFusion!" << std::endl;
    
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    std::cout << "Numbers: ";
    for (int n : numbers) {
        std::cout << n << " ";
    }
    std::cout << std::endl;
    
    return 0;
}
</cpp>"""
    print(_send_test_request(url, trajectory_id, action, "C++"))
    
    # Test Go execution
    print("--- Testing Go ---")
    action = """```go
package main

import (
    "fmt"
)

func main() {
    fmt.Println("Hello from Go via SandboxFusion!")
    
    // Create a slice
    numbers := []int{1, 2, 3, 4, 5}
    fmt.Println("Numbers:", numbers)
    
    // Calculate sum
    sum := 0
    for _, num := range numbers {
        sum += num
    }
    fmt.Printf("Sum of numbers: %d\n", sum)
}
```"""
    print(_send_test_request(url, trajectory_id, action, "Go"))
    
    # Test edge cases
    print("--- Testing timeout case ---")
    action = """<python>
import time
print("Starting sleep...")
time.sleep(35)  # This should exceed the default timeout
print("Done sleeping")
</python>"""
    print(_send_test_request(url, trajectory_id, action, "Timeout"))
    
    print("--- Testing syntax error ---")
    action = """<python>
prnit("This has a typo and will fail")
</python>"""
    print(_send_test_request(url, trajectory_id, action, "Syntax Error"))
    
    print("--- Testing multiple code blocks ---")
    action = """Here's some Python code: <python>print("First code block")</python>
And here's some JavaScript: <javascript>console.log("Second code block")</javascript>"""
    print(_send_test_request(url, trajectory_id, action, "Multiple Blocks"))
    
    print("--- Testing safety checks ---")
    action = """<python>
import os
os.system("ls -la")  # This should be blocked by safety checks
</python>"""
    print(_send_test_request(url, trajectory_id, action, "Safety Check"))
    
    return True
    
def _send_test_request(url, trajectory_id, action, test_name):
    """Helper function to send test requests and process responses"""
    logger.info(f"Testing {test_name} code execution...")
    
    # Use server API
    payload = {
        "trajectory_ids": [trajectory_id],
        "actions": [action],
        "extra_fields": [{}]
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise exception for error status codes
        
        result = response.json()
        logger.info(f"Response received for {test_name} test")
        
        # Print observation
        if "observations" in result and len(result["observations"]) > 0:
            observation = result["observations"][0]
            logger.info(f"\n--- {test_name} Result ---\n{observation}\n")
        else:
            logger.error(f"No observation found in response for {test_name}")
        
        return result
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {"error": str(e)}

def test_sandbox_fusion_batch(
    url: str = None,
    trajectory_id: str = "test-sandbox-fusion-batch-001",
):
    """Test batch processing of multiple test cases at once"""
    
    test_cases = [
        {
            "name": "Python Basic",
            "action": """<python>print('Hello from Python!')</python>"""
        },
        {
            "name": "Ruby Basic",
            "action": """<ruby>puts 'Hello from Ruby!'</ruby>"""
        },
        {
            "name": "Java Basic",
            "action": """<java>
public class Main {
    public static void main(String[] args) {
        System.out.println("Hello from Java!");
    }
}
</java>"""
        },
    ]
    
    results = {}
    for test_case in test_cases:
        logger.info(f"Running batch test: {test_case['name']}")
        payload = {
            "trajectory_ids": [f"{trajectory_id}-{test_case['name']}"],
            "actions": [test_case['action']],
            "extra_fields": [{}]
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            
            if "observations" in result and len(result["observations"]) > 0:
                results[test_case['name']] = result["observations"][0]
                logger.info(f"\n--- {test_case['name']} Result ---\n{result['observations'][0]}\n")
            else:
                results[test_case['name']] = "No observation found"
                logger.error(f"No observation found in response for {test_case['name']}")
                
        except Exception as e:
            results[test_case['name']] = f"Error: {str(e)}"
            logger.error(f"Error in {test_case['name']}: {str(e)}")
    
    return results

def main():
    """Main entry point for the test script
    Run with:
        python -m verl_tool.servers.tests.test_sandbox_fusion_tool sandbox --url=http://localhost:5000/get_observation
        python -m verl_tool.servers.tests.test_sandbox_fusion_tool batch --url=http://localhost:5000/get_observation
    """
    fire.Fire({
        "sandbox": test_sandbox_fusion,
        "batch": test_sandbox_fusion_batch,
    })

if __name__ == "__main__":
    main()