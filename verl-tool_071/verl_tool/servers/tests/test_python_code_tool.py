#!/usr/bin/env python
import json
import requests
import fire
import logging
import sys
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_python(
    url: str = None,
    trajectory_id: str = "test-python-001",
):
    """Test Python code execution"""
    
    print("--- Testing 1 ---")
    action = """<python>print('Hello from Python!')</python> ..."""
    print(_send_test_request(url, trajectory_id, action, "Python"))
    
#     print("--- Testing 2 ---")
#     action = """<python>import sys\n\nprint('Hello from Python!')\nprint(f'Arguments: {sys.argv[1:]}')\nfor i in range(5):\n    print(f'Number {i}')</python> ..."""
#     print(_send_test_request(url, trajectory_id, action, "Python"))
    
#     print("--- Testing 3 ---")
#     action = """```python\nprint('Hello from Python!')\n``` ..."""
#     print(_send_test_request(url, trajectory_id, action, "Python"))
    
#     print("--- Testing 3.1 ---")
#     action = """```python\nx = 10\nprint(f'Value of x is {x}')\n``` ..."""
#     print(_send_test_request(url, trajectory_id, action, "Python"))
    
#     print("--- Testing 3.2 ---")
#     action = """```python\nprint(x)\n``` ..."""
#     print(_send_test_request(url, trajectory_id, action, "Python"))
    
#     print("--- Testing 3.3 ---")
#     action = """```python\nraise ValueError('This is a test error')\n``` ..."""
#     print(_send_test_request(url, trajectory_id, action, "Python"))
    
#     print("--- Testing 3.4 ---")
#     action = """```python\nprint('This will run after error')\nprint(x)\n``` ..."""
#     print(_send_test_request(url, trajectory_id, action, "Python"))
    
#     print("--- Testing 3.5 ---")
#     action = """
# ```python
# import sympy as sp

# # define sqrt133
# sqrt133 = sp.sqrt(133)

# # define points
# A = sp.Point(0,0)
# F = sp.Point(sqrt133, 6)
# N = sp.Point(40 - 5*sqrt133, -30)
# B = sp.Point(28, 0)
# C = sp.Point(7*sqrt133, 42)
# E = sp.Point(20, 0)
# M = sp.Point(2*sqrt133 - 4, 12)

# # define polygon vertices order: A, F, N, B, C, E, M
# verts = [A, F, N, B, C, E, M]

# def polygon_area(verts):
#     # compute shoelace sum
#     n = len(verts)
#     sum1 = sum(verts[i].x*verts[(i+1)%n].y for i in range(n))
#     sum2 = sum(verts[i].y*verts[(i+1)%n].x for i in range(n))
#     area = sp.simplify(abs(sum1 - sum2)/2)
#     return area

# area_heptagon = polygon_area(verts)
# area_heptagon
# ```
# """
#     print(_send_test_request(url, trajectory_id, action, "Python"))
    
#     print("--- Testing 3.6 ---")
#     action = """
# ```python
# # compute area of quadrilateral D-E-G-F
# D = sp.Point(4,0)
# E = sp.Point(20,0)
# G = sp.Point(5*sqrt133, 30)
# F = sp.Point(sqrt133, 6)

# def quad_area(pts):
#     # pts: list of points in order
#     n = len(pts)
#     sum1 = sum(pts[i].x*pts[(i+1)%n].y for i in range(n))
#     sum2 = sum(pts[i].y*pts[(i+1)%n].x for i in range(n))
#     return sp.simplify(abs(sum1 - sum2)/2)

# area_DEGF = quad_area([D, E, G, F])
# area_DEGF
# ```
# """
#     print(_send_test_request(url, trajectory_id, action, "Python"))
    
#     print("--- Testing 3.7 ---")
#     action = """
# ```python
# area_heptagon + area_DEGF
# ```
# """
#     print(_send_test_request(url, trajectory_id, action, "Another Python"))
    
#     print("--- Testing 4 ---")
#     action = """```<python>\nprint('Hello from Python!')</python> ... <python>print('Hello again!')</python>``` ..."""
#     print(_send_test_request(url, trajectory_id, action, "Python"))
    
#     print("--- Testing 5 ---")
#     action = """```<python>import time\ntime.sleep(30)\nprint('Hello from Python!')</python> ... <python>print('Hello again!')</python>``` ..."""
#     print(_send_test_request(url, trajectory_id, action, "Python"))
    
#     print("--- Testing 6 ---") # syntax error, this prnit is intended!
#     action = """```<python>prnit('Hello from Python!')</python> ..."""
#     print(_send_test_request(url, trajectory_id, action, "Python"))

#     print("--- Testing 7 ---") # memory limit
#     action = """```<python>\nimport numpy as np\nx = np.random.rand(5000, 5000)\nsize_of_x_in_bytes = x.nbytes\nprint(f'Memory test completed after allocating a {len(x)}x{len(x[0])} array, which is {size_of_x_in_bytes / (1024 * 1024):.2f} MB.')</python> ...```"""
#     print(_send_test_request(url, trajectory_id, action, "Python Memory Test"))

#     print("--- Testing 8 ---") # memory limit
#     action = """```<python>\nimport numpy as np\nx = np.random.rand(40000, 40000)\nsize_of_x_in_bytes = x.nbytes\nprint(f'Memory test completed after allocating a {len(x)}x{len(x[0])} array, which is {size_of_x_in_bytes / (1024 * 1024):.2f} MB.')</python> ...```"""
#     print(_send_test_request(url, trajectory_id, action, "Python Memory Test"))
    
    print("--- Testing 9 ---") # test finish
    action = ""
    print(_send_test_request(url, trajectory_id, action, "test_finish", finish=[True]))

    return True
    
    
def _send_test_request(url, trajectory_id, action, test_name, **kwargs):
    """Helper function to send test requests and process responses"""
    logger.info(f"Testing {test_name} code execution...")
    
    # Use server API
    payload = {
        "trajectory_ids": [trajectory_id],
        "actions": [action],
        "extra_fields": [{}],
        **kwargs
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

def main():
    """Main entry point for the test script
    Run with:
        python -m verl_tool.servers.tests.test_python_code_tool python --url=http://localhost:5000/get_observation
    """
    fire.Fire({
        "python": test_python,
    })

if __name__ == "__main__":
    main()
