#!/usr/bin/env python
import json
import requests
import fire
import logging
import sys
import os
import io
import base64
from PIL import Image
# Add parent directory to path to import PistonTool
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.piston import PistonTool

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
def encode_image(img):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# Create JSON with the encoded image
def decode_image(img_str):
    img_data = base64.b64decode(img_str)
    img = Image.open(io.BytesIO(img_data))
    return img
def test_crop(
    url: str = None,
    trajectory_id: str = "test-crop-001",
):
    """Test Python code execution"""
    
    print("--- Testing 1 ---")
    image1 = encode_image(Image.open("/home/ma-user/work/haozhe/muze/traj/Qwen_392_t1_nop_sa1b/checkpoint-100_VStar/0/0.jpg"))

    action = """<tool_call>{"tool_name": "crop_image", "arguments": {"target_image": 1, "bbox_2d": [0, 0, 100, 100]}}</tool_call>"""
    print(_send_test_request(url, trajectory_id, action, {"image1": image1}, "crop"))
    
    # print("--- Testing 2 ---")
    # action = """<python>import sys\n\nprint('Hello from Python!')\nprint(f'Arguments: {sys.argv[1:]}')\nfor i in range(5):\n    print(f'Number {i}')</python> ..."""
    # print(_send_test_request(url, trajectory_id, action, "Python"))
    
    # print("--- Testing 3 ---")
    # action = """```python\nprint('Hello from Python!')\n``` ..."""
    # print(_send_test_request(url, trajectory_id, action, "Python"))
    
    # print("--- Testing 4 ---")
    # action = """```<python>\nprint('Hello from Python!')</python> ... <python>print('Hello again!')</python>``` ..."""
    # print(_send_test_request(url, trajectory_id, action, "Python"))
    
    # print("--- Testing 5 ---")
    # action = """```<python>import time\ntime.sleep(30)\nprint('Hello from Python!')</python> ... <python>print('Hello again!')</python>``` ..."""
    # print(_send_test_request(url, trajectory_id, action, "Python"))
    
    # print("--- Testing 6 ---") # syntax error
    # action = """```<python>prnit('Hello from Python!')</python> ..."""
    # print(_send_test_request(url, trajectory_id, action, "Python"))
    
    return True
    
    
def _send_test_request(url, trajectory_id, action, extra_field, test_name):
    """Helper function to send test requests and process responses"""
    logger.info(f"Testing {test_name} code execution...")
    
    # Use server API
    payload = {
        "trajectory_ids": [trajectory_id],
        "actions": [action],
        "extra_fields": extra_field
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
        "crop": test_crop,
    })

if __name__ == "__main__":
    main()
