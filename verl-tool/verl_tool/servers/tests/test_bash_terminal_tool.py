#!/usr/bin/env python
import json
import requests
import fire
import logging
import sys
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_bash(
    url: str = None,
    trajectory_id: str = "test-bash-001",
):
    """Test Bash terminal command execution"""
    
    print("--- Testing 1: Basic echo command ---")
    action = """<bash>echo 'Hello from Bash!'</bash>"""
    print(_send_test_request(url, trajectory_id, action, "Bash"))
    
    print("--- Testing 2: File operations ---")
    action = """<bash>
echo 'Creating test files...'
echo 'Hello World' > test.txt
echo 'Line 2' >> test.txt
cat test.txt
ls -la
</bash>"""
    print(_send_test_request(url, trajectory_id, action, "Bash"))
    
    print("--- Testing 3: Code block format ---")
    action = """```bash
echo 'Testing code block format'
pwd
whoami
date
```"""
    print(_send_test_request(url, trajectory_id, action, "Bash"))
    
    print("--- Testing 4: Multiple command blocks ---")
    action = """<bash>echo 'First block'</bash>
<bash>echo 'Second block'</bash>"""
    print(_send_test_request(url, trajectory_id, action, "Bash"))
    
    print("--- Testing 5: Directory operations ---")
    action = """```bash
mkdir -p testdir/subdir
cd testdir
echo 'content' > file1.txt
echo 'more content' > subdir/file2.txt
find . -name "*.txt" -type f
tree . || ls -R
```"""
    print(_send_test_request(url, trajectory_id, action, "Bash"))
    
    print("--- Testing 6: Environment variables ---")
    action = """<bash>
export MY_VAR="test_value"
echo "MY_VAR is: $MY_VAR"
env | grep MY_VAR
echo $HOME
echo $PATH
</bash>"""
    print(_send_test_request(url, trajectory_id, action, "Bash"))
    
    print("--- Testing 7: Text processing ---")
    action = """```sh
echo -e "apple\nbanana\ncherry\ndate" > fruits.txt
echo "Contents of fruits.txt:"
cat fruits.txt
echo "Sorted fruits:"
sort fruits.txt
echo "Fruits containing 'a':"
grep 'a' fruits.txt
```"""
    print(_send_test_request(url, trajectory_id, action, "Bash"))
    
    print("--- Testing 8: Process information ---")
    action = """<bash>
echo "Current processes:"
ps aux | head -5
echo "System uptime:"
uptime || echo "uptime not available"
echo "Memory info:"
free -h || echo "free command not available"
</bash>"""
    print(_send_test_request(url, trajectory_id, action, "Bash"))
    
    print("--- Testing 9: Command with input ---")
    action = """```bash
echo "Please enter your name:"
read name
echo "Hello, $name!"
```
```input
TestUser
```"""
    print(_send_test_request(url, trajectory_id, action, "Bash"))
    
    print("--- Testing 10: Error handling ---")
    action = """<bash>
echo "This should work"
ls nonexistent_file.txt
echo "This should still execute despite the error above"
</bash>"""
    print(_send_test_request(url, trajectory_id, action, "Bash"))
    
    print("--- Testing 11: Timeout test (should timeout) ---")
    action = """<bash>
echo "Starting long sleep..."
sleep 15
echo "This should not appear due to timeout"
</bash>"""
    print(_send_test_request(url, trajectory_id, action, "Bash"))
    
    print("--- Testing 12: Dangerous command (should be blocked) ---")
    action = """<bash>
echo "Trying dangerous command..."
rm -rf /
echo "This should be blocked"
</bash>"""
    print(_send_test_request(url, trajectory_id, action, "Bash"))
    
    print("--- Testing 13: Network command (should be blocked) ---")
    action = """```bash
echo "Trying network command..."
curl http://google.com
echo "This should be blocked"
```"""
    print(_send_test_request(url, trajectory_id, action, "Bash"))
    
    print("--- Testing 14: Terminal format ---")
    action = """```terminal
echo 'Testing terminal format'
which bash
echo $SHELL
```"""
    print(_send_test_request(url, trajectory_id, action, "Bash"))
    
    print("--- Testing 15: Complex pipeline ---")
    action = """<bash>
echo -e "Name,Age,City\nJohn,25,NYC\nJane,30,LA\nBob,35,Chicago" > people.csv
echo "CSV contents:"
cat people.csv
echo "Processing with awk:"
awk -F',' 'NR>1 {print $1 " is " $2 " years old"}' people.csv
echo "Counting lines:"
wc -l people.csv
</bash>"""
    print(_send_test_request(url, trajectory_id, action, "Bash"))
    
    return True

def _send_test_request(url, trajectory_id, action, test_name):
    """Helper function to send test requests and process responses"""
    logger.info(f"Testing {test_name} command execution...")
    print(f"Sending action: {action}")
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

def main():
    """Main entry point for the test script
    Run with:
        python -m verl_tool.servers.tests.test_bash_terminal_tool bash --url=http://localhost:5000/get_observation
    """
    fire.Fire({
        "bash": test_bash,
    })

if __name__ == "__main__":
    main()