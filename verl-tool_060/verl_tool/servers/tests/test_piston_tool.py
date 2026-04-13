#!/usr/bin/env python
import json
import requests
import fire
import logging
import sys
import os

# Add parent directory to path to import PistonTool
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.piston import PistonTool

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_python(
    url: str = None,
    use_local: bool = False,
    format_type: str = "xml",
    trajectory_id: str = "test-python-001",
):
    """Test Python code execution"""
    if format_type.lower() == "json":
        action = """{
  "language": "python",
  "version": "3.10",
  "args": ["arg1", "arg2"],
  "stdin": "",
  "files": [
    {
      "name": "main.py",
      "content": "import sys\\n\\nprint('Hello from Python!')\\nprint(f'Arguments: {sys.argv[1:]}')\\nfor i in range(5):\\n    print(f'Number {i}')"
    }
  ]
}"""
    else:  # XML format
        action = """<piston>
  <language>python</language>
  <version>3.10</version>
  <args>arg1 arg2</args>
  <file name="main.py">
import sys

print('Hello from Python!')
print(f'Arguments: {sys.argv[1:]}')
for i in range(5):
    print(f'Number {i}')
  </file>
</piston>"""
    
    return _send_test_request(url, trajectory_id, action, "Python", use_local)

def test_cpp(
    url: str = None,
    use_local: bool = False,
    format_type: str = "xml",
    trajectory_id: str = "test-cpp-001",
):
    """Test C++ code execution"""
    if format_type.lower() == "json":
        action = """{
  "language": "cpp",
  "version": "*",
  "args": ["arg1", "arg2"],
  "files": [
    {
      "name": "main.cpp",
      "content": "#include <iostream>\\n#include <vector>\\n\\nint main(int argc, char* argv[]) {\\n    std::cout << \\"Hello from C++!\\" << std::endl;\\n    \\n    std::cout << \\"Arguments: \\";\\n    for(int i=1; i<argc; i++) {\\n        std::cout << argv[i] << \\" \\";\\n    }\\n    std::cout << std::endl;\\n    \\n    for(int i=0; i<5; i++) {\\n        std::cout << \\"Number \\" << i << std::endl;\\n    }\\n    \\n    return 0;\\n}"
    }
  ]
}"""
    else:  # XML format
        # Use JSON format for C++ to avoid XML parsing issues with << operator
        action = """{
  "language": "cpp",
  "version": "*",
  "args": ["arg1", "arg2"],
  "files": [
    {
      "name": "main.cpp",
      "content": "#include <iostream>\\n#include <vector>\\n\\nint main(int argc, char* argv[]) {\\n    std::cout << \\"Hello from C++!\\" << std::endl;\\n    \\n    std::cout << \\"Arguments: \\";\\n    for(int i=1; i<argc; i++) {\\n        std::cout << argv[i] << \\" \\";\\n    }\\n    std::cout << std::endl;\\n    \\n    for(int i=0; i<5; i++) {\\n        std::cout << \\"Number \\" << i << std::endl;\\n    }\\n    \\n    return 0;\\n}"
    }
  ]
}"""
    
    return _send_test_request(url, trajectory_id, action, "C++", use_local)

def test_bash(
    url: str = None,
    use_local: bool = False,
    format_type: str = "xml",
    trajectory_id: str = "test-bash-001",
):
    """Test Bash code execution"""
    if format_type.lower() == "json":
        action = """{
  "language": "bash",
  "version": "*",
  "args": ["arg1", "arg2"],
  "files": [
    {
      "name": "script.sh",
      "content": "#!/bin/bash\\n\\necho \\"Hello from Bash!\\"\\necho \\"Arguments: $@\\"\\n\\nfor i in {0..4}; do\\n    echo \\"Number $i\\"\\ndone\\n\\necho \\"Current directory: $(pwd)\\"\\necho \\"Files: $(ls -la)\\"\\n"
    }
  ]
}"""
    else:  # XML format
        action = """<piston>
  <language>bash</language>
  <version>*</version>
  <args>arg1 arg2</args>
  <file name="script.sh">
#!/bin/bash

echo "Hello from Bash!"
echo "Arguments: $@"

for i in {0..4}; do
    echo "Number $i"
done

echo "Current directory: $(pwd)"
echo "Files: $(ls -la)"
  </file>
</piston>"""
    
    return _send_test_request(url, trajectory_id, action, "Bash", use_local)

def test_php(
    url: str = None,
    use_local: bool = False,
    format_type: str = "xml",
    trajectory_id: str = "test-php-001",
):
    """Test PHP code execution"""
    if format_type.lower() == "json":
        action = """{
  "language": "php",
  "version": "*",
  "args": ["arg1", "arg2"],
  "files": [
    {
      "name": "index.php",
      "content": "<?php\\n\\necho \\"Hello from PHP!\\\\n\\";\\n\\necho \\"Arguments: \\";\\nfor ($i = 1; $i < count($argv); $i++) {\\n    echo $argv[$i] . \\" \\";\\n}\\necho \\"\\\\n\\";\\n\\nfor ($i = 0; $i < 5; $i++) {\\n    echo \\"Number \\" . $i . \\"\\\\n\\";\\n}\\n\\necho \\"PHP Version: \\" . phpversion() . \\"\\\\n\\";\\n"
    }
  ]
}"""
    else:  # XML format
        action = """<piston>
  <language>php</language>
  <version>*</version>
  <args>arg1 arg2</args>
  <file name="index.php">
<?php

echo "Hello from PHP!\n";

echo "Arguments: ";
for ($i = 1; $i < count($argv); $i++) {
    echo $argv[$i] . " ";
}
echo "\n";

for ($i = 0; $i < 5; $i++) {
    echo "Number " . $i . "\n";
}

echo "PHP Version: " . phpversion() . "\n";
  </file>
</piston>"""
    
    return _send_test_request(url, trajectory_id, action, "PHP", use_local)

def test_multifile(
    url: str = None,
    use_local: bool = False,
    format_type: str = "xml",
    trajectory_id: str = "test-multifile-001",
):
    """Test multi-file code execution with Python"""
    if format_type.lower() == "json":
        action = """{
  "language": "python",
  "version": "3.10",
  "files": [
    {
      "name": "main.py",
      "content": "import helper\\n\\nprint('Multi-file test with Python')\\nprint(f'Sum: {helper.add(5, 3)}')\\nprint(f'Product: {helper.multiply(4, 2)}')\\nprint(f'Greeting: {helper.greeting(\\"World\\")}')"
    },
    {
      "name": "helper.py",
      "content": "def add(a, b):\\n    return a + b\\n\\ndef multiply(a, b):\\n    return a * b\\n\\ndef greeting(name):\\n    return f\\"Hello, {name}!\\""
    }
  ]
}"""
    else:  # XML format
        action = """<piston>
  <language>python</language>
  <version>3.10</version>
  <file name="main.py">
import helper

print('Multi-file test with Python')
print(f'Sum: {helper.add(5, 3)}')
print(f'Product: {helper.multiply(4, 2)}')
print(f'Greeting: {helper.greeting("World")}')
  </file>
  <file name="helper.py">
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def greeting(name):
    return f"Hello, {name}!"
  </file>
</piston>"""
    
    return _send_test_request(url, trajectory_id, action, "Multi-file Python", use_local)

def _send_test_request(url, trajectory_id, action, test_name, use_local=False):
    """Helper function to send test requests and process responses"""
    logger.info(f"Testing {test_name} code execution...")
    
    # Handle different execution methods
    if url is None:
        # Use PistonTool directly (no server required)
        try:
            # Initialize the tool
            tool = PistonTool(use_local=use_local)
            
            # Execute the code
            observation, done, valid = tool.conduct_action(trajectory_id, action, {})
            
            logger.info(f"\n--- {test_name} Result ---\n{observation}\n")
            return {"observations": [observation], "dones": [done], "valids": [valid]}
            
        except Exception as e:
            logger.error(f"PistonTool error: {str(e)}")
            return {"error": str(e)}
    else:
        # Use server API
        payload = {
            "trajectory_ids": [trajectory_id],
            "actions": [action],
            "extra_field": {}
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

def _run_all_tests(url=None, use_local=False, format_type="xml"):
    """Run all test cases"""
    logger.info(f"Running all tests using {format_type.upper()} format")
    results = {}
    results["python"] = test_python(url, use_local, format_type)
    results["cpp"] = test_cpp(url, use_local, format_type)
    results["bash"] = test_bash(url, use_local, format_type)
    results["php"] = test_php(url, use_local, format_type)
    results["multifile"] = test_multifile(url, use_local, format_type)
    return results

def main():
    """Main entry point for the test script
    Run with:
        python -m verl_tool.servers.tests.test_piston_tool python --url=http://localhost:5000/get_observation
        python -m verl_tool.servers.tests.test_piston_tool all --url=http://localhost:5000/get_observation
    """
    fire.Fire({
        "python": test_python,
        "cpp": test_cpp,
        "bash": test_bash,
        "php": test_php,
        "multifile": test_multifile,
        "all": _run_all_tests
    })

if __name__ == "__main__":
    main()
