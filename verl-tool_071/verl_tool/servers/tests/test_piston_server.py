#!/usr/bin/env python
import json
import requests
import fire
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_piston_server(
    url: str = "http://localhost:5000/get_observation",
    language: str = "python",
    format_type: str = "xml",
    trajectory_id: str = "test-piston-server-001",
):
    """
    Test the PistonTool through the server API.
    
    Args:
        url: The URL of the server endpoint (default: http://localhost:5000/get_observation)
        language: The programming language to test (default: python)
        format_type: Format type - xml or json (default: xml)
        trajectory_id: The test trajectory ID
    
    Returns:
        True if test passed, False otherwise
    """
    
    # Prepare test code based on language and format
    action = create_test_action(language, format_type)
    
    # Prepare the request payload
    payload = {
        "trajectory_ids": [trajectory_id],
        "actions": [action],
        "extra_field": {
            "tool_type": "piston"  # Explicitly request the piston tool
        }
    }
    
    logger.info(f"Testing Piston execution for {language} via server API")
    logger.info(f"Sending request to {url}")
    
    try:
        # Send the POST request
        start_time = time.time()
        response = requests.post(url, json=payload)
        elapsed_time = time.time() - start_time
        
        # Check if request was successful
        response.raise_for_status()
        
        # Get the response data
        result = response.json()
        
        # Validate the response
        if "observations" not in result:
            logger.error("Error: Response missing 'observations' field")
            return False
        
        observations = result["observations"]
        if not observations or not isinstance(observations, list):
            logger.error(f"Error: Expected observations to be a non-empty list, got {type(observations)}")
            return False
        
        # Print the observation (code execution result)
        observation = observations[0]
        
        logger.info(f"Server response time: {elapsed_time:.2f} seconds")
        logger.info(f"\n--- {language.upper()} Result via Server ---\n{observation}\n")
        
        # Check if the observation contains expected content based on language
        success = validate_observation(language, observation)
        
        if success:
            logger.info(f"✅ {language.upper()} test via server API: PASSED")
        else:
            logger.error(f"❌ {language.upper()} test via server API: FAILED")
        
        return success
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Connection error: {e}")
        return False
    except json.JSONDecodeError:
        logger.error("Error: Invalid JSON response")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False

def create_test_action(language, format_type):
    """Create test action based on language and format type"""
    
    if language == "python":
        if format_type.lower() == "json":
            return """{
  "language": "python",
  "version": "3.10",
  "args": ["arg1", "arg2"],
  "files": [
    {
      "name": "main.py",
      "content": "import sys\\n\\nprint('Hello from Python via Server!')\\nprint(f'Arguments: {sys.argv[1:]}')\\nfor i in range(3):\\n    print(f'Number {i}')"
    }
  ]
}"""
        else:  # XML format
            return """<piston>
  <language>python</language>
  <version>3.10</version>
  <args>arg1 arg2</args>
  <file name="main.py">
import sys

print('Hello from Python via Server!')
print(f'Arguments: {sys.argv[1:]}')
for i in range(3):
    print(f'Number {i}')
  </file>
</piston>"""
            
    elif language == "cpp":
        # Use JSON format for C++ to avoid XML parsing issues with << operator
        return """{
  "language": "cpp",
  "version": "*",
  "args": ["arg1", "arg2"],
  "files": [
    {
      "name": "main.cpp",
      "content": "#include <iostream>\\n\\nint main(int argc, char* argv[]) {\\n    std::cout << \\"Hello from C++ via Server!\\" << std::endl;\\n    \\n    std::cout << \\"Arguments: \\";\\n    for(int i=1; i<argc; i++) {\\n        std::cout << argv[i] << \\" \\";\\n    }\\n    std::cout << std::endl;\\n    \\n    for(int i=0; i<3; i++) {\\n        std::cout << \\"Number \\" << i << std::endl;\\n    }\\n    \\n    return 0;\\n}"
    }
  ]
}"""
            
    elif language == "bash":
        if format_type.lower() == "json":
            return """{
  "language": "bash",
  "version": "*",
  "args": ["arg1", "arg2"],
  "files": [
    {
      "name": "script.sh",
      "content": "#!/bin/bash\\n\\necho \\"Hello from Bash via Server!\\"\\necho \\"Arguments: $@\\"\\n\\nfor i in {0..2}; do\\n    echo \\"Number $i\\"\\ndone\\n"
    }
  ]
}"""
        else:  # XML format
            return """<piston>
  <language>bash</language>
  <version>*</version>
  <args>arg1 arg2</args>
  <file name="script.sh">
#!/bin/bash

echo "Hello from Bash via Server!"
echo "Arguments: $@"

for i in {0..2}; do
    echo "Number $i"
done
  </file>
</piston>"""
            
    else:
        # Default to Python if language not recognized
        return create_test_action("python", format_type)

def validate_observation(language, observation):
    """Validate the observation contains expected content based on language"""
    
    # Common checks
    if "Error:" in observation and "Execution result:" not in observation:
        logger.error(f"Error in observation: {observation}")
        return False
    
    # Language-specific expected content
    expected_content = {
        "python": ["Hello from Python via Server", "Arguments:", "Number 0", "Number 1", "Number 2"],
        "cpp": ["Hello from C++ via Server", "Arguments:", "Number 0", "Number 1", "Number 2"],
        "bash": ["Hello from Bash via Server", "Arguments:", "Number 0", "Number 1", "Number 2"]
    }
    
    # Check if all expected strings are in the observation
    for expected in expected_content.get(language, []):
        if expected not in observation:
            logger.error(f"Expected content not found: '{expected}'")
            return False
    
    return True

def test_all_languages(url="http://localhost:5000/get_observation", format_type="xml"):
    """Test all languages through the server API"""
    
    logger.info(f"Testing all languages via server API using {format_type.upper()} format")
    results = {}
    
    languages = ["python", "cpp", "bash"]
    for lang in languages:
        results[lang] = test_piston_server(url, lang, format_type)
        # Add a small delay to avoid overwhelming the server or API rate limits
        time.sleep(1)
    
    # Report overall results
    logger.info("\n===== OVERALL TEST RESULTS =====")
    all_passed = True
    for lang, result in results.items():
        status = "PASSED" if result else "FAILED"
        logger.info(f"{lang.upper()}: {status}")
        if not result:
            all_passed = False
    
    logger.info(f"Overall test status: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed

def main():
    """
    Entry point for the test script.
    Run with: 
        python -m verl_tool.servers.tests.test_piston_server python --url=http://localhost:5000/get_observation
        python -m verl_tool.servers.tests.test_piston_server all --url=http://localhost:5000/get_observation
    """
    fire.Fire({
        "python": lambda url=None, format_type="xml": test_piston_server(url, "python", format_type),
        "cpp": lambda url=None, format_type="json": test_piston_server(url, "cpp", format_type),
        "bash": lambda url=None, format_type="xml": test_piston_server(url, "bash", format_type),
        "all": test_all_languages
    })

if __name__ == "__main__":
    main()
