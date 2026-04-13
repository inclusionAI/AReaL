#!/usr/bin/env python
import json
import requests
import fire
import logging
import sys
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_bing_search(
    url: str = None,
    trajectory_id: str = "test-search-001",
):
    """Test Bing search functionality"""
    
    print("--- Testing 1: Basic search with <search> tags ---")
    action = """<search>Python machine learning tutorials</search>"""
    print(_send_test_request(url, trajectory_id + "-1", action, "Basic Search"))
    
    print("--- Testing 2: Search with code block format ---")
    action = """```search\nartificial intelligence latest news\n```"""
    print(_send_test_request(url, trajectory_id + "-2", action, "Code Block Search"))
    
    print("--- Testing 3: Search with search: prefix ---")
    action = """search: OpenAI GPT-4 capabilities"""
    print(_send_test_request(url, trajectory_id + "-3", action, "Prefix Search"))
    
    print("--- Testing 4: Chinese language search ---")
    action = """<search>深度学习算法</search>"""
    print(_send_test_request(url, trajectory_id + "-4", action, "Chinese Search"))
    
    print("--- Testing 5: Complex search query ---")
    action = """<search>"machine learning" AND "neural networks" best practices 2024</search>"""
    print(_send_test_request(url, trajectory_id + "-5", action, "Complex Query"))
    
    print("--- Testing 6: Multiple search tags (should use first one) ---")
    action = """<search>first query</search> some text <search>second query</search>"""
    print(_send_test_request(url, trajectory_id + "-6", action, "Multiple Search Tags"))
    
    print("--- Testing 7: Empty search query ---")
    action = """<search></search>"""
    print(_send_test_request(url, trajectory_id + "-7", action, "Empty Query"))
    
    print("--- Testing 8: Invalid format (no search query) ---")
    action = """This is just regular text without any search tags"""
    print(_send_test_request(url, trajectory_id + "-8", action, "Invalid Format"))
    
    print("--- Testing 9: Very long search query ---")
    long_query = "machine learning " * 50  # Create a very long query
    action = f"""<search>{long_query}</search>"""
    print(_send_test_request(url, trajectory_id + "-9", action, "Long Query"))
    
    print("--- Testing 10: Search with special characters ---")
    action = """<search>C++ programming & memory management: best practices?</search>"""
    print(_send_test_request(url, trajectory_id + "-10", action, "Special Characters"))
    
    print("--- Testing 11: Search with quotes ---")
    action = """<search>"exact phrase search" programming</search>"""
    print(_send_test_request(url, trajectory_id + "-11", action, "Quoted Search"))
    
    print("--- Testing 12: Search with extra field timeout ---")
    action = """<search>fast search query</search>"""
    extra_field = {"timeout": 30}
    print(_send_test_request_with_extra(url, trajectory_id + "-12", action, extra_field, "Custom Timeout"))
    
    print("--- Testing 13: Nested code block format ---")
    action = """
    Here's my search:
    ```search
    Python web scraping libraries comparison
    ```
    Please find relevant information.
    """
    print(_send_test_request(url, trajectory_id + "-13", action, "Nested Code Block"))
    
    print("--- Testing 14: Cache test (repeat previous query) ---")
    action = """<search>Python machine learning tutorials</search>"""
    print(_send_test_request(url, trajectory_id + "-14", action, "Cache Test"))
    
    return True

def test_bing_search_edge_cases(
    url: str = None,
    trajectory_id: str = "test-search-edge-001",
):
    """Test edge cases for Bing search"""
    
    print("--- Edge Case 1: Malformed XML-like tags ---")
    action = """<search>unclosed search tag"""
    print(_send_test_request(url, trajectory_id + "-1", action, "Malformed Tags"))
    
    print("--- Edge Case 2: Nested search tags ---")
    action = """<search>outer <search>inner</search> query</search>"""
    print(_send_test_request(url, trajectory_id + "-2", action, "Nested Tags"))
    
    print("--- Edge Case 3: Mixed formats ---")
    action = """<search>xml format</search> and ```search\ncode block format\n```"""
    print(_send_test_request(url, trajectory_id + "-3", action, "Mixed Formats"))
    
    print("--- Edge Case 4: Search with newlines ---")
    action = """<search>
    multi-line
    search query
    with newlines
    </search>"""
    print(_send_test_request(url, trajectory_id + "-4", action, "Multi-line Query"))
    
    print("--- Edge Case 5: Unicode characters ---")
    action = """<search>机器学习 🤖 人工智能 émojis café naïve</search>"""
    print(_send_test_request(url, trajectory_id + "-5", action, "Unicode Search"))
    
    return True

def test_bing_search_performance(
    url: str = None,
    trajectory_id: str = "test-search-perf-001",
    num_requests: int = 5
):
    """Test performance with multiple concurrent-like requests"""
    
    print(f"--- Performance Test: {num_requests} sequential requests ---")
    
    queries = [
        "artificial intelligence",
        "machine learning algorithms", 
        "deep learning frameworks",
        "natural language processing",
        "computer vision techniques"
    ]
    
    for i in range(num_requests):
        query = queries[i % len(queries)]
        action = f"""<search>{query} {i}</search>"""
        print(f"\n--- Request {i+1}/{num_requests} ---")
        result = _send_test_request(url, f"{trajectory_id}-{i}", action, f"Performance Test {i+1}")
    
    return True

def _send_test_request(url, trajectory_id, action, test_name):
    """Helper function to send test requests and process responses"""
    return _send_test_request_with_extra(url, trajectory_id, action, {}, test_name)

def _send_test_request_with_extra(url, trajectory_id, action, extra_field, test_name):
    """Helper function to send test requests with extra fields and process responses"""
    logger.info(f"Testing {test_name} search...")
    
    # Use server API
    payload = {
        "trajectory_ids": [trajectory_id],
        "actions": [action],
        "extra_fields": [extra_field]
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
            
            # Check if search was successful
            if "Search results for" in observation:
                logger.info(f"✓ {test_name}: Search executed successfully")
            elif "No valid search query" in observation:
                logger.warning(f"⚠ {test_name}: Invalid search query format")
            elif "Search failed" in observation:
                logger.error(f"✗ {test_name}: Search failed")
            else:
                logger.info(f"? {test_name}: Unexpected response format")
        else:
            logger.error(f"No observation found in response for {test_name}")
        
        # Print additional response details
        if "dones" in result:
            logger.info(f"Done status: {result['dones']}")
        if "valids" in result:
            logger.info(f"Valid status: {result['valids']}")
        
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
        python -m verl_tool.servers.tests.test_bing_search_tool bing_search --url=http://localhost:5000/get_observation
        python -m verl_tool.servers.tests.test_bing_search_tool edge_cases --url=http://localhost:5000/get_observation
        python -m verl_tool.servers.tests.test_bing_search_tool performance --url=http://localhost:5000/get_observation --num_requests=10
    """
    fire.Fire({
        "bing_search": test_bing_search,
        "edge_cases": test_bing_search_edge_cases,
        "performance": test_bing_search_performance,
    })

if __name__ == "__main__":
    main()