#!/usr/bin/env python
import json
import requests
import fire
import logging
import sys
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_search_retrieval(
    url: str = None,
    trajectory_id: str = "test-search-001",
):
    """Test Search Retrieval functionality"""
    
    print("--- Testing 1: Basic Search Query ---")
    action = """<search>What is machine learning?</search>"""
    print(_send_test_request(url, trajectory_id + "-1", action, "Basic Search"))
    
    print("--- Testing 2: Multi-line Search Query ---")
    action = """<search>
    How does neural network training work?
    What are the key concepts?
    </search>"""
    print(_send_test_request(url, trajectory_id + "-2", action, "Multi-line Search"))
    
    print("--- Testing 3: Search with Additional Text ---")
    action = """I need to find information about artificial intelligence.
    <search>artificial intelligence history and applications</search>
    This search should help me understand the topic better."""
    print(_send_test_request(url, trajectory_id + "-3", action, "Search with Context"))
    
    print("--- Testing 4: Multiple Search Tags (should use last one) ---")
    action = """<search>first query</search>
    Some text in between.
    <search>second query about deep learning</search>"""
    print(_send_test_request(url, trajectory_id + "-4", action, "Multiple Search Tags"))
    
    print("--- Testing 5: Answer Tag (should finish trajectory) ---")
    action = """<answer>Based on my research, machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.</answer>"""
    print(_send_test_request(url, trajectory_id + "-5", action, "Answer Tag"))
    
    print("--- Testing 6: Empty Search Query ---")
    action = """<search></search>"""
    print(_send_test_request(url, trajectory_id + "-6", action, "Empty Search"))
    
    print("--- Testing 7: Search with Special Characters ---")
    action = """<search>What is "reinforcement learning" & how does it work? (with examples)</search>"""
    print(_send_test_request(url, trajectory_id + "-7", action, "Special Characters"))
    
    print("--- Testing 8: No Valid Tags ---")
    action = """This is just plain text without any search or answer tags."""
    print(_send_test_request(url, trajectory_id + "-8", action, "No Valid Tags"))
    
    print("--- Testing 9: Malformed Tags ---")
    action = """<search>incomplete search tag without closing"""
    print(_send_test_request(url, trajectory_id + "-9", action, "Malformed Tags"))
    
    print("--- Testing 10: Long Search Query ---")
    action = """<search>I need comprehensive information about the latest developments in transformer architectures, attention mechanisms, and their applications in natural language processing, computer vision, and multimodal AI systems including GPT, BERT, Vision Transformers, and recent innovations in the field</search>"""
    print(_send_test_request(url, trajectory_id + "-10", action, "Long Search Query"))
    
    print("--- Testing 11: Search Query with Code ---")
    action = """<search>Python machine learning libraries like scikit-learn, TensorFlow, and PyTorch for beginners</search>"""
    print(_send_test_request(url, trajectory_id + "-11", action, "Search with Code Terms"))
    
    print("--- Testing 12: Mathematical/Scientific Query ---")
    action = """<search>gradient descent optimization algorithms in machine learning mathematics</search>"""
    print(_send_test_request(url, trajectory_id + "-12", action, "Mathematical Query"))
    
    return True


def test_search_retrieval_error_cases(
    url: str = None,
    trajectory_id: str = "test-search-error-001",
):
    """Test Search Retrieval error handling"""
    
    print("--- Error Testing 1: Retrieval Service Unavailable ---")
    # This test assumes the retrieval service might be down
    action = """<search>test query when service is down</search>"""
    print(_send_test_request(url, trajectory_id + "-error-1", action, "Service Unavailable"))
    
    print("--- Error Testing 2: Very Long Query (Stress Test) ---")
    long_query = "machine learning " * 1000  # Very long repeated query
    action = f"""<search>{long_query}</search>"""
    print(_send_test_request(url, trajectory_id + "-error-2", action, "Very Long Query"))
    
    print("--- Error Testing 3: Unicode and Special Characters ---")
    action = """<search>机器学习 и искусственный интеллект وذكاء اصطناعي 🤖🧠💻</search>"""
    print(_send_test_request(url, trajectory_id + "-error-3", action, "Unicode Characters"))
    
    return True


def test_search_answer_workflow(
    url: str = None,
    trajectory_id: str = "test-workflow-001",
):
    """Test complete search-answer workflow"""
    
    print("--- Workflow Testing: Search -> Answer Sequence ---")
    
    # Step 1: Initial search
    print("Step 1: Initial search")
    action1 = """<search>What are the main types of machine learning?</search>"""
    result1 = _send_test_request(url, trajectory_id, action1, "Workflow Step 1")
    
    # Step 2: Follow-up search
    print("Step 2: Follow-up search")
    action2 = """<search>supervised learning examples and applications</search>"""
    result2 = _send_test_request(url, trajectory_id, action2, "Workflow Step 2")
    
    # Step 3: Another search
    print("Step 3: Third search")
    action3 = """<search>unsupervised learning clustering algorithms</search>"""
    result3 = _send_test_request(url, trajectory_id, action3, "Workflow Step 3")
    
    # Step 4: Final answer (should end trajectory)
    print("Step 4: Final answer")
    action4 = """<answer>There are three main types of machine learning:
    1. Supervised Learning - uses labeled data to train models (e.g., classification, regression)
    2. Unsupervised Learning - finds patterns in unlabeled data (e.g., clustering, dimensionality reduction)
    3. Reinforcement Learning - learns through interaction with an environment using rewards and penalties
    Each type has different applications and use cases depending on the problem and available data.</answer>"""
    result4 = _send_test_request(url, trajectory_id, action4, "Workflow Step 4 (Final)")
    
    return True


def _send_test_request(url, trajectory_id, action, test_name):
    """Helper function to send test requests and process responses"""
    logger.info(f"Testing {test_name}...")
    
    # Use server API
    payload = {
        "trajectory_ids": [trajectory_id],
        "actions": [action],
        "extra_fields": [{}]
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()  # Raise exception for error status codes
        
        result = response.json()
        logger.info(f"Response received for {test_name}")
        
        # Print observation and metadata
        if "observations" in result and len(result["observations"]) > 0:
            observation = result["observations"][0]
            logger.info(f"\n--- {test_name} Result ---")
            logger.info(f"Observation: {observation}")
            
            # Print additional metadata if available
            if "dones" in result and len(result["dones"]) > 0:
                done = result["dones"][0]
                logger.info(f"Done: {done}")
            
            if "valids" in result and len(result["valids"]) > 0:
                valid = result["valids"][0]
                logger.info(f"Valid: {valid}")
            
            logger.info("--- End Result ---\n")
        else:
            logger.error(f"No observation found in response for {test_name}")
        
        return result
        
    except requests.exceptions.Timeout:
        logger.error(f"Request timeout for {test_name}")
        return {"error": "Request timeout"}
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection error for {test_name} - is the retrieval service running?")
        return {"error": "Connection error - check if retrieval service is running"}
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error for {test_name}: {str(e)}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error for {test_name}: {str(e)}")
        return {"error": str(e)}


def check_retrieval_service(retriever_url: str = "http://127.0.0.1:8000/retrieve"):
    """Check if the retrieval service is available"""
    logger.info("Checking retrieval service availability...")
    
    test_payload = {
        "queries": ["test query"],
        "topk": 3,
        "return_scores": True
    }
    
    try:
        response = requests.post(retriever_url, json=test_payload, timeout=10)
        response.raise_for_status()
        logger.info(f"✓ Retrieval service is available at {retriever_url}")
        return True
    except Exception as e:
        logger.warning(f"✗ Retrieval service not available at {retriever_url}: {e}")
        logger.warning("Some tests may fail if the retrieval service is not running")
        return False


def main():
    """Main entry point for the test script
    Run with:
        python -m verl_tool.servers.tests.test_search_retrieval_tool search --url=http://localhost:5000/get_observation
        python -m verl_tool.servers.tests.test_search_retrieval_tool error --url=http://localhost:5000/get_observation
        python -m verl_tool.servers.tests.test_search_retrieval_tool workflow --url=http://localhost:5000/get_observation
        python -m verl_tool.servers.tests.test_search_retrieval_tool check_service --retriever_url=http://127.0.0.1:8000/retrieve
    """
    fire.Fire({
        "search": test_search_retrieval,
        "error": test_search_retrieval_error_cases, 
        "workflow": test_search_answer_workflow,
        "check_service": check_retrieval_service,
    })


if __name__ == "__main__":
    main()