#!/usr/bin/env python
import json
import requests
import fire
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_google_search(
    url: str = "http://localhost:5000/get_observation",
    query: str = "Python machine learning tutorials"
):
    """Test Google search functionality with a simple query"""
    
    logger.info(f"Testing Google search with query: '{query}'")
    
    # Simple search action with <search> tags
    action = f"<search>{query}</search>"
    trajectory_id = "google-search-test"
    
    # Prepare request payload
    payload = {
        "trajectory_ids": [trajectory_id],
        "actions": [action],
        "extra_fields": [{}]
    }
    
    try:
        # Send request
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        logger.info("Response received successfully")
        
        # Print the search results
        if "observations" in result and len(result["observations"]) > 0:
            observation = result["observations"][0]
            print(f"\n--- Google Search Results ---")
            print(f"Query: {query}")
            print(f"Results:\n{observation}\n")
            
            # Check if search was successful
            obs = observation['obs'] if isinstance(observation, dict) else observation
            if "Search results for" in obs or "results" in obs.lower():
                logger.info("✓ Google search executed successfully")
            else:
                logger.warning("⚠ Unexpected response format")
        else:
            logger.error("No observation found in response")
            
        return result
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {str(e)}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {"error": str(e)}

def test_multiple_searches(
    url: str = "http://localhost:5000/get_observation"
):
    """Test multiple Google searches with different queries"""
    
    queries = [
        "PyTorch CUDA memory optimization",
        "FSDP checkpoint loading best practices", 
        "distributed training memory management"
    ]
    
    logger.info(f"Testing {len(queries)} different Google searches...")
    
    results = []
    for i, query in enumerate(queries):
        logger.info(f"Search {i+1}/{len(queries)}: {query}")
        result = test_google_search(url, query)
        results.append(result)
    
    return results

def main():
    """Main entry point for the test script
    
    Usage:
        python -m verl_tool.servers.tests.test_google_search_tool test_google_search --url=http://localhost:5000/get_observation
        python -m verl_tool.servers.tests.test_google_search_tool test_google_search --query="your search query"
        python -m verl_tool.servers.tests.test_google_search_tool test_multiple_searches
    """
    fire.Fire({
        "test_google_search": test_google_search,
        "test_multiple_searches": test_multiple_searches,
    })

if __name__ == "__main__":
    main()