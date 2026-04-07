#!/usr/bin/env python
import json
import requests
import fire
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_connection(
    url: str = "http://localhost:5000/get_observation",
    trajectory_id: str = "test-trajectory-001",
    action: str = "test-action",
    query: str = "test-query",
):
    """
    Test the connection to the tool server.
    
    Args:
        url: The URL of the server endpoint (default: http://localhost:5000/get_observation)
        trajectory_id: The test trajectory ID
        action: The test action
        query: The test query
        extra_field: Optional extra data to include in the request
    
    Returns:
        True if test passed, False otherwise
    """
    
    # Prepare the request payload
    payload = {
        "trajectory_ids": [trajectory_id],
        "actions": [action],
    }
    
    logger.info(f"Sending request to {url}")
    logger.info(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        # Send the POST request
        response = requests.post(url, json=payload)
        
        # Check if request was successful
        response.raise_for_status()
        
        # Get the response data
        result = response.json()
        logger.info(f"Response: {json.dumps(result, indent=2)}")
        
        # Validate the response
        if "observations" not in result:
            logger.error("Error: Response missing 'observations' field")
            return False
        
        observations = result["observations"]
        if not observations or not isinstance(observations, list):
            logger.error(f"Error: Expected observations to be a non-empty list, got {type(observations)}")
            return False
        
        logger.info("Test passed! ✅")
        logger.info(f"Received {len(observations)} observations:")
        for i, obs in enumerate(observations):
            logger.info(f"  Observation {i+1}: {obs}")
        
        return True
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Connection error: {e}")
        return False
    except json.JSONDecodeError:
        logger.error("Error: Invalid JSON response")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False

def main():
    """
    Entry point for the test script.
    Run with: 
        python -m verl_tool.servers.tests.test_base --url=http://localhost:5000/get_observation
    """
    fire.Fire(test_connection)

if __name__ == "__main__":
    main()