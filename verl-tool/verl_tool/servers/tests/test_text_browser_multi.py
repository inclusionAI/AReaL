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


def _send_request(url, trajectory_id, action):
    """
    Internal helper function to send a single request to the server and log the response.
    Returns True if successful, otherwise False.
    """
    payload = {
        "trajectory_ids": [trajectory_id],
        "actions": [action],
        "extra_fields": [
            {
                "question": "when is the next deadpool movie being released",
                "gt": "gt",
                "url": "http://localhost:22015/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
            }
        ]
    }

    logger.info(f"Sending request to {url}")
    logger.info(f"Payload: {json.dumps(payload, indent=2)}")

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        logger.info(f"Response: {json.dumps(result, indent=2)}")

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
            logger.info(f"  Observation {i + 1}: {obs}")

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


def test_connection(url="http://localhost:5000/get_observation"):
    """
    Test the connection to the tool server by sending multiple actions sequentially.
    """
    def exec_actions(trajectory_id, actions):
        """
        Execute a list of actions for a given trajectory ID.
        """
        for action in actions:
            success = _send_request(url, trajectory_id, action)
            if not success:
                return False
        return True

    actions = [
        "<think>balabalabalabala</think>\n```click [99]```",
        "<think>balabala</think>```type [1407] [death row inmates in the US] [1]```",
        "<think>balabala</think>```scroll [down]```",
        # "<think>balabala</think>```stop [Here is a stop message]```",
    ]

    trajectory_ids, results = [], []
    for i in range(32):
        trajectory_ids.append(f"trajectory-{i}")
    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=16) as executor:
        future_to_id = {
            executor.submit(exec_actions, trajectory_id, actions): trajectory_id
            for trajectory_id in trajectory_ids
        }
        for future in as_completed(future_to_id):
            trajectory_id = future_to_id[future]
            try:
                result = future.result()
                logger.info(f"Result for {trajectory_id}: {result}")
                results.append(result)
            except Exception as e:
                logger.error(f"Error for {trajectory_id}: {e}")
                results.append(False)

    return all(results)

def main():
    """
    Entry point for the test script.

    Start the Server:
    rm *.db* # Delete existed environment
    python -m verl_tool.servers.serve --tool_type text_browser --url=http://localhost:5000/get_observation

    Run with:
    python -m verl_tool.servers.tests.test_text_browser_multi --url=http://localhost:5000/get_observation
    """
    fire.Fire(test_connection)


if __name__ == "__main__":
    main()