#!/usr/bin/env python
"""
Structured smoke-test for the Text-Browser tool server.

Run the server first, e.g.:
    python -m verl_tool.servers.serve \
        --tool_type text_browser \
        --url=http://localhost:5000/get_observation

Then execute:
    python -m verl_tool.servers.tests.test_text_browser browser \
        --url=http://localhost:5000/get_observation
"""

import json
import uuid
import logging
import requests
import fire

# ───────────────────────────────────────────────
# Logging
# ───────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────
def _send_test_request(url: str,
                       trajectory_ids: list[str],
                       actions: list[str],
                       extra_fields: list[dict],
                       test_name: str):
    """
    Build the payload, POST to the tool server, and pretty-print the response.
    """
    payload = {
        "trajectory_ids": trajectory_ids,
        "actions": actions,
        "extra_fields": extra_fields,
    }

    logger.info(f"=== {test_name} ===")
    logger.info("POST %s", url)
    logger.info("Payload:\n%s", json.dumps(payload, indent=2))

    try:
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error("Request error: %s", e)
        return {"error": str(e)}

    try:
        data = resp.json()
    except json.JSONDecodeError:
        logger.error("Response is not valid JSON:\n%s", resp.text[:500])
        return {"error": "invalid-json"}

    logger.info("Response:\n%s", json.dumps(data, indent=2))
    return data


# ───────────────────────────────────────────────
# Browser tests
# ───────────────────────────────────────────────
def test_browser(url: str = "http://localhost:5000/get_observation",
                 trajectory_id: str = "test-browser"):
    """
    Fire a couple of minimal actions against the text-browser endpoint.
    """

    # Generate two unique trajectory IDs to simulate two parallel agents
    traj_ids = [
        f"{trajectory_id}-{uuid.uuid4()}",
        f"{trajectory_id}-{uuid.uuid4()}"
    ]

    # Action: simple “type” into the search box with element id 16
    action_str = (
        "<think>Locate Cristiano Ronaldo on Wikipedia</think>\n"
        "```type [16] [Cristiano Ronaldo] [1]```"
    )

    actions = ["", action_str]

    # Same metadata for both trajectories
    extra_fields = [
        {
            "question": "who plays the wildling woman in game of thrones",
            "golden_answers": ["Natalia Gastiain Tena"],
            "gt": "Natalia Gastiain Tena",
            "id": 41214,
            "index": 41214,
            "split": "train",
            "url": (
                "https://tigerai.ca/wiki/"
                "wikipedia_en_all_maxi_2022-05/A/"
                "User:The_other_Kiwix_guy/Landing"
            )
        }
    ] * len(traj_ids)

    # Send request
    _send_test_request(
        url=url,
        trajectory_ids=traj_ids,
        actions=actions,
        extra_fields=extra_fields,
        test_name="Browser-Smoke-Test"
    )

    return True


# ───────────────────────────────────────────────
# CLI entry-point
# ───────────────────────────────────────────────
def main():
    """
    Expose the test via Fire.

    Example:
        python -m verl_tool.servers.tests.test_text_browser browser \
            --url=http://localhost:5000/get_observation
    """
    fire.Fire({
        "browser": test_browser,
    })


if __name__ == "__main__":
    main()
