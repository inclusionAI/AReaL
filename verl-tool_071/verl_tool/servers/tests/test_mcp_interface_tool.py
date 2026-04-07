#!/usr/bin/env python
import os
import json
import requests
import fire
import logging

# Import our result parser
from verl_tool.servers.tools.utils.mcp_result_parser import format_mcp_response

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _make_payload(action: str, trajectory_id: str, extra: dict | None = None):
    return {
        "trajectory_ids": [trajectory_id],
        "actions": [action],
        "extra_fields": [extra or {}],
    }


def test_mcp_invalid_tool(url: str = "http://localhost:5000/get_observation"):
    """Attempt to call a non-existent MCP tool and expect invalid handling."""
    action_obj = {"name": "__non_existing_tool__", "arguments": {}}
    action = f"<mcp_call>{json.dumps(action_obj)}</mcp_call>"
    payload = _make_payload(action, "mcp-invalid")

    try:
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        result = resp.json()
        logger.info("Response received successfully (invalid tool)")
        print(json.dumps(result, indent=2))
        return result
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return {"error": str(e)}


def test_mcp_call(
    url: str = "http://localhost:5000/get_observation",
    tool_name: str = "",
    args_json: str = "{}",
    format_type: str = "clean",
):
    """Call a real MCP tool by name with JSON-encoded arguments.

    Usage:
      python -m verl_tool.servers.tests.test_mcp_interface_tool test_mcp_call --tool_name your_tool --args_json='{"key":"value"}'
    """
    assert tool_name, "tool_name is required"
    try:
        # Handle case where fire already parsed JSON string to dict
        if isinstance(args_json, dict):
            args = args_json
        else:
            args = json.loads(args_json)
    except Exception as e:
        raise ValueError(f"Invalid args_json: {e}")

    action_obj = {"name": tool_name, "arguments": args}
    action = f"<mcp_call>{json.dumps(action_obj)}</mcp_call>"
    payload = _make_payload(action, "mcp-call")

    try:
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        result = resp.json()
        logger.info("Response received successfully (call)")

        # Print observation
        if "observations" in result and result["observations"]:
            obs = result["observations"][0]
            raw_text = obs["obs"] if isinstance(obs, dict) else obs
            
            # Use our parser to clean up the response
            cleaned_text = format_mcp_response(raw_text, tool_name, format_type)
            
            print(f"\n--- MCP Tool Call Result ({format_type}) ---")
            print(cleaned_text)
        else:
            logger.warning("No observations in result")

        return result
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return {"error": str(e)}


def main():
    fire.Fire({
        "test_mcp_invalid_tool": test_mcp_invalid_tool,
        "test_mcp_call": test_mcp_call,
    })


if __name__ == "__main__":
    main()


