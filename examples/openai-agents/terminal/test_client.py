"""Simple MCP client to test terminal server functionality."""

import asyncio

from mcp import ClientSession
from mcp.client.sse import sse_client


async def test_mcp_server():
    """Test the MCP server by calling tools."""
    server_url = "http://localhost:8000"

    print("Connecting to MCP server...")
    async with sse_client(f"{server_url}/sse") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("✓ Connected to MCP server\n")

            # List available tools
            print("=" * 60)
            print("Available Tools:")
            print("=" * 60)
            tools_result = await session.list_tools()
            for tool in tools_result.tools:
                print(f"  • {tool.name}: {tool.description}")
            print()

            # Test 1: Start a task
            print("=" * 60)
            print("Test 1: Starting a task container")
            print("=" * 60)
            import requests

            response = requests.post(
                f"{server_url}/tasks/start",
                json={"uuid": "test-123", "task_name": "hello-world"},
                timeout=60,
            )
            if response.status_code == 200:
                data = response.json()
                container_name = data["container_name"]
                print(f"✓ Task started: {container_name}\n")

                # Test 2: Send keystrokes
                print("=" * 60)
                print("Test 2: Sending keystrokes (echo 'Hello MCP')")
                print("=" * 60)
                result = await session.call_tool(
                    name="keystrokes",
                    arguments={
                        "container_name": container_name,
                        "keystrokes": "echo 'Hello MCP'",
                        "append_enter": True,
                        "wait_time_sec": 1.0,
                    },
                )
                output = result.content[0].text if result.content else ""
                print(f"Output:\n{output}\n")

                print("=" * 60)
                print("Test 3: Capturing terminal pane")
                print("=" * 60)
                result = await session.call_tool(
                    name="capture_pane",
                    arguments={
                        "container_name": container_name,
                        "wait_before_capture_sec": 0.5,
                    },
                )
                output = result.content[0].text if result.content else ""
                print(f"Captured output:\n{output}\n")

                print("=" * 60)
                print("Test 4: Listing active tasks")
                print("=" * 60)
                response = requests.get(f"{server_url}/tasks")
                if response.status_code == 200:
                    tasks = response.json()
                    print(f"✓ Active tasks: {len(tasks)}")
                    for task in tasks:
                        print(f"  • {task['container_name']} (UUID: {task['uuid']})")
                print()

                # validate task
                print("=" * 60)
                print("Test 4: Validating task container")
                print("=" * 60)
                response = requests.post(
                    f"{server_url}/tasks/validate",
                    json={"uuid": "test-123", "task_name": "hello-world"},
                    timeout=180,
                )
                if response.status_code == 200:
                    print(
                        f"✓ Task validated successfully, score: {response.json().get('score')}\n"
                    )
                else:
                    print(f"✗ Failed to validate task: {response.text}\n")

                print("=" * 60)
                print("Test 5: Stopping task container")
                print("=" * 60)
                response = requests.post(
                    f"{server_url}/tasks/stop",
                    json={"uuid": "test-123", "task_name": "hello-world"},
                    timeout=30,
                )
                if response.status_code == 200:
                    print("✓ Task stopped successfully\n")
                else:
                    print(f"✗ Failed to stop task: {response.text}\n")
            else:
                print(f"✗ Failed to start task: {response.text}\n")

            print("=" * 60)
            print("All tests completed!")
            print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(test_mcp_server())
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
