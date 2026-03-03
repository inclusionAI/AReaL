"""Test script to verify Ray actor creation and connection."""
import ray
from geo_edit.tool_definitions import ToolRouter
from geo_edit.environment.tool_agents import get_manager

def test_main_process():
    """Test creating actors in main process."""
    print("=" * 60)
    print("MAIN PROCESS: Creating Ray actors")
    print("=" * 60)

    # Create ToolRouter (should create actors)
    router = ToolRouter(tool_mode="force", node_resource="tool_agent")
    enabled_agents = router.get_enabled_agents()

    print(f"\nEnabled agents: {enabled_agents}")
    print(f"Number of actors created: {len(router._agents)}")

    if router.is_agent_enabled():
        print("\nActors created successfully:")
        for name, actor in router._agents.items():
            print(f"  - {name}: {actor}")

    return enabled_agents


def test_worker_process(enabled_agents):
    """Simulate worker process connecting to existing actors."""
    print("\n" + "=" * 60)
    print("WORKER PROCESS: Connecting to existing actors")
    print("=" * 60)

    # Reset manager to simulate fresh worker process
    from geo_edit.environment.tool_agents import manager as mgr_module
    mgr_module._MANAGER = None

    # Create new manager and connect
    manager = get_manager()
    print(f"\nAttempting to connect to: {enabled_agents}")

    connected_actors = manager.connect_to_existing_agents(enabled_agents)

    print(f"\nConnected actors: {len(connected_actors)}")
    for name, actor in connected_actors.items():
        print(f"  - {name}: {actor}")

    # Test calling an actor
    if "gllava" in connected_actors:
        print("\n" + "=" * 60)
        print("TEST: Calling gllava actor")
        print("=" * 60)
        try:
            from PIL import Image
            import numpy as np

            # Create a dummy image
            dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

            result = manager.call("gllava", [dummy_image], 0, "What is in this image?")
            print(f"Result: {result[:100]}...")
        except Exception as e:
            print(f"Error calling actor: {e}")


if __name__ == "__main__":
    # Test main process
    enabled_agents = test_main_process()

    # Test worker process
    if enabled_agents:
        test_worker_process(enabled_agents)
    else:
        print("\nNo agents enabled. Check tool_definitions/config.yaml")
