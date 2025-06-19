#!/usr/bin/env python3
"""
Manual mode script for interacting with TauGymEnv as the agent.
Allows users to play the role of the agent in a domain.
"""

from tau2.agent.gym_agent import TauGymEnv
from tau2.run import get_options, load_tasks


def display_domains():
    """Display available domains and let user choose one."""
    options = get_options()
    domains = options.domains

    print("\n=== Available Domains ===")
    for i, domain in enumerate(domains, 1):
        print(f"{i}. {domain}")

    while True:
        try:
            choice = input(f"\nSelect a domain (1-{len(domains)}): ").strip()
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(domains):
                return domains[choice_idx]
            else:
                print(f"Please enter a number between 1 and {len(domains)}")
        except ValueError:
            print("Please enter a valid number")


def display_tasks(domain: str):
    """Display available tasks for the domain and let user choose one."""
    # Try to load tasks for the domain
    try:
        tasks = load_tasks(domain)
    except Exception as e:
        print(f"Error loading tasks for domain '{domain}': {e}")
        # Try alternative task sets
        options = get_options()
        task_sets = [ts for ts in options.task_sets if domain in ts]
        if task_sets:
            print(f"Available task sets for {domain}: {task_sets}")
            task_set = task_sets[0]  # Use first available task set
            print(f"Using task set: {task_set}")
            tasks = load_tasks(task_set)
        else:
            raise ValueError(f"No task sets found for domain '{domain}'")

    print(f"\n=== Available Tasks for {domain} ===")
    for i, task in enumerate(tasks, 1):
        print(f"{i}. {task.id}")
        if task.description:
            print(f"   Description: {task.description}")
        print()

    while True:
        try:
            choice = input(f"Select a task (1-{len(tasks)}): ").strip()
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(tasks):
                return tasks[choice_idx]
            else:
                print(f"Please enter a number between 1 and {len(tasks)}")
        except ValueError:
            print("Please enter a valid number")


def get_available_tools(env: TauGymEnv):
    """Get available tools for the environment."""
    try:
        environment = env.get_environment()
        tools = environment.get_tools()
        return tools
    except Exception as e:
        print(f"Warning: Could not load tools: {e}")
        return []


def display_tools(tools):
    """Display available tools to the user."""
    if not tools:
        print("No tools available for this domain.")
        return

    print("\n=== Available Tools ===")
    for tool in tools:
        # Get the description from the tool
        if hasattr(tool, "short_desc") and tool.short_desc:
            desc = tool.short_desc
        elif hasattr(tool, "long_desc") and tool.long_desc:
            desc = tool.long_desc
        else:
            desc = "No description available"

        print(f"- {tool.name}: {desc}")

        # Show parameters if available
        if hasattr(tool, "params") and tool.params:
            try:
                params_schema = tool.params.model_json_schema()
                if "properties" in params_schema:
                    param_names = list(params_schema["properties"].keys())
                    if param_names:
                        print(f"  Parameters: {', '.join(param_names)}")
            except Exception:
                pass


def format_observation(observation: str, step_count: int):
    """Format the observation for better display."""
    if not observation.strip():
        print("\n=== No observation available ===")
        return

    print("\n" + "=" * 60)
    print(f"STEP {step_count} - CURRENT OBSERVATION:")
    print("=" * 60)

    # Split by lines and format each message
    lines = observation.strip().split("\n")
    for line in lines:
        if line.strip():
            if line.startswith("user:"):
                print(f"👤 USER: {line[5:].strip()}")
            elif line.startswith("assistant:"):
                print(f"🤖 ASSISTANT: {line[10:].strip()}")
            elif line.startswith("system:"):
                print(f"⚙️  SYSTEM: {line[7:].strip()}")
            else:
                print(f"📝 {line.strip()}")

    print("=" * 60)


def get_user_action(env: TauGymEnv, step_count: int) -> str:
    """Get the next action from the user."""
    print(f"\n🎯 STEP {step_count} - Enter your action as the agent:")
    print("(Type 'quit' to exit, 'help' for commands, 'tools' to see available tools)")

    while True:
        action = input("Action: ").strip()
        if action.lower() == "quit":
            return None
        elif action.lower() == "help":
            print("\n📋 Available commands:")
            print("- Type any text to send as your response")
            print("- 'quit': Exit the simulation")
            print("- 'help': Show this help message")
            print("- 'tools': Show available tools")
            print("\n💡 Tips:")
            print("- You can use tools by typing their names and parameters")
            print("- Example: 'search_flights' or 'book_ticket'")
            print("- Be conversational and helpful to the user")
        elif action.lower() == "tools":
            tools = get_available_tools(env)
            display_tools(tools)
        elif action:
            return action
        else:
            print("Please enter a valid action")


def main():
    """Main function for the manual mode."""
    print("🎮 Welcome to Tau2 Manual Mode!")
    print("You will be playing the role of the agent in a domain.")
    print(
        "This allows you to interact with the simulation as if you were the AI agent."
    )

    try:
        # Step 1: Choose domain
        domain = display_domains()
        print(f"\n✅ Selected domain: {domain}")

        # Step 2: Choose task
        task = display_tasks(domain)
        print(f"\n✅ Selected task: {task.id}")
        if task.description:
            print(f"📝 Task description: {task.description}")

        # Step 3: Create TauGymEnv instance
        print(
            f"\n🔧 Initializing environment for domain '{domain}' and task '{task.id}'..."
        )
        env = TauGymEnv(domain=domain, task_id=task.id)

        # Show available tools
        tools = get_available_tools(env)
        display_tools(tools)

        # Step 4: Reset environment and get initial observation
        print("\n🚀 Starting simulation...")
        observation, info = env.reset()

        # Main interaction loop
        step_count = 0
        while True:
            step_count += 1

            # Display current observation
            format_observation(observation, step_count)

            # Get user action
            action = get_user_action(env, step_count)
            if action is None:
                print("👋 Exiting simulation...")
                break

            # Step the environment
            try:
                observation, reward, terminated, truncated, info = env.step(action)

                if terminated:
                    print("\n🏁 === Simulation Terminated ===")
                    if reward != 0.0:
                        print(f"🏆 Final reward: {reward}")
                    break
                elif truncated:
                    print("\n⏰ === Simulation Truncated ===")
                    break

            except Exception as e:
                print(f"❌ Error during simulation step: {e}")
                print("🔄 Continuing with next step...")
                continue

        print("\n🎉 Simulation ended. Thank you for playing!")

    except KeyboardInterrupt:
        print("\n\n⏹️  Simulation interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Please check your domain and task selection.")


if __name__ == "__main__":
    main()
