#!/usr/bin/env python3
from loguru import logger
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from tau2.agent.gym_agent import AgentGymEnv
from tau2.run import get_options, load_tasks
from tau2.utils.tools import is_functional_tool_call, parse_functional_tool_call

# Initialize Rich console
console = Console()


def disable_logging():
    """Disable all logging during manual mode for cleaner CLI output."""
    # Remove all existing handlers
    logger.remove()
    # Add a handler that does nothing (suppresses all logs)
    logger.add(lambda msg: None, level="CRITICAL")


def enable_logging():
    """Re-enable logging after manual mode."""
    # Remove the silent handler
    logger.remove()
    # Re-add default console handler
    logger.add(lambda msg: print(msg), level="INFO")


def display_domains():
    """Display available domains and let user choose one."""
    options = get_options()
    domains = options.domains

    # Create a table for domains
    table = Table(title="🎯 Available Domains", box=box.ROUNDED)
    table.add_column("Number", style="cyan", justify="center")
    table.add_column("Domain", style="green", justify="left")

    for i, domain in enumerate(domains, 1):
        table.add_row(str(i), domain)

    console.print(table)

    while True:
        try:
            choice = Prompt.ask(
                f"\n[bold blue]Select a domain[/bold blue] (1-{len(domains)})",
                default="1",
            )
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(domains):
                return domains[choice_idx]
            else:
                console.print(
                    f"[red]Please enter a number between 1 and {len(domains)}[/red]"
                )
        except ValueError:
            console.print("[red]Please enter a valid number[/red]")


def display_tasks(domain: str):
    """Display available tasks for the domain and let user choose one."""
    # Try to load tasks for the domain
    try:
        tasks = load_tasks(domain)
    except Exception as e:
        console.print(f"[red]Error loading tasks for domain '{domain}': {e}[/red]")
        # Try alternative task sets
        options = get_options()
        task_sets = [ts for ts in options.task_sets if domain in ts]
        if task_sets:
            console.print(
                f"[yellow]Available task sets for {domain}: {task_sets}[/yellow]"
            )
            task_set = task_sets[0]  # Use first available task set
            console.print(f"[green]Using task set: {task_set}[/green]")
            tasks = load_tasks(task_set)
        else:
            raise ValueError(f"No task sets found for domain '{domain}'")

    # Create a table for tasks
    table = Table(title=f"📋 Available Tasks for {domain}", box=box.ROUNDED)
    table.add_column("Number", style="cyan", justify="center")
    table.add_column("Task ID", style="green", justify="left")
    table.add_column("Description", style="white", justify="left")

    for i, task in enumerate(tasks, 1):
        # Safely handle task description
        try:
            if hasattr(task, "description") and task.description:
                if isinstance(task.description, str):
                    description = task.description
                else:
                    description = str(task.description)
            else:
                description = "No description available"
        except Exception:
            description = "Description unavailable"

        table.add_row(str(i), task.id, description)

    console.print(table)

    while True:
        try:
            choice = Prompt.ask(
                f"\n[bold blue]Select a task[/bold blue] (1-{len(tasks)})", default="1"
            )
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(tasks):
                return tasks[choice_idx]
            else:
                console.print(
                    f"[red]Please enter a number between 1 and {len(tasks)}[/red]"
                )
        except ValueError:
            console.print("[red]Please enter a valid number[/red]")


def get_available_tools(env: AgentGymEnv):
    """Get available tools for the environment."""
    try:
        environment = env.get_environment()
        tools = environment.get_tools()
        return tools
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load tools: {e}[/yellow]")
        return []


def display_tools(tools):
    """Display available tools to the user."""
    if not tools:
        console.print(Panel("No tools available for this domain.", style="red"))
        return

    # Create a table for tools
    table = Table(title="🔧 Available Tools", box=box.ROUNDED)
    table.add_column("Tool Name", style="cyan", justify="left")
    table.add_column("Description", style="white", justify="left")
    table.add_column("Parameters", style="yellow", justify="left")

    for tool in tools:
        # Get the description from the tool
        if hasattr(tool, "short_desc") and tool.short_desc:
            desc = tool.short_desc
        elif hasattr(tool, "long_desc") and tool.long_desc:
            desc = tool.long_desc
        else:
            desc = "No description available"

        # Show parameters if available
        params_text = ""
        if hasattr(tool, "params") and tool.params:
            try:
                params_schema = tool.params.model_json_schema()
                if "properties" in params_schema:
                    param_names = list(params_schema["properties"].keys())
                    if param_names:
                        params_text = ", ".join(param_names)
            except Exception:
                pass

        table.add_row(tool.name, desc, params_text)

    console.print(table)


def format_observation(observation: str, step_count: int):
    """Format the observation for better display."""
    if not observation.strip():
        console.print(Panel("No observation available", style="red"))
        return

    # Create a panel for the observation
    title = f"STEP {step_count} - CURRENT OBSERVATION"

    # Split by lines and format each message
    formatted_lines = []
    lines = observation.strip().split("\n")

    for line in lines:
        if line.strip():
            if line.startswith("user:"):
                formatted_lines.append(
                    f"[bold blue]👤 USER:[/bold blue] {line[5:].strip()}"
                )
            elif line.startswith("assistant:"):
                formatted_lines.append(
                    f"[bold green]🤖 ASSISTANT:[/bold green] {line[10:].strip()}"
                )
            elif line.startswith("system:"):
                formatted_lines.append(
                    f"[bold yellow]⚙️  SYSTEM:[/bold yellow] {line[7:].strip()}"
                )
            else:
                formatted_lines.append(f"[white]📝 {line.strip()}[/white]")

    content = "\n\n".join(formatted_lines)
    panel = Panel(content, title=title, border_style="blue", box=box.ROUNDED)
    console.print(panel)


def get_user_action(env: AgentGymEnv, step_count: int) -> str:
    """Get the next action from the user."""
    console.print(
        f"\n[bold cyan] STEP {step_count} - Enter your action as the agent:[/bold cyan]"
    )
    console.print(
        "[dim](Type 'quit' to exit, 'help' for commands, 'tools' to see available tools)[/dim]"
    )

    while True:
        action = Prompt.ask("[bold green]Action[/bold green]")
        if action.lower() == "quit":
            return None
        elif action.lower() == "help":
            help_panel = Panel(
                """[bold]📋 Available commands:[/bold]
• Type any text to send as your response
• 'quit': Exit the simulation
• 'help': Show this help message
• 'tools': Show available tools

[bold]💡 Tips:[/bold]
• You can use tools by typing their names and parameters
• Example: [cyan]search_flights(origin="NYC", destination="LAX")[/cyan]
• Be conversational and helpful to the user""",
                title="🆘 Help",
                border_style="green",
                box=box.ROUNDED,
            )
            console.print(help_panel)
        elif action.lower() == "tools":
            tools = get_available_tools(env)
            display_tools(tools)
        elif action:
            # Check if the action looks like a functional tool call
            if is_functional_tool_call(action):
                try:
                    # Parse the functional tool call
                    tool_call = parse_functional_tool_call(action)
                    console.print(
                        f"[green]🔧 Parsed tool call:[/green] [cyan]{tool_call.name}[/cyan] "
                        f"with arguments: [yellow]{tool_call.arguments}[/yellow]"
                    )
                    # Return the action as-is for now - the environment will handle the tool call
                    return action
                except (ValueError, SyntaxError) as e:
                    console.print(f"[red]❌ Error parsing tool call: {e}[/red]")
                    console.print(
                        "[yellow]Please check the format. Example: function_name(arg1='value1', arg2=123)[/yellow]"
                    )
                    continue
            return action
        else:
            console.print("[red]Please enter a valid action[/red]")


def main():
    """Main function for the manual mode."""
    # Disable logging for cleaner CLI output
    disable_logging()

    # Welcome message with Rich styling
    welcome_text = Text()
    welcome_text.append("🎮 Welcome to ", style="bold blue")
    welcome_text.append("Tau2 Manual Mode", style="bold green")
    welcome_text.append("!", style="bold blue")

    welcome_panel = Panel(
        """You will be playing the role of the agent in a domain.
This allows you to interact with the simulation as if you were the AI agent.

[bold]Ready to start your adventure?[/bold] 🚀""",
        title=welcome_text,
        border_style="blue",
        box=box.DOUBLE,
    )

    console.print(welcome_panel)

    try:
        # Step 1: Choose domain
        domain = display_domains()
        console.print(f"\n[green]✅ Selected domain:[/green] [bold]{domain}[/bold]")

        # Step 2: Choose task
        task = display_tasks(domain)
        console.print(f"\n[green]✅ Selected task:[/green] [bold]{task.id}[/bold]")
        if task.description:
            try:
                if isinstance(task.description, str):
                    description = task.description
                else:
                    description = str(task.description)
                console.print(f"[dim]📝 Task description: {description}[/dim]")
            except Exception:
                console.print(
                    "[dim]📝 Task description: [red]Unable to display[/red][/dim]"
                )

        # Step 3: Create TauGymEnv instance
        with console.status("[bold green]Initializing environment...", spinner="dots"):
            env = AgentGymEnv(domain=domain, task_id=task.id)

        # Show available tools
        tools = get_available_tools(env)
        display_tools(tools)

        # Step 4: Reset environment and get initial observation
        console.print("\n[bold green]🚀 Starting simulation...[/bold green]")
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
                console.print("[yellow]👋 Exiting simulation...[/yellow]")
                break

            # Step the environment
            try:
                with console.status("[bold green]Processing action...", spinner="dots"):
                    observation, reward, terminated, truncated, info = env.step(action)

                if terminated:
                    if reward != 0.0:
                        console.print(
                            Panel(
                                f"[bold green]🏆 Simulation Completed![/bold green]\n"
                                f"Final reward: [bold yellow]{reward}[/bold yellow]",
                                title="🏁 Simulation Terminated",
                                border_style="green",
                                box=box.ROUNDED,
                            )
                        )
                    else:
                        console.print(
                            Panel(
                                "[bold red]Simulation ended without reward[/bold red]",
                                title="🏁 Simulation Terminated",
                                border_style="red",
                                box=box.ROUNDED,
                            )
                        )
                    break
                elif truncated:
                    console.print(
                        Panel(
                            "[bold yellow]Simulation was truncated (time limit reached)[/bold yellow]",
                            title="⏰ Simulation Truncated",
                            border_style="yellow",
                            box=box.ROUNDED,
                        )
                    )
                    break

            except Exception as e:
                console.print(f"[red]❌ Error during simulation step: {e}[/red]")
                console.print("[yellow]🔄 Continuing with next step...[/yellow]")
                continue

        console.print(
            Panel(
                "[bold green]🎉 Simulation ended. Thank you for playing![/bold green]",
                border_style="green",
                box=box.ROUNDED,
            )
        )

    except KeyboardInterrupt:
        console.print("\n\n[red]⏹️  Simulation interrupted by user.[/red]")
    except Exception as e:
        console.print(
            Panel(
                f"[bold red]❌ Error: {e}[/bold red]\n"
                "Please check your domain and task selection.",
                title="Error",
                border_style="red",
                box=box.ROUNDED,
            )
        )
    finally:
        # Re-enable logging when exiting
        enable_logging()


if __name__ == "__main__":
    main()
