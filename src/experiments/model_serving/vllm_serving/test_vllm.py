#!/usr/bin/env python3
"""
vLLM test script using litellm
Usage: python test_vllm.py [--debug] [MODEL_NAME]
"""

import argparse
import os
import sys
from typing import Optional

import litellm
from litellm import completion
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text

# Configuration
HOST = "127.0.0.1"
PORT = 8000
VLLM_PREFIX = "hosted_vllm"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-instruct"
API_BASE = f"http://{HOST}:{PORT}"
API_BASE_V1 = f"{API_BASE}/v1"

# Initialize Rich console
console = Console()


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Test vLLM server with litellm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_vllm.py                           # Use default model
  python test_vllm.py Qwen/Qwen2.5-14B         # Test specific model
  python test_vllm.py --debug                   # Enable debug mode with default model
  python test_vllm.py --debug Qwen/Qwen2.5-14B  # Enable debug mode with specific model
        """
    )
    
    parser.add_argument(
        "model_name",
        nargs="?",
        default=DEFAULT_MODEL,
        help=f"Name of the model to test (default: {DEFAULT_MODEL})"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for litellm"
    )
    
    return parser.parse_args()


def test_health_endpoint() -> bool:
    """Test if the vLLM server is healthy"""
    try:
        import requests

        response = requests.get(f"{API_BASE}/health", timeout=5)
        return response.status_code == 200
    except Exception as e:
        console.print(f"[yellow]Health check failed: {e}[/yellow]")
        return False


def test_completion(
    model_name: str, prompt: str = "Hello, how are you today?"
) -> Optional[dict]:
    """Test completion endpoint using litellm"""
    try:

        # Create the model name for litellm
        litellm_model_name = f"{VLLM_PREFIX}/{model_name}"

        console.print(
            f"[blue]Testing completion with model:[/blue] {litellm_model_name}"
        )

        # Make the completion request
        response = completion(
            model=litellm_model_name,
            # api_base=API_BASE_V1,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.7,
            timeout=30,
        )

        return response

    except Exception as e:
        console.print(f"[red]Completion request failed: {e}[/red]")
        return None


def test_chat_completion(
    model_name: str, prompt: str = "What is the capital of France?"
) -> Optional[dict]:
    """Test chat completion endpoint using litellm"""
    try:
        # Configure litellm to use vLLM server
        litellm_model_name = f"{VLLM_PREFIX}/{model_name}"

        console.print(
            f"[blue]Testing chat completion with model:[/blue] {litellm_model_name}"
        )

        # Make the chat completion request
        response = completion(
            model=litellm_model_name,
            api_base=API_BASE_V1,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=100,
            temperature=0.7,
            timeout=30,
        )

        return response

    except Exception as e:
        console.print(f"[red]Chat completion request failed: {e}[/red]")
        return None


def test_streaming_completion(
    model_name: str, prompt: str = "Write a short story about a robot."
) -> bool:
    """Test streaming completion endpoint using litellm"""
    try:
        # Configure litellm to use vLLM server
        litellm_model_name = f"{VLLM_PREFIX}/{model_name}"

        console.print(
            f"[blue]Testing streaming completion with model:[/blue] {litellm_model_name}"
        )

        # Make the streaming completion request
        response = completion(
            model=litellm_model_name,
            api_base=API_BASE_V1,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.7,
            stream=True,
            timeout=30,
        )

        console.print("[green]Streaming response:[/green]")
        for chunk in response:
            if chunk.choices[0].delta.content:
                console.print(chunk.choices[0].delta.content, end="", style="white")
        console.print()  # New line after streaming

        return True

    except Exception as e:
        console.print(f"[red]Streaming completion request failed: {e}[/red]")
        return False


def main():
    """Main function to run all tests"""
    # Parse command line arguments
    args = parse_arguments()
    
    if args.debug:
        litellm._turn_on_debug()
        console.print("[yellow]Debug mode enabled[/yellow]")

    # Get model name from parsed arguments
    model_name = args.model_name

    # Print header
    title = Text("vLLM Server Test Suite", style="bold blue")
    subtitle = Text(f"Testing model: {model_name} at {API_BASE}", style="dim")
    console.print(Panel(title + "\n" + subtitle, border_style="blue"))
    console.print()

    # Test 1: Health check
    console.print("[bold cyan]=== Test 1: Health Check ===[/bold cyan]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        _ = progress.add_task("Checking server health...", total=None)
        is_healthy = test_health_endpoint()

    if is_healthy:
        console.print("✅ [green]Server is healthy[/green]")
    else:
        console.print(
            "⚠️  [yellow]Health check failed, but continuing with tests...[/yellow]"
        )
    console.print()

    # Test 2: Basic completion
    console.print("[bold cyan]=== Test 2: Basic Completion ===[/bold cyan]")
    completion_response = test_completion(model_name)
    if completion_response:
        console.print("✅ [green]Completion test successful[/green]")
        response_text = completion_response.choices[0].message.content
        console.print(Panel(response_text, title="Response", border_style="green"))
    else:
        console.print("❌ [red]Completion test failed[/red]")
    console.print()

    # Test 3: Chat completion
    console.print("[bold cyan]=== Test 3: Chat Completion ===[/bold cyan]")
    chat_response = test_chat_completion(model_name)
    if chat_response:
        console.print("✅ [green]Chat completion test successful[/green]")
        response_text = chat_response.choices[0].message.content
        console.print(Panel(response_text, title="Response", border_style="green"))
    else:
        console.print("❌ [red]Chat completion test failed[/red]")
    console.print()

    # Test 4: Streaming completion
    console.print("[bold cyan]=== Test 4: Streaming Completion ===[/bold cyan]")
    if test_streaming_completion(model_name):
        console.print("✅ [green]Streaming completion test successful[/green]")
    else:
        console.print("❌ [red]Streaming completion test failed[/red]")
    console.print()

    # Summary
    console.print(
        Panel(
            "🎉 [bold green]All tests completed![/bold green]",
            border_style="green",
            title="Test Summary",
        )
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[red]Test interrupted by user[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        sys.exit(1)
