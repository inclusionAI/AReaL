import asyncio
import subprocess
from typing import Any

from agents import (
    Agent,
    ModelSettings,
    RunConfig,
    SQLiteSession,
    function_tool,
    set_default_openai_api,
)
from agents import Runner as OpenAIRunner
from pydantic import BaseModel

set_default_openai_api("chat_completions")


# Define response model for MCP tool calls
class CodeExecutionResult(BaseModel):
    success: bool
    output: str
    error: str | None = ""


# Actual tool function implementations (without @function_tool decorator)
def run_python_code_impl(code: str, env_name: str = "system") -> CodeExecutionResult:
    """
    Execute code in the specified Python environment
    """
    try:
        result = subprocess.run(
            ["python", "-c", code], capture_output=True, text=True, timeout=30
        )

        print(
            f"run_python_code_impl run code: {code}, stdout: {result.stdout}, stderr: {result.stderr}"
        )

        return CodeExecutionResult(
            success=result.returncode == 0,
            output=result.stdout,
            error=result.stderr if result.returncode != 0 else "",
        )
    except Exception as e:
        return CodeExecutionResult(
            success=False, output="", error=f"Execution failed: {str(e)}"
        )


def list_python_environments_impl() -> list:
    """
    List all available Python environments
    """
    try:
        result = subprocess.run(
            ["python", "-c", "import sys; print(sys.executable)"],
            capture_output=True,
            text=True,
        )

        print(
            f"list_python_environments_impl returned code: {result.returncode}, stdout: {result.stdout}, stderr: {result.stderr}"
        )

        if result.returncode == 0:
            return [
                {"name": "system", "path": result.stdout.strip(), "version": "unknown"}
            ]
        else:
            return [{"name": "system", "path": "default", "version": "unknown"}]
    except Exception as e:
        return [
            {"name": "system", "path": "default", "version": "unknown", "error": str(e)}
        ]


def install_python_package_impl(package_name: str, env_name: str = "system") -> dict:
    """
    Install Python package in specified environment
    """
    try:
        result = subprocess.run(
            ["pip", "install", package_name], capture_output=True, text=True
        )

        print(
            f"install_python_package_impl returned code: {result.returncode}, stdout: {result.stdout}, stderr: {result.stderr}"
        )

        return {
            "success": result.returncode == 0,
            "message": result.stdout if result.returncode == 0 else result.stderr,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# Create tools using decorator (for Agent usage)
@function_tool
def run_python_code(code: str, env_name: str = "system") -> CodeExecutionResult:
    return run_python_code_impl(code, env_name)


@function_tool
def list_python_environments() -> list:
    return list_python_environments_impl()


@function_tool
def install_python_package(package_name: str, env_name: str = "system") -> dict:
    return install_python_package_impl(package_name, env_name)


# Create Python programming assistant Agent
agent = Agent(
    name="RLVR Math with Code Interpreter",
    tools=[
        run_python_code,
        # list_python_environments,
        # install_python_package,
    ],
)


######### run agent
async def run_agent(data):
    content = data["messages"][-1]["content"]

    run_config = RunConfig(
        tracing_disabled=True,
        model_settings=ModelSettings(
            temperature=1.0,
            top_p=1.0,
            # max_tokens=16384,
        ),
    )
    session = SQLiteSession("math")

    result = await OpenAIRunner.run(
        agent, input=content, session=session, run_config=run_config
    )

    return result


########## reward function
def gsm8k_reward_fn(result, answer):
    from areal.reward.math_parser import process_results

    return float(process_results(result, answer)[0])


async def run_agent_return_reward(data: Any) -> float:
    result = await run_agent(data)
    reward = gsm8k_reward_fn(result.final_output, data["answer"])
    return reward


async def main():
    """Main function to run the Python programming assistant"""

    print("Python Programming Assistant started!")
    print("Type 'quit' to exit the program\n")

    while True:
        try:
            user_input = input("User: ").strip()

            if user_input.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break

            if not user_input:
                continue

            result = await run_agent(
                {"messages": [{"role": "user", "content": user_input}]}
            )
            print(f"\nAssistant: {result.final_output}\n")
            print("-" * 50)

        except KeyboardInterrupt:
            print("\n\nProgram interrupted by user")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
