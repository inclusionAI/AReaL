import asyncio
import subprocess

from agents import (
    Agent,
    ModelSettings,
    RunConfig,
    RunResult,
    SQLiteSession,
    function_tool,
)
from agents import Runner as OpenAIRunner
from pydantic import BaseModel

from areal.api.cli_args import GenerationHyperparameters
from areal.reward import get_math_verify_worker
from areal.utils.proxy_utils import run_and_submit_rewards


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


######### run agent
async def run_agent(messages: list[dict], run_config: RunConfig) -> RunResult:
    # Create Python programming assistant Agent
    agent = Agent(
        name="RLVR Math with Code Interpreter",
        tools=[
            run_python_code,
            # list_python_environments,
            # install_python_package,
        ],
    )

    content = messages[-1]["content"]
    session = SQLiteSession("math")
    return await OpenAIRunner.run(
        agent, input=content, session=session, run_config=run_config
    )


########## reward function
def gsm8k_reward_fn(result, answer):
    try:
        worker = get_math_verify_worker()
        return worker.verify(str(result), str(answer))
    except Exception:
        return 0.0


async def run_agent_return_reward(data: dict) -> float:
    messages = data["messages"]
    answer = data["answer"]
    gconfig = data.get("gconfig", {})
    model_settings = GenerationHyperparameters(**gconfig).to_openai_args_dict(
        api_format="openai-agents"
    )
    run_config = RunConfig(
        model="default",  # no need to pass
        tracing_disabled=True,
        model_settings=ModelSettings(**model_settings),
    )
    result = await run_agent(messages=messages, run_config=run_config)
    reward = gsm8k_reward_fn(result.final_output, answer)
    return reward


async def run_and_submit(data: dict):
    await run_and_submit_rewards(func=run_agent_return_reward, data=data)


# Compatible to be run in subprocess mode
if __name__ == "__main__":
    import json
    import sys

    data = json.loads(sys.stdin.readline())
    asyncio.run(run_and_submit(data))
