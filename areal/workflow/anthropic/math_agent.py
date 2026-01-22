import anthropic

from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import AgentWorkflow
from areal.reward import get_math_verify_worker


def gsm8k_reward_fn(result, answer):
    try:
        worker = get_math_verify_worker()
        return worker.verify(str(result), str(answer))
    except Exception:
        return 0.0


class MathAgent(AgentWorkflow):
    """Simple single-turn math agent using Anthropic Messages API."""

    def __init__(self, **kwargs):
        # Store kwargs for client.messages.create call
        self.kwargs = kwargs
        # Cache AsyncRewardWrapper to avoid creating new ProcessPoolExecutor per call
        self.reward_fn = AsyncRewardWrapper(gsm8k_reward_fn)

    async def run(self, data: dict, **extra_kwargs) -> float:
        """Run the agent on a single problem.

        Args:
            data: Input data containing "messages" and "answer"
            **extra_kwargs: Contains base_url and http_client from proxy

        Returns:
            Reward value for this trajectory
        """
        http_client = extra_kwargs.get("http_client", None)
        base_url = extra_kwargs.get("base_url", None)

        client = anthropic.AsyncAnthropic(
            api_key="placeholder",
            base_url=base_url,
            http_client=http_client,
            max_retries=0,
        )

        # Convert OpenAI-style messages to Anthropic format
        messages = data["messages"]
        system_prompt = None
        anthropic_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                anthropic_messages.append(
                    {
                        "role": msg["role"],
                        "content": msg["content"],
                    }
                )

        # Call the Anthropic Messages API (via proxy)
        try:
            response = await client.messages.create(
                model="default",
                messages=anthropic_messages,
                system=system_prompt if system_prompt else anthropic.NOT_GIVEN,
                **self.kwargs,
            )
        except Exception:
            return 0.0

        # Extract response text
        completion_text = ""
        for block in response.content:
            if hasattr(block, "text"):
                completion_text += block.text

        # Calculate reward
        return await self.reward_fn(
            result=completion_text,
            answer=data["answer"],
        )
