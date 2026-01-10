from openai import AsyncOpenAI

from areal.api.workflow_api import AgentWorkflow


class SimpleAgent(AgentWorkflow):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def run(self, base_url: str, data: dict):
        async with AsyncOpenAI(max_retries=0) as client:
            _ = await client.chat.completions.create(
                messages=data["messages"], model="default", **self.kwargs
            )

        return 1.0
