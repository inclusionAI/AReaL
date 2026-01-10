from math_verify import parse, verify
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from areal.api.workflow_api import AgentWorkflow


class GSM8kAgent(AgentWorkflow):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def run(self, base_url: str, data: dict):
        # custom_timeout = httpx.Timeout(30.0, read=600.0)
        # async with AsyncOpenAI(base_url=base_url, max_retries=0,
        # timeout=custom_timeout) as client:
        async with AsyncOpenAI(max_retries=0) as client:
            comp: ChatCompletion = await client.chat.completions.create(
                messages=data["messages"], model="default", **self.kwargs
            )

        ans = parse(comp.choices[0].message.content)
        gold = parse(data["answer"])
        reward = verify(ans, gold)
        return float(reward)
