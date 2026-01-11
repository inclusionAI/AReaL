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
        # from areal.experimental.openai.client import ArealOpenAI
        # client = ArealOpenAI(data['engine'])
        # comp: ChatCompletion = await client.chat.completions.create(
        #         messages=data["messages"], model="default", **self.kwargs
        #     )
        async with AsyncOpenAI(base_url=base_url, max_retries=0) as client:
            _: ChatCompletion = await client.chat.completions.create(
                messages=data["messages"], model="default", **self.kwargs
            )
        # from openai import OpenAI
        # with OpenAI(base_url=base_url, max_retries=0) as client:
        #     comp: ChatCompletion = client.chat.completions.create(
        #         messages=data["messages"], model="default", **self.kwargs
        #     )

        return 1.0
        # ans = parse(comp.choices[0].message.content, parsing_timeout=None)
        # gold = parse(data["answer"], parsing_timeout=None)
        # reward = verify(ans, gold, timeout_seconds=None)
        # return float(reward)
