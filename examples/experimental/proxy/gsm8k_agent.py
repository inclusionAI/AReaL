from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from areal.api.workflow_api import AgentWorkflow


class GSM8kAgent(AgentWorkflow):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def run(self, base_url: str, data: dict):
        async with AsyncOpenAI(base_url=base_url) as client:
            comp: ChatCompletion = await client.chat.completions.create(
                messages=data["messages"], model="default", **self.kwargs
            )

        # compute reward with areal's existing implementation
        # Use the following wrapper to suppress the annoying warning of math-verify
        from areal.api.reward_api import AsyncRewardWrapper
        from areal.reward.gsm8k import gsm8k_reward_fn

        reward = await AsyncRewardWrapper(gsm8k_reward_fn)(
            None, comp.choices[0].message.content, None, None, answer=data["answer"]
        )
        return reward
