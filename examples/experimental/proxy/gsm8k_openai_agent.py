from agents import (
    Agent,
    ModelSettings,
    OpenAIProvider,
    RunConfig,
    SQLiteSession,
)
from agents import Runner as OpenAIRunner

from areal.api.workflow_api import AgentWorkflow


class GSM8kAgent(AgentWorkflow):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    async def run(self, base_url: str, data: dict):
        content = data["messages"][-1]["content"]
        run_config = RunConfig(
            model_provider=OpenAIProvider(base_url=base_url),
            model="default",  # no need to pass
            tracing_disabled=True,
            model_settings=ModelSettings(**self.kwargs),
        )
        agent = Agent(name="Assistant")
        session = SQLiteSession("math")
        result = await OpenAIRunner.run(
            agent, input=content, session=session, run_config=run_config
        )

        # compute reward with areal's existing implementation
        # Use the following wrapper to suppress the annoying warning of math-verify
        from areal.api.reward_api import AsyncRewardWrapper
        from areal.reward.gsm8k import gsm8k_reward_fn

        reward = await AsyncRewardWrapper(gsm8k_reward_fn)(
            None, result.final_output, None, None, answer=data["answer"]
        )
        return reward
