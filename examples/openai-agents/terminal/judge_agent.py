import os

from agents import Agent, ModelSettings, RunConfig, SQLiteSession
from agents import Runner as OpenAIRunner
from agents.extensions.models.litellm_model import LitellmModel
from pydantic import BaseModel

from .prompt import JUDGE_PROMPT


class JudgeOutput(BaseModel):
    score: float


class JudgeAgent:
    def __init__(self, base_url: str, api_key: str, model_name: str):
        self.base_url = base_url
        self.api_key = api_key
        # Parse model_name as comma-separated list for round-robin
        self.model_names = [
            name.strip() for name in model_name.split(",") if name.strip()
        ]
        self.current_model_index = 0

    async def get_reward_from_judge(
        self,
        session: SQLiteSession,
        dockerfile_contents: str,
    ) -> float:
        items = await session.get_items()

        # Round-robin model selection
        selected_model = self.model_names[self.current_model_index]
        self.current_model_index = (self.current_model_index + 1) % len(
            self.model_names
        )

        agent = Agent(
            name="JudgeAgent",
            instructions=JUDGE_PROMPT,
            model=LitellmModel(
                model=selected_model,
                api_key=self.api_key,
                base_url=self.base_url,
            ),
            output_type=JudgeOutput,
        )
        try:
            result = await OpenAIRunner.run(
                agent,
                input=items,
                run_config=RunConfig(
                    tracing_disabled=True,
                    model_settings=ModelSettings(
                        temperature=0.0,
                    ),
                ),
            )
            judge_output = result.final_output_as(JudgeOutput)
            return judge_output.score
        except Exception:
            return -1


def judge_from_env() -> JudgeAgent | None:
    base_url = os.environ.get("LITELLM_API_BASE", None)
    api_key = os.environ.get("LITELLM_API_KEY", None)
    model_name = os.environ.get("LITELLM_MODEL_NAME", None)
    if not base_url or not api_key or not model_name:
        return None
    return JudgeAgent(base_url, api_key, model_name)
