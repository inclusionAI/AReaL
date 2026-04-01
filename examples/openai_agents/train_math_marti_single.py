from dataclasses import dataclass, field
from string import Template
from typing import Any

from agents import Agent as OpenAIAgent
from agents import ModelSettings, OpenAIProvider, RunConfig
from agents import Runner as OpenAIRunner
from transformers import PreTrainedTokenizerFast

from areal import PPOTrainer, workflow_context
from areal.api import AsyncRewardWrapper, RolloutWorkflow
from areal.api.cli_args import GenerationHyperparameters, GRPOConfig, load_expr_config
from areal.dataset import get_custom_dataset
from areal.experimental.openai import ArealOpenAI
from areal.utils import logging, stats_tracker
from areal.utils.dynamic_import import import_from_string
from areal.utils.episode_data import clone_episode_data
from areal.utils.hf_utils import load_hf_tokenizer

logger = logging.getLogger("OpenAIAgentWorkflow")

GENERATOR_TEMPLATE = (
    "$question\n\nPlease reason step by step, and put your final answer within \\boxed{}."
)

def render_template(template: str, **kwargs: str) -> str:
    return Template(template).safe_substitute(**kwargs)

def apply_template_with_tokenizer(
    tokenizer: PreTrainedTokenizerFast,
    prompt: str,
    tools: list[dict[str, Any]] | None = None,
    enable_thinking: bool = False,
) -> str:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    if tools is not None:
        kwargs["tools"] = tools
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **kwargs)

class OpenAIAgentWrapper:
    def __init__(
        self,
        agent_name: str,
        reward_fn_path: str,
        is_last: bool = False,
        temperature: float = 1.0,
        max_tokens: int = 512,
    ):
        self.agent = OpenAIAgent(name=agent_name)
        self.is_last = is_last

        self.async_reward_fn: AsyncRewardWrapper | None = None
        # Since there is only one agent, it acts as both the first and the last stage (for reward calculation).
        if self.is_last and reward_fn_path:
            self.async_reward_fn = AsyncRewardWrapper(import_from_string(reward_fn_path))
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def run_agent(self, data: dict[str, Any], client: ArealOpenAI) -> tuple[str, float]:
        run_config = RunConfig(
            model_provider=OpenAIProvider(openai_client=client),
            tracing_disabled=True,
            model_settings=ModelSettings(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            ),
        )
        try:
            result = await OpenAIRunner.run(
                    self.agent, input=data["messages"][-1]["content"], run_config=run_config
                )
        except Exception as e:
            logger.error(f"!! Agent {self.agent.name} inference failed: {e}")
            return "Error: Inference failed", 0.0
        
        reward = 0.0
        if self.is_last and self.async_reward_fn:
            reward = await self.async_reward_fn(
                completions=result.final_output,
                answer=data["answer"],
                prompt=data.get("prompt"),
                prompt_ids=data.get("prompt_ids"),
                completion_ids=data.get("completion_ids"),
                **{
                    k: v
                    for k, v in data.items()
                    if k
                    not in ["messages", "answer", "prompt", "prompt_ids", "completion_ids"]
                },
            )
        client.set_last_reward(reward)
        return result.final_output, reward

class OpenAIAgentWorkflow(RolloutWorkflow):
    def __init__(
        self,
        reward_fn_path: str,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
    ):
        if isinstance(tokenizer, str):
            tokenizer = load_hf_tokenizer(tokenizer)
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(tokenizer)
        self.tokenizer = tokenizer

        # Keep only the generator agent and set is_last=True to trigger reward calculation.
        self.generator = OpenAIAgentWrapper(
            temperature=gconfig.temperature,
            max_tokens=gconfig.max_tokens,
            agent_name="generator",
            reward_fn_path=reward_fn_path,
            is_last=True, 
        )

    def _set_norm_group_if_missing(self, client: ArealOpenAI, norm_group: str) -> None:
        for interaction in client._cache.values():
            if interaction.norm_group is None:
                interaction.norm_group = norm_group

    async def _run_stage(
        self,
        *,
        wrapper: OpenAIAgentWrapper,
        client: ArealOpenAI,
        data: dict[str, Any],
        prompt: str,
        norm_group: str,
    ) -> tuple[str, float]:
        input_prompt = apply_template_with_tokenizer(client.tokenizer, prompt)
        data["messages"].append({"role": "user", "content": input_prompt})
        output, reward = await wrapper.run_agent(data=data, client=client)
        output = output.strip()
        data["messages"].append({"role": "assistant", "content": output})
        self._set_norm_group_if_missing(client, norm_group)
        return output, reward

    async def arun_episode(self, engine, data: dict[str, Any]):
        data = clone_episode_data(data)
        client = ArealOpenAI(
            engine=engine, tokenizer=self.tokenizer, tool_call_parser="qwen25"
        )

        if not data.get("messages"):
            raise ValueError("Expected episode data to contain non-empty 'messages'.")

        question = data["messages"][-1]["content"]

        generator_prompt = render_template(GENERATOR_TEMPLATE, question=question)
        _, final_reward = await self._run_stage(
            wrapper=self.generator,
            client=client,
            data=data,
            prompt=generator_prompt,
            norm_group="group_1",
        )

        for interaction in client._cache.values():
            interaction.reward = final_reward

        stats_tracker.get(workflow_context.stat_scope()).scalar(reward=final_reward)
        
        interactions_with_reward = client.export_interactions(style="individual")
        return interactions_with_reward

@dataclass
class AgentRLConfig(GRPOConfig):
    reward_fn_path: str = field(
        default="areal.reward.gsm8k.gsm8k_reward_fn",
        metadata={
            "help": "The path to the reward function."
        },
    )

def main(args):
    config, _ = load_expr_config(args, AgentRLConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)

    train_dataset = get_custom_dataset(
        split="train",
        dataset_config=config.train_dataset,
        tokenizer=tokenizer,
    )

    valid_dataset = get_custom_dataset(
        split="test",
        dataset_config=config.valid_dataset,
        tokenizer=tokenizer,
    )

    workflow_kwargs = dict(
        reward_fn_path=config.reward_fn_path,
        gconfig=config.gconfig,
        tokenizer=config.tokenizer_path,
    )
    eval_workflow_kwargs = workflow_kwargs.copy()
    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="examples.openai_agents.train_math_marti_single.OpenAIAgentWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="examples.openai_agents.train_math_marti_single.OpenAIAgentWorkflow",
            eval_workflow_kwargs=eval_workflow_kwargs,
        )

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])