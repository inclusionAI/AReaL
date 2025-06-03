# Licensed under the Apache License, Version 2.0 (the "License").

import dataclasses

import realhf.base.logging as logging
from realhf.api.core.config import AgentAbstraction
from realhf.api.quickstart.entrypoint import register_quickstart_exp
from realhf.experiments.async_exp.async_ppo_math_exp import AsyncPPOMATHConfig

logger = logging.getLogger("Async PPO Math exp", "colored")


@dataclasses.dataclass
class AsyncPPOMATHMultiTurnConfig(AsyncPPOMATHConfig):
    turn_level_discount: float = dataclasses.field(
        default=1.0, metadata={"help": "discount factor for multi turn reasoning"}
    )
    num_turns: int = dataclasses.field(
        default=5, metadata={"help": "number of turns for llm reasoning"}
    )

    @property
    def agent(self) -> AgentAbstraction:
        assert self.group_size == self.num_turns
        return AgentAbstraction(
            "math-multi-turn",
            args=dict(
                gconfig=self.generation_config,
                tokenizer_path=self.actor.path,
                reward_scaling=self.ppo.reward_output_scaling,
                reward_bias=self.ppo.reward_output_bias,
                turn_level_discount=self.turn_level_discount,
                num_turns=self.num_turns,
            ),
        )


register_quickstart_exp("async-ppo-math-multi-turn", AsyncPPOMATHMultiTurnConfig)
