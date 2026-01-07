import copy

import gem
from gem.envs.game_env.mastermind import MastermindEnv


class MastermindWithTemplateEnv(MastermindEnv):
    template = None

    def set_template(self, template: MastermindEnv):
        self.template = template

    def reset(self, seed=None):
        obs, info = super().reset(seed)
        if self.template is None:
            return obs, info
        assert (
            self.code_length == self.template.code_length
            and self.num_numbers == self.template.num_numbers
            and self.max_turns == self.template.max_turns
        ), self.__dict__
        self.game_code = self.template.game_code
        return self._get_instructions(), {}


gem.register(
    "game:Mastermind-v0-hard-with-template",
    "multitask_agent.gem_train.mastermind_with_template:MastermindWithTemplateEnv",
    code_length=4,
    num_numbers=8,
    max_turns=30,
    duplicate_numbers=False,
)
