import copy

import gem
from gem.envs.game_env.minesweeper import MinesweeperEnv


class MinesweeperWithTemplateEnv(MinesweeperEnv):
    template = None

    def set_template(self, template: MinesweeperEnv):
        self.template = template

    def reset(self, seed=None):
        obs, info = super().reset(seed)
        if self.template is None:
            return obs, info
        template = self.template
        assert (
            self.rows == template.rows
            and self.cols == template.cols
            and self.num_mines == template.num_mines
            and self._is_random == template._is_random
        ), (self.__dict__, template.__dict__)
        self.grid = copy.deepcopy(template.grid)
        self.revealed = copy.deepcopy(template.revealed)
        self.flags = copy.deepcopy(template.flags)
        self.first_reveal = template.first_reveal
        return self._get_instructions(), {"suffix": self.get_task_suffix()}


gem.register(
    "game:Minesweeper-v0-easy-with-template",
    "multitask_agent.gem_train.minesweeper_with_template:MinesweeperWithTemplateEnv",
    rows=5,
    cols=5,
    num_mines=5,
    max_turns=25,
)
gem.register(
    "game:Minesweeper-v0-legacy",
    "gem.envs.game_env.minesweeper:MinesweeperEnv",
    rows=8,
    cols=8,
    num_mines=10,
    max_turns=64,
)
