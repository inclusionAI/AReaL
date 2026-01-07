import copy

import gem
from gem.envs.game_env.sudoku import SudokuEnv


class SudokuWithTemplateEnv(SudokuEnv):
    template = None

    def set_template(self, template: SudokuEnv):
        self.template = template

    def reset(self, seed=None):
        obs, info = super().reset(seed)
        if self.template is None:
            return obs, info
        self.full_grid = copy.deepcopy(self.template.full_grid)
        self.board = copy.deepcopy(self.template.board)
        self.init_num_empty = copy.deepcopy(self.template.init_num_empty)
        return self._get_instructions(), {"suffix": self.get_task_suffix()}


gem.register(
    "game:Sudoku-v0-easy-with-template",
    "multitask_agent.gem_train.sudoku_with_template:SudokuWithTemplateEnv",
    clues=10,
    max_turns=15,
    scale=4,
)
gem.register(
    "game:Sudoku-v0-hard-with-template",
    "multitask_agent.gem_train.sudoku_with_template:SudokuWithTemplateEnv",
    clues=50,
    max_turns=50,
    scale=9,
)
gem.register(
    "game:Sudoku-v0-hard-max-turns-60",
    "gem.envs.game_env.sudoku:SudokuEnv",
    clues=50,
    max_turns=60,
    scale=9,
)
