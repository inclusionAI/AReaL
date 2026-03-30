import gym
from gym_sokoban.envs.sokoban_env import SokobanEnv as GymSokobanEnv
import numpy as np
from agent_system.environments.env_package.sokoban.sokoban.utils import NoLoggerWarnings, set_seed
from agent_system.environments.env_package.sokoban.sokoban.room_utils import generate_room
import copy

from agent_system.environments.env_package.sokoban.sokoban.base import BaseDiscreteActionEnv

class SokobanEnv(BaseDiscreteActionEnv, GymSokobanEnv):

    GRID_LOOKUP = {
        0: " # \t",  # wall
        1: " _ \t",  # floor
        2: " O \t",  # target
        3: " √ \t",  # box on target
        4: " X \t",  # box
        5: " P \t",  # player
        6: " S \t",  # player on target
        # Use tab separator to separate columns and \n\n to separate rows.
    }

    ACTION_LOOKUP = {
        0: "None",
        1: "Up",
        2: "Down",
        3: "Left",
        4: "Right",
    }

    INVALID_ACTION = 0
    PENALTY_FOR_INVALID = -1

    def __init__(self, mode, **kwargs):
        BaseDiscreteActionEnv.__init__(self)
        self.cur_seq = []
        self.action_sequence = []
        self.search_depth = kwargs.pop('search_depth', 300)
        self.mode = mode
        assert mode in ['tiny_rgb_array', 'list', 'state', 'rgb_array']
        GymSokobanEnv.__init__(
            self,
            dim_room=kwargs.pop('dim_room', (6, 6)), 
            max_steps=kwargs.pop('max_steps', 100),
            num_boxes=kwargs.pop('num_boxes', 3),
            **kwargs
        )
        self.ACTION_SPACE = gym.spaces.discrete.Discrete(4, start=1)
        self.reward = 0
        self._valid_actions = []


    def reset(self, seed=None):
        self.seed = seed
        self._reset_tracking_variables()
        with NoLoggerWarnings():
            try:
                with set_seed(seed):
                    self.room_fixed, self.room_state, self.box_mapping, action_sequence = generate_room(
                        dim=self.dim_room,
                        num_steps=self.num_gen_steps,
                        num_boxes=self.num_boxes,
                        search_depth=self.search_depth
                    )
            except (RuntimeError, RuntimeWarning) as e:
                print("[SOKOBAN] Runtime Error/Warning: {}".format(e))
                print("[SOKOBAN] Retry . . .")
                next_seed = abs(hash(str(seed))) % (2 ** 32) if seed is not None else None
                return self.reset(next_seed)
            
            # self.action_sequence = self._reverse_action_sequence(action_sequence)
            self.player_position = np.argwhere(self.room_state == 5)[0]
            self.num_env_steps = self.reward_last = self.boxes_on_target = 0

            info = {
                "won": False,
            }
            return self.render(self.mode), info
        

    def finished(self):
        return self.num_env_steps >= self.max_steps or self.success()

    def success(self):
        return self.boxes_on_target == self.num_boxes

    def step(self, action: int):
        """
        - Step the environment with the given action.
        - Check if the action is effective (whether player moves in the env).
        """
        # assert not self.success()

        if action == self.INVALID_ACTION:
            return self.render(self.mode), -0.1, False, {"action_is_effective": False, "won": False}
        prev_player_position = self.player_position
        _, reward, done, _ = GymSokobanEnv.step(self, action, observation_mode=self.mode)
        
        obs = self.render(self.mode)
        info = {
            "action_is_effective": not np.array_equal(prev_player_position, self.player_position),
            "won": self.success(),
        }
        return obs, reward, done, info
     

    def render(self, mode):
        assert mode in ['tiny_rgb_array', 'list', 'state', 'rgb_array']

        if mode == 'rgb_array':
            img = self.get_image(mode, scale=1) # numpy array
            return img


        if mode == 'state':
            return np.where((self.room_state == 5) & (self.room_fixed == 2), 6, self.room_state)
        
        room_state = self.render(mode='state').tolist()

        if mode == 'list':
            lookup = lambda cell: self.GRID_LOOKUP.get(cell, "?").strip("\t").strip()
            return [" ".join(lookup(cell) for cell in row) for row in room_state]
        
        if mode == 'tiny_rgb_array':
            lookup = lambda cell: self.GRID_LOOKUP.get(cell, "?")
            return "\n".join("".join(lookup(cell) for cell in row) for row in room_state)
    
        
    def copy(self):
        new_self = SokobanEnv(
            dim_room=self.dim_room,
            max_steps=self.max_steps,
            num_boxes=self.num_boxes,
            search_depth=self.search_depth
        )
        new_self.room_fixed = self.room_fixed.copy()
        new_self.room_state = self.room_state.copy()
        new_self.box_mapping = self.box_mapping.copy()
        new_self.action_sequence = self.action_sequence.copy()
        new_self.player_position = self.player_position.copy()
        new_self.reward = self.reward
        new_self._valid_actions = copy.deepcopy(self._valid_actions)
        return new_self
    



    # def _reverse_action_sequence(self, action_sequence):
    #     def reverse_action(action):
    #         return (action % 2 + 1) % 2 + 2 * (action // 2) # 0 <-> 1, 2 <-> 3
    #     return [reverse_action(action) + 1 for action in action_sequence[::-1]] # action + 1 to match the action space
            
    def set_state(self, rendered_state):
        # from the rendered state, set the room state and player position
        self.room_state = np.where(rendered_state == 6, 5, rendered_state)
        self.player_position = np.argwhere(self.room_state == 5)[0]