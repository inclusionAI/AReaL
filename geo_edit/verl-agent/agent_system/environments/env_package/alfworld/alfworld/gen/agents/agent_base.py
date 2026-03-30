import copy
import time
import numpy as np


class AgentBase(object):
    def __init__(self, thread_id=0, game_state=None):
        assert(game_state is not None)
        self.game_state = game_state
        self.thread_id = thread_id
        self.timers = np.zeros((2, 2))
        self.total_frame_count = 0
        self.current_frame_count = 0
        self.gt_graph = None
        self.bounds = None
        self.pose = None
        self.terminal = False
        self.num_invalid_actions = 0
        self.total_num_invalid_actions = 0

    def setup_problem(self, game_state_problem_args, scene=None, objs=None):
        self.game_state.setup_problem(**game_state_problem_args, scene=scene, objs=objs)

    def reset(self, game_state_reset_args, scene=None, objs=None):
        self.game_state.reset(**game_state_reset_args, scene=scene, objs=objs)

        self.timers = np.zeros((2, 2))
        self.current_frame_count = 0
        self.gt_graph = None
        self.bounds = None
        self.pose = None
        self.terminal = False
        self.num_invalid_actions = 0

        self.total_frame_count += 1
        self.gt_graph = self.game_state.gt_graph
        self.bounds = self.game_state.bounds
        self.pose = self.game_state.pose

    def step(self, action):
        self.total_frame_count += 1
        self.current_frame_count += 1
        t_start = time.time()
        self.game_state.step(action)
        if not self.game_state.event.metadata['lastActionSuccess']:
            self.num_invalid_actions += 1
            self.total_num_invalid_actions += 1
        self.timers[0, 0] += time.time() - t_start
        self.timers[0, 1] += 1
        if self.timers[0, 1] % 100 == 0:
            print('game state step time %.3f' % (self.timers[0, 0] / self.timers[0, 1]))
            self.timers[0, :] = 0
        self.pose = self.game_state.pose

    def get_action(self, action_ind):
        action = copy.deepcopy(self.game_state.action_space[action_ind])
        if action['action'] == 'End':
            # Remove other arguments
            action = {'action': 'End'}
        return action
