import os
import json

import alfworld.gen
import alfworld.gen.constants as constants
from alfworld.gen.game_states.task_game_state_full_knowledge import TaskGameStateFullKnowledge
from alfworld.gen.agents.deterministic_planner_agent import DeterministicPlannerAgent
from alfworld.gen.graph import graph_obj
from alfworld.agents.controller.oracle import OracleAgent


class OracleAStarAgent(OracleAgent):

    def __init__(self, env, traj_data, traj_root,
                 load_receps=False, debug=False,
                 goal_desc_human_anns_prob=0.0,
                 use_gt_relations=False):

        super().__init__(env, traj_data, traj_root,
                         load_receps=load_receps, debug=debug,
                         use_gt_relations=use_gt_relations,
                         goal_desc_human_anns_prob=goal_desc_human_anns_prob)

    def setup_navigator(self):
        game_state = TaskGameStateFullKnowledge(self.env)

        # reset
        game_state.receptacle_to_point = None
        game_state.task_target = None
        game_state.success = False

        # load nav graph
        scene_num = self.traj_data['scene']['scene_num']
        game_state.gt_graph = graph_obj.Graph(use_gt=True, construct_graph=True, scene_id=scene_num)
        game_state.gt_graph.clear()

        game_state.agent_height = self.env.last_event.metadata['agent']['position']['y']
        game_state.camera_height = game_state.agent_height + constants.CAMERA_HEIGHT_OFFSET

        points_source = os.path.join(alfworld.gen.__path__[0], 'layouts/FloorPlan%s-openable.json' % scene_num)
        with open(points_source, 'r') as f:
            openable_object_to_point = json.load(f)
        game_state.openable_object_to_point = openable_object_to_point

        game_state.update_receptacle_nearest_points() # TODO: save to desk
        game_state.planner.process_pool.terminate()

        self.navigator = DeterministicPlannerAgent(thread_id=0, game_state=game_state)


    def navigate(self, teleport_loc):
        self.navigator.pose = self.env.last_event.pose_discrete
        self.navigator.step(teleport_loc)
        return self.env.last_event