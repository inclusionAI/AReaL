import glob
import cv2
import alfworld.gen.constants as constants
from alfworld.gen.agents.agent_base import AgentBase
from alfworld.gen.agents.plan_agent import PlanAgent
from alfworld.gen.game_states.planned_game_state import PlannedGameState


class SemanticMapPlannerAgent(AgentBase):
    def __init__(self, thread_id=0, game_state=None):
        assert(isinstance(game_state, PlannedGameState))
        super(SemanticMapPlannerAgent, self).__init__(thread_id, game_state)

        self.plan_agent = PlanAgent(thread_id, game_state, self)
        self.planning = False

    def reset(self, seed=None, info=None, scene=None, objs=None):
        self.planning = False
        info = self.game_state.get_setup_info(info, scene=scene)[0]
        super(SemanticMapPlannerAgent, self).reset({'seed': seed, 'info': info}, scene=scene, objs=objs)
        if self.plan_agent is not None:
            self.plan_agent.reset()
        return info

    def setup_problem(self, game_state_problem_args, scene=None, objs=None):
        super(SemanticMapPlannerAgent, self).setup_problem(game_state_problem_args, scene=scene, objs=objs)
        self.pose = self.game_state.pose

    def get_reward(self):
        raise NotImplementedError

    def step(self, action, executing_plan=False):
        if action['action'] == 'End':
            self.current_frame_count += 1
            self.total_frame_count += 1
            self.terminal = True

            if constants.RECORD_VIDEO_IMAGES:
                im_ind = len(glob.glob(constants.save_path + '/*.png'))
                for _ in range(10):
                    cv2.imwrite(constants.save_path + '/%09d.png' % im_ind,
                                self.game_state.s_t[:, :, ::-1])
                    im_ind += 1
        else:
            if 'Teleport' in action['action']:
                start_pose = self.pose
                end_angle = action['horizon']
                end_pose = (int(action['x'] / constants.AGENT_STEP_SIZE),
                            int(action['z'] / constants.AGENT_STEP_SIZE),
                            int(action['rotation'] / 90),
                            int(end_angle))

                self.game_state.gt_graph.navigate_to_goal(self.game_state, start_pose, end_pose)
                self.pose = self.game_state.pose
            elif action['action'] == 'Plan':
                self.plan_agent.execute_plan()
                if not constants.EVAL:
                    self.current_frame_count += 1
                self.total_frame_count += 1
            elif action['action'] == 'Scan':
                self.game_state.step(action)
                if not constants.EVAL:
                    self.current_frame_count += 1
                self.total_frame_count += 1
            elif action['action'] == 'Explore':
                if not constants.EVAL:
                    self.current_frame_count += 1
                self.total_frame_count += 1
            else:
                super(SemanticMapPlannerAgent, self).step(action)


