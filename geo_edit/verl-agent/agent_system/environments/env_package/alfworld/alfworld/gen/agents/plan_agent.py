import alfworld.gen.constants as constants
from alfworld.gen.agents.agent_base import AgentBase
from alfworld.gen.game_states.planned_game_state import PlannedGameState
from alfworld.gen.utils import game_util


class PlanAgent(AgentBase):
    def __init__(self, thread_id=0, game_state=None, controller_agent=None):
        super(PlanAgent, self).__init__(thread_id, game_state)
        assert(isinstance(game_state, PlannedGameState))
        self.controller_agent = controller_agent
        self.planned = False

    def reset(self):
        self.planned = False

    def execute_plan(self):
        step_count = 0
        self.planned = True
        self.controller_agent.planning = True
        if constants.OPEN_LOOP:
            plan = self.game_state.get_current_plan(force_update=True)

            if plan[0]['action'] == 'End':
                raise ValueError('Empty plan is successful, no work to do')

            elif len(plan) == 0 or len(plan) > constants.PLANNER_MAX_STEPS:
                print ("Planning failed. Possibly because the goal was already satisfied")
                raise ValueError("Symbolic Planning Failed")

            for idx, plan_action in enumerate(plan):
                self.save_plan(plan, idx)

                if plan_action['action'] == 'GotoLocation':
                    plan_action = self.game_state.get_teleport_action(plan_action)
                elif plan_action['action'] == 'End':
                    break
                self.controller_agent.step(plan_action, executing_plan=True)
                step_count += 1
                if self.controller_agent.current_frame_count >= constants.MAX_EPISODE_LENGTH:
                    break
        else:
            past_plans = []
            plan = self.game_state.get_current_plan(force_update=True)

            if plan[0]['action'] == 'End':
                raise ValueError('Empty plan is successful, no work to do')
            elif len(plan) == 0 or len(plan) > constants.PLANNER_MAX_STEPS:
                print("Symbolic Planning Failed")
                raise ValueError("Symbolic Planning Failed")

            plan_action = plan[0]
            if constants.USE_DETERMINISTIC_CONTROLLER:
                # Don't fail right away, just rotate a few times.
                rotations = 0
                while rotations < 4 and (plan_action is None or plan_action['action'] == 'End'):
                    action = {'action': 'RotateLeft'}
                    self.controller_agent.step(action, executing_plan=True)
                    rotations += 1
                    plan = self.game_state.get_current_plan(force_update=True)
                    plan_action = plan[0]

            while not(plan_action is None or plan_action['action'] == 'End'):
                self.controller_agent.step(self.game_state.get_plan_action(plan_action), executing_plan=True)

                # save data
                self.save_plan(plan, 0)

                step_count += 1
                past_plans.append(plan)
                if len(past_plans) > 5:
                    past_plans = past_plans[-5:]
                plan = self.game_state.get_current_plan(force_update=True)
                if plan[0]['action'] == 'End':
                    break
                if (step_count >= constants.MAX_PLANNER_STEP_COUNT or
                        self.controller_agent.current_frame_count >= constants.MAX_EPISODE_LENGTH):
                    # Too many steps, plan may be looping.
                    break
                if len(plan) > 1 and any([plan == past_plan for past_plan in past_plans]):
                    plan_action = plan[0]
                    self.controller_agent.step(self.game_state.get_plan_action(plan_action), executing_plan=True)
                    step_count += 1
                    plan = plan[1:]
                plan_action = plan[0]

        self.controller_agent.planning = False

    def save_plan(self, plan, idx=0):
        plan_action = plan[idx]
        constants.data_dict['plan']['high_pddl'].append({"high_idx": len(constants.data_dict['plan']['high_pddl']),
                                                         "planner_action": plan_action,
                                                         "discrete_action": game_util.get_discrete_hl_action(plan, idx)})
        constants.data_dict['template']['high_descs'].append(game_util.get_templated_action_str(plan, idx))
