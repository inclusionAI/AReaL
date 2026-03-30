import os
import sys
import json

import alfworld.gen.constants as constants
from alfworld.agents.expert.handcoded_expert_tw import HandCodedTWAgent, PickAndPlaceSimplePolicy, PickTwoObjAndPlacePolicy, LookAtObjInLightPolicy, PickHeatThenPlaceInRecepPolicy, PickCoolThenPlaceInRecepPolicy, PickCleanThenPlaceInRecepPolicy

class PickAndPlaceSimpleThorPolicy(PickAndPlaceSimplePolicy):

    def __init__(self, task_params, env, max_steps):
        super().__init__(task_params, max_steps=max_steps)
        self.env = env

    def get_predicates(self, game_state, obj, parent):
        admissible_commands = game_state['admissible_commands']
        admissible_commands_wo_num_ids = [self.remove_num_ids(ac) for ac in admissible_commands]
        metadata = self.env.last_event.metadata

        self.is_agent_holding_right_object = any(obj in o for o in self.inventory)
        obs_at_curr_recep = self.obs_at_recep[self.curr_recep] if self.curr_recep in self.obs_at_recep else ""
        is_obj_in_obs = "you see" in obs_at_curr_recep and " {} ".format(obj) in obs_at_curr_recep
        at_right_recep = parent in self.curr_recep
        can_put_object = "put {} in/on {}".format(obj, parent) in admissible_commands_wo_num_ids
        can_take_object = any("take {}".format(obj) in ac for ac in admissible_commands_wo_num_ids)
        return at_right_recep, can_put_object, can_take_object, is_obj_in_obs


class PickTwoObjAndPlaceThorPolicy(PickTwoObjAndPlacePolicy):

    def __init__(self, task_params, env, max_steps):
        super().__init__(task_params, max_steps=max_steps)
        self.env = env

    def get_predicates(self, game_state, obj, parent):
        admissible_commands = game_state['admissible_commands']
        admissible_commands_wo_num_ids = [self.remove_num_ids(ac) for ac in admissible_commands]
        metadata = self.env.last_event.metadata

        relevant_receptacles = [r.split("|")[0].lower() for o in metadata['objects'] if o['objectType'].lower() in obj and o['parentReceptacles'] for r in o['parentReceptacles'] if r is not None]
        is_one_object_already_inside_receptacle = any(parent in r for r in relevant_receptacles)
        trying_to_take_the_same_object = "take {} from {}".format(obj, parent) in admissible_commands_wo_num_ids
        self.is_agent_holding_right_object = any(obj in o for o in self.inventory)
        obs_at_curr_recep = self.obs_at_recep[self.curr_recep] if self.curr_recep in self.obs_at_recep else ""
        is_obj_in_obs = "you see" in obs_at_curr_recep and " {} ".format(obj) in obs_at_curr_recep
        at_right_recep = parent in self.curr_recep
        can_put_object = "put {} in/on {}".format(obj, parent) in admissible_commands_wo_num_ids
        can_take_object = any("take {}".format(obj) in ac for ac in admissible_commands_wo_num_ids)
        return at_right_recep, can_put_object, can_take_object, is_obj_in_obs, is_one_object_already_inside_receptacle, trying_to_take_the_same_object


class LookAtObjInLightThorPolicy(LookAtObjInLightPolicy):

    def __init__(self, task_params, env, max_steps):
        super().__init__(task_params, max_steps=max_steps)
        self.env = env

    def get_predicates(self, game_state, obj, toggle):
        admissible_commands = game_state['admissible_commands']
        admissible_commands_wo_num_ids = [self.remove_num_ids(ac) for ac in admissible_commands]
        metadata = self.env.last_event.metadata

        self.is_agent_holding_right_object = any(obj in o for o in self.inventory)
        obs_at_curr_recep = self.obs_at_recep[self.curr_recep] if self.curr_recep in self.obs_at_recep else ""
        is_obj_in_obs = "you see" in obs_at_curr_recep and " {} ".format(obj) in obs_at_curr_recep
        can_toggle_lamp = "use {}".format(toggle) in admissible_commands_wo_num_ids
        can_take_object = any("take {}".format(obj) in ac for ac in admissible_commands_wo_num_ids)
        return can_take_object, can_toggle_lamp, is_obj_in_obs


class PickHeatThenPlaceInRecepThorPolicy(PickHeatThenPlaceInRecepPolicy):

    def __init__(self, task_params, env, max_steps):
        super().__init__(task_params, max_steps=max_steps)
        self.env = env

    def get_predicates(self, game_state, obj, parent):
        admissible_commands = game_state['admissible_commands']
        admissible_commands_wo_num_ids = [self.remove_num_ids(ac) for ac in admissible_commands]
        metadata = self.env.last_event.metadata

        self.is_agent_holding_right_object = any(obj in o for o in self.inventory)
        hot_objects = [o.split("|")[0].lower() for o in list(self.env.heated_objects) if o in [io['objectId'] for io in metadata['inventoryObjects']]]
        is_the_object_agent_holding_hot = len(hot_objects) > 0
        obs_at_curr_recep = self.obs_at_recep[self.curr_recep] if self.curr_recep in self.obs_at_recep else ""
        is_obj_in_obs = "you see" in obs_at_curr_recep and " {} ".format(obj) in obs_at_curr_recep
        at_right_recep = parent in self.curr_recep
        can_put_object = "put {} in/on {}".format(obj, parent) in admissible_commands_wo_num_ids
        can_take_object = any("take {}".format(obj) in ac for ac in admissible_commands_wo_num_ids)
        can_heat_object = "heat {} with {}".format(obj, "microwave") in admissible_commands_wo_num_ids
        return at_right_recep, can_heat_object, can_put_object, can_take_object, is_obj_in_obs, is_the_object_agent_holding_hot


class PickCoolThenPlaceInRecepThorPolicy(PickCoolThenPlaceInRecepPolicy):

    def __init__(self, task_params, env, max_steps):
        super().__init__(task_params, max_steps=max_steps)
        self.env = env

    def get_predicates(self, game_state, obj, parent):
        admissible_commands = game_state['admissible_commands']
        admissible_commands_wo_num_ids = [self.remove_num_ids(ac) for ac in admissible_commands]
        metadata = self.env.last_event.metadata

        self.is_agent_holding_right_object = any(obj in o for o in self.inventory)
        cool_objects = [o.split("|")[0].lower() for o in list(self.env.cooled_objects) if o in [io['objectId'] for io in metadata['inventoryObjects']]]
        is_the_object_agent_holding_cool = len(cool_objects) > 0
        obs_at_curr_recep = self.obs_at_recep[self.curr_recep] if self.curr_recep in self.obs_at_recep else ""
        is_obj_in_obs = "you see" in obs_at_curr_recep and " {} ".format(obj) in obs_at_curr_recep
        at_right_recep = parent in self.curr_recep
        can_put_object = "put {} in/on {}".format(obj, parent) in admissible_commands_wo_num_ids
        can_cool_object = "cool {} with {}".format(obj, "fridge") in admissible_commands_wo_num_ids
        can_take_object = any("take {}".format(obj) in ac for ac in admissible_commands_wo_num_ids)
        return at_right_recep, can_cool_object, can_put_object, can_take_object, is_obj_in_obs, is_the_object_agent_holding_cool


class PickCleanThenPlaceInRecepThorPolicy(PickCleanThenPlaceInRecepPolicy):

    def __init__(self, task_params, env, max_steps):
        super().__init__(task_params, max_steps=max_steps)
        self.env = env

    def get_predicates(self, game_state, obj, parent):
        admissible_commands = game_state['admissible_commands']
        admissible_commands_wo_num_ids = [self.remove_num_ids(ac) for ac in admissible_commands]
        metadata = self.env.last_event.metadata

        self.is_agent_holding_right_object = any(obj in o for o in self.inventory)
        clean_objects = [o.split("|")[0].lower() for o in list(self.env.cleaned_objects) if o in [io['objectId'] for io in metadata['inventoryObjects']]]
        is_the_object_agent_holding_isclean = len(clean_objects) > 0
        obs_at_curr_recep = self.obs_at_recep[self.curr_recep] if self.curr_recep in self.obs_at_recep else ""
        is_obj_in_obs = "you see" in obs_at_curr_recep and " {} ".format(obj) in obs_at_curr_recep
        at_right_recep = parent in self.curr_recep
        can_put_object = "put {} in/on {}".format(obj, parent) in admissible_commands_wo_num_ids
        can_clean_object = "clean {} with {}".format(obj, "sinkbasin") in admissible_commands_wo_num_ids
        can_take_object = any("take {}".format(obj) in ac for ac in admissible_commands_wo_num_ids)
        return at_right_recep, can_clean_object, can_put_object, can_take_object, is_obj_in_obs, is_the_object_agent_holding_isclean


class HandCodedThorAgent(HandCodedTWAgent):
    """ THOR instance of handcoded expert.
        Uses predicates from THOR metadata for making decisions."""

    def __init__(self, env, max_steps=150):
        self.env = env
        super().__init__(max_steps=max_steps)

    def reset(self, game="INVALID"):
        traj_data_file = os.path.join(game)
        with open(traj_data_file, 'r') as f:
            traj_data = json.load(f)

        self.task_params = {**traj_data['pddl_params'],
                            'task_type': traj_data['task_type']}
        self.task_params = dict((k, v.lower() if v in constants.OBJECTS else v) for k, v in self.task_params.items())

        policy_class = self.get_task_policy(self.task_params)
        self.policy = policy_class(self.task_params, self.env, max_steps=self.max_steps)

    def get_task_policy(self, task_param):
        task_type = task_param['task_type']
        task_class_str = task_type.replace("_", " ").title().replace(" ", '') + "ThorPolicy"
        if task_class_str in globals():
            return globals()[task_class_str]
        else:
            raise Exception("Invalid Task Type: %s" % task_type)