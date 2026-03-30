import os
import sys

from alfworld.agents.expert.handcoded_expert import HandCodedAgent, PickAndPlaceSimplePolicy, PickTwoObjAndPlacePolicy, LookAtObjInLightPolicy, PickHeatThenPlaceInRecepPolicy, PickCoolThenPlaceInRecepPolicy, PickCleanThenPlaceInRecepPolicy

class PickAndPlaceSimpleTWPolicy(PickAndPlaceSimplePolicy):

    def __init__(self, task_params, max_steps):
        super().__init__(task_params, max_steps=max_steps)

    def get_predicates(self, game_state, obj, parent):
        facts, facts_wo_num_ids, admissible_commands, admissible_commands_wo_num_ids = self.get_state_info(game_state)
        self.is_agent_holding_right_object = "holds agent {}".format(obj) in facts_wo_num_ids
        obs_at_curr_recep = self.obs_at_recep[self.curr_recep] if self.curr_recep in self.obs_at_recep else ""
        is_obj_in_obs = "you see" in obs_at_curr_recep and " {} ".format(obj) in obs_at_curr_recep
        at_right_recep = parent in self.curr_recep
        can_put_object = "move {} to {}".format(obj, parent) in admissible_commands_wo_num_ids
        can_take_object = any("take {}".format(obj) in ac for ac in admissible_commands_wo_num_ids)
        return at_right_recep, can_put_object, can_take_object, is_obj_in_obs


class PickTwoObjAndPlaceTWPolicy(PickTwoObjAndPlacePolicy):

    def __init__(self, task_params, max_steps):
        super().__init__(task_params, max_steps=max_steps)

    def get_predicates(self, game_state, obj, parent):
        facts, facts_wo_num_ids, admissible_commands, admissible_commands_wo_num_ids = self.get_state_info(game_state)
        in_recep_predicate = "inreceptacle {} {}".format(obj, parent)
        is_one_object_already_inside_receptacle = in_recep_predicate in facts_wo_num_ids
        in_receptacle_obj_ids = [" ".join(f.split()[1:3]) for f in facts if in_recep_predicate in self.remove_num_ids(f)]
        in_receptacle_obj_id = in_receptacle_obj_ids[0] if len(in_receptacle_obj_ids) > 0 else ""
        trying_to_take_the_same_object = any("take {}".format(in_receptacle_obj_id) in ac for ac in admissible_commands)
        self.is_agent_holding_right_object = "holds agent {}".format(obj) in facts_wo_num_ids
        obs_at_curr_recep = self.obs_at_recep[self.curr_recep] if self.curr_recep in self.obs_at_recep else ""
        is_obj_in_obs = "you see" in obs_at_curr_recep and " {} ".format(obj) in obs_at_curr_recep
        at_right_recep = parent in self.curr_recep
        can_put_object = "move {} to {}".format(obj, parent) in admissible_commands_wo_num_ids
        can_take_object = any("take {}".format(obj) in ac for ac in admissible_commands_wo_num_ids)
        return at_right_recep, can_put_object, can_take_object, is_obj_in_obs, is_one_object_already_inside_receptacle, trying_to_take_the_same_object


class LookAtObjInLightTWPolicy(LookAtObjInLightPolicy):

    def __init__(self, task_params, max_steps):
        super().__init__(task_params, max_steps=max_steps)

    def get_predicates(self, game_state, obj, toggle):
        facts, facts_wo_num_ids, admissible_commands, admissible_commands_wo_num_ids = self.get_state_info(game_state)
        self.is_agent_holding_right_object = "holds agent {}".format(obj) in facts_wo_num_ids
        obs_at_curr_recep = self.obs_at_recep[self.curr_recep] if self.curr_recep in self.obs_at_recep else ""
        is_obj_in_obs = "you see" in obs_at_curr_recep and " {} ".format(obj) in obs_at_curr_recep
        can_toggle_lamp = "use {}".format(toggle) in admissible_commands_wo_num_ids
        can_take_object = any("take {}".format(obj) in ac for ac in admissible_commands_wo_num_ids)
        return can_take_object, can_toggle_lamp, is_obj_in_obs


class PickHeatThenPlaceInRecepTWPolicy(PickHeatThenPlaceInRecepPolicy):

    def __init__(self, task_params, max_steps):
        super().__init__(task_params, max_steps=max_steps)

    def get_predicates(self, game_state, obj, parent):
        facts, facts_wo_num_ids, admissible_commands, admissible_commands_wo_num_ids = self.get_state_info(game_state)
        inventory = self.inventory
        self.is_agent_holding_right_object = "holds agent {}".format(obj) in facts_wo_num_ids
        is_the_object_agent_holding_hot = "ishot {}".format(inventory[0]) in facts if len(inventory) > 0 else False
        obs_at_curr_recep = self.obs_at_recep[self.curr_recep] if self.curr_recep in self.obs_at_recep else ""
        is_obj_in_obs = "you see" in obs_at_curr_recep and " {} ".format(obj) in obs_at_curr_recep
        at_right_recep = parent in self.curr_recep
        can_put_object = "move {} to {}".format(obj, parent) in admissible_commands_wo_num_ids
        can_take_object = any("take {}".format(obj) in ac for ac in admissible_commands_wo_num_ids)
        can_heat_object = "heat {} with {}".format(obj, "microwave") in admissible_commands_wo_num_ids
        return at_right_recep, can_heat_object, can_put_object, can_take_object, is_obj_in_obs, is_the_object_agent_holding_hot


class PickCoolThenPlaceInRecepTWPolicy(PickCoolThenPlaceInRecepPolicy):

    def __init__(self, task_params, max_steps):
        super().__init__(task_params, max_steps=max_steps)

    def get_predicates(self, game_state, obj, parent):
        facts, facts_wo_num_ids, admissible_commands, admissible_commands_wo_num_ids = self.get_state_info(game_state)
        inventory = self.inventory
        self.is_agent_holding_right_object = "holds agent {}".format(obj) in facts_wo_num_ids
        is_the_object_agent_holding_cool = "iscool {}".format(inventory[0]) in facts if len(inventory) > 0 else False
        obs_at_curr_recep = self.obs_at_recep[self.curr_recep] if self.curr_recep in self.obs_at_recep else ""
        is_obj_in_obs = "you see" in obs_at_curr_recep and " {} ".format(obj) in obs_at_curr_recep
        at_right_recep = parent in self.curr_recep
        can_put_object = "move {} to {}".format(obj, parent) in admissible_commands_wo_num_ids
        can_cool_object = "cool {} with {}".format(obj, "fridge") in admissible_commands_wo_num_ids
        can_take_object = any("take {}".format(obj) in ac for ac in admissible_commands_wo_num_ids)
        return at_right_recep, can_cool_object, can_put_object, can_take_object, is_obj_in_obs, is_the_object_agent_holding_cool


class PickCleanThenPlaceInRecepTWPolicy(PickCleanThenPlaceInRecepPolicy):

    def __init__(self, task_params, max_steps):
        super().__init__(task_params, max_steps=max_steps)

    def get_predicates(self, game_state, obj, parent):
        facts, facts_wo_num_ids, admissible_commands, admissible_commands_wo_num_ids = self.get_state_info(game_state)
        inventory = self.inventory
        self.is_agent_holding_right_object = "holds agent {}".format(obj) in facts_wo_num_ids
        is_the_object_agent_holding_isclean = "isclean {}".format(inventory[0]) in facts if len(inventory) > 0 else False
        obs_at_curr_recep = self.obs_at_recep[self.curr_recep] if self.curr_recep in self.obs_at_recep else ""
        is_obj_in_obs = "you see" in obs_at_curr_recep and " {} ".format(obj) in obs_at_curr_recep
        at_right_recep = parent in self.curr_recep
        can_put_object = "move {} to {}".format(obj, parent) in admissible_commands_wo_num_ids
        can_clean_object = "clean {} with {}".format(obj, "sinkbasin") in admissible_commands_wo_num_ids
        can_take_object = any("take {}".format(obj) in ac for ac in admissible_commands_wo_num_ids)
        return at_right_recep, can_clean_object, can_put_object, can_take_object, is_obj_in_obs, is_the_object_agent_holding_isclean


class HandCodedTWAgent(HandCodedAgent):
    """ Textworld instance of handcoded expert.
        Uses predicates from Textworld Engine for making decisions."""

    def __init__(self, max_steps=150):
        super().__init__(max_steps=max_steps)

    def get_task_policy(self, task_param):
        task_type = task_param['task_type']
        task_class_str = task_type.replace("_", " ").title().replace(" ", '') + "TWPolicy"
        if task_class_str in globals():
            return globals()[task_class_str]
        else:
            raise Exception("Invalid Task Type: %s" % task_type)
