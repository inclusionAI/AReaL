import os
import re
import sys
import json
import random
from textworld import Agent

import alfworld.gen.constants as constants

class HandCodedAgentTimeout(NameError):
    pass

class HandCodedAgentFailed(NameError):
    pass

class BasePolicy(object):

    # Object and affordance priors
    OBJECTS = constants.OBJECTS_SINGULAR
    RECEPTACLES = [r.lower() for r in constants.RECEPTACLES]
    OPENABLE_OBJECTS = [o.lower() for o in constants.OPENABLE_CLASS_LIST]
    VAL_RECEPTACLE_OBJECTS = dict((r.lower(), [o.lower() for o in objs]) for r, objs in constants.VAL_RECEPTACLE_OBJECTS.items())
    OBJECT_TO_VAL_RECEPTACLE = dict((o, list()) for o in OBJECTS)
    for o in constants.OBJECTS:
        for r, os in constants.VAL_RECEPTACLE_OBJECTS.items():
            if o in os:
                OBJECT_TO_VAL_RECEPTACLE[o.lower()].append(r.lower())

    def __init__(self, task_params, max_steps=150):
        self.max_steps = max_steps
        self.task_params = task_params
        self.subgoals = [
            {'action': "look", 'param': ""}
        ]
        # receptacle
        self.receptacles = {}
        self.visible_objects = {}
        self.obj_cls_to_receptacle_map = {}

        # state memory
        self.subgoal_idx = 0
        self.curr_recep = ""
        self.checked_inside_curr_recep = False
        self.receptacles_to_check = []
        self.inventory = []
        self.obs_at_recep = {}
        self.got_inventory_from_recep = ""
        self.action_backlog = []
        self.object_blacklist = []
        self.receptacle_whitelist = []
        self.steps = 0
        self.is_agent_holding_right_object = False

    def get_objects_and_classes(self, obs):
        obj_str = re.split("you see", obs, flags=re.IGNORECASE)[-1].replace(" and a ", "").replace(" a ", "").split("Your task is to", 1)[0].strip(".,\n\r")
        objects_with_ids = obj_str.split(",")
        return dict((o, self.object_id_to_cls(o)) for o in objects_with_ids)

    def object_id_to_cls(self, object_hash, special_char=' '):
        return object_hash.split(special_char)[0]

    def remove_num_ids(self, str):
        return ' '.join(''.join(i for i in str if not i.isdigit()).split())

    def get_list_of_objects_of_class(self, obj_cls, obj_dict):
        return [name for name, cls in obj_dict.items() if cls == obj_cls]

    def get_next_subgoal(self):
        subgoal = self.subgoals[self.subgoal_idx]
        sub_action, sub_param = subgoal['action'], subgoal['param']
        objs_of_interest = self.get_list_of_objects_of_class(sub_param, self.visible_objects)

        # special case: check if bathbasin should be used instead of sinkbasin
        if sub_param == 'sinkbasin' and not any('sinkbasin' in r for r in self.receptacles.values()):
            sub_param = "bathtubbasin"

        return sub_action, sub_param, objs_of_interest

    def get_list_of_receptacles_to_search_for_object_cls(self, obj_cls):
        obj_found_in_receps = self.OBJECT_TO_VAL_RECEPTACLE[obj_cls]
        return [r_name for r_name, r_cls in self.receptacles.items() if r_cls in obj_found_in_receps]

    def get_list_of_receptacles_of_type(self, recep_cls):
        return [r_name for r_name, r_cls in self.receptacles.items() if r_cls == recep_cls]

    def is_receptacle_openable(self, recep):
        return recep and recep in self.receptacles and self.receptacles[recep] in self.OPENABLE_OBJECTS

    def is_already_at_receptacle(self, sub_param):
        return (sub_param in self.obj_cls_to_receptacle_map and self.curr_recep == self.obj_cls_to_receptacle_map[sub_param]) or \
               (self.curr_recep in self.receptacles and self.receptacles[self.curr_recep] == sub_param)

    def is_blacklisted_object_in_visible_objects(self, visible_objects):
        return len(list(set().union(visible_objects, self.object_blacklist))) > 0

    def is_obj_cls_in_inventory(self, obj_cls):
        return len(self.inventory) > 0 and obj_cls in self.object_id_to_cls(self.inventory[0])

    def blacklist_obj(self, obj):
        obj_cls = self.object_id_to_cls(obj)
        if obj_cls in self.obj_cls_to_receptacle_map:
            del self.obj_cls_to_receptacle_map[obj_cls]
        if obj in self.visible_objects:
            del self.visible_objects[obj]
        if self.curr_recep in self.obs_at_recep:
            self.obs_at_recep[self.curr_recep] = self.obs_at_recep[self.curr_recep].replace(obj, "")

    def check_subgoal_completion(self, game_state):
        subgoal_idx = 0
        return subgoal_idx

    def get_facts(self, game_state):
        facts = [f"{fact.name} " + " ".join(name.strip() for name in fact.names) for fact in game_state["facts"]]
        return facts

    def get_state_info(self, game_state):
        facts = self.get_facts(game_state)
        facts_wo_num_ids = [self.remove_num_ids(f) for f in facts]

        admissible_commands = game_state['admissible_commands']
        admissible_commands_wo_num_ids = [self.remove_num_ids(ac) for ac in admissible_commands]
        return facts, facts_wo_num_ids, admissible_commands, admissible_commands_wo_num_ids

    def observe(self, obs):
        if "Welcome" in obs:  # intro text with receptacles
            self.receptacles = self.get_objects_and_classes(obs)
        else:
            # get the objects which are visible in the current frame
            if "you see nothing" in obs:
                self.visible_objects = dict()
            elif "you see" in obs:
                if "You open the" in obs:
                    curr_recep = " ".join(obs.split("You open the", 1)[-1].split()[:2]).strip(",.")
                elif "is open." in obs:
                    curr_recep = " ".join(obs.split("is open.", 1)[0].split()[-2:]).strip(",.")
                else:
                    curr_recep = " ".join(obs.split("On the", 1)[-1].split()[:2]).strip(",.")

                self.visible_objects = self.get_objects_and_classes(obs)
                self.obs_at_recep[curr_recep] = str(obs)

                # ignore blacklisted object
                for obj in self.object_blacklist:
                    if obj in self.visible_objects:
                        del self.visible_objects[obj]

                # keep track of where all the objects are
                for o_name, o_cls in self.visible_objects.items():
                    self.obj_cls_to_receptacle_map[o_cls] = curr_recep

    def update_state_tracking(self, game_state, last_action):
        obs = game_state['feedback']
        was_last_action_successful = "Nothing happens" not in obs

        if "go to" in last_action:
            went_to_receptacle = last_action.replace("go to", "").strip()
            self.curr_recep = str(went_to_receptacle)
            if went_to_receptacle in self.receptacles_to_check:
                self.receptacles_to_check.remove(went_to_receptacle)
            self.checked_inside_curr_recep = (not self.is_receptacle_openable(self.curr_recep)) or "open" in obs

        if was_last_action_successful:
            if "open" in last_action:
                open_curr_recep_str = "open {}".format(self.curr_recep)
                if open_curr_recep_str == last_action:
                    self.checked_inside_curr_recep = True
                    self.action_backlog.append("close {}".format(self.curr_recep))
            elif "close" in last_action:
                close_curr_recep_str = "close {}".format(self.curr_recep)
                if close_curr_recep_str == last_action:
                    if close_curr_recep_str in self.action_backlog:
                        self.action_backlog.remove(close_curr_recep_str)
            elif "take" in last_action:
                inv_obj = " ".join(last_action.split()[1:3])
                self.got_inventory_from_recep = " ".join(last_action.split()[-2:])
                self.inventory.append(inv_obj)
                self.blacklist_obj(inv_obj)
            elif "put" in last_action:
                inv_obj = self.inventory.pop()
                self.got_inventory_from_recep = ""
                self.blacklist_obj(inv_obj)
                if self.task_params['object_target'] in last_action:
                    self.object_blacklist.append(inv_obj)

    def act(self, game_state, last_action):
        obs = game_state['feedback']
        self.steps += 1

        # timeout
        if self.steps > self.max_steps:
            raise HandCodedAgentTimeout()

        # finished all subgoals but still didn't achieve the goal
        elif self.subgoal_idx >= len(self.subgoals):
            while len(self.action_backlog) > 0:
                return self.action_backlog.pop()
            raise HandCodedAgentFailed()

        # update state tracking
        self.update_state_tracking(game_state, last_action)

        # update observations
        self.observe(obs)

        # get subgoal
        self.subgoal_idx = self.check_subgoal_completion(game_state)
        sub_action, sub_param, objs_of_interest = self.get_next_subgoal()

        # FIND
        if sub_action == 'find':
            # done criteria
            if len(objs_of_interest) > 0 or (self.is_already_at_receptacle(sub_param) and self.checked_inside_curr_recep) and (len(self.inventory) == 0 or (len(self.inventory) > 0 and self.is_agent_holding_right_object)):
                self.receptacles_to_check = []
                self.action_backlog = []
            else:
                # saw the obj class somewhere before
                if sub_param in self.obj_cls_to_receptacle_map:
                    self.receptacles_to_check = [self.obj_cls_to_receptacle_map[sub_param]]
                # use heuristic to determine which receptacle to check
                elif len(self.receptacles_to_check) == 0:
                    if sub_param in self.RECEPTACLES:
                        self.receptacles_to_check = self.get_list_of_receptacles_of_type(sub_param)
                    else:
                        self.receptacles_to_check = self.get_list_of_receptacles_to_search_for_object_cls(sub_param)

                        # in case priors don't work
                        self.receptacles_to_check = [r for r in list(self.receptacles.keys()) if r not in self.receptacles_to_check] + self.receptacles_to_check

                    # still no idea where to look? look at all receptacles (necessary?)
                    if len(self.receptacles_to_check) == 0:
                        self.receptacles_to_check = list(self.receptacles.keys())

                if self.curr_recep in self.receptacles_to_check and self.checked_inside_curr_recep:
                    self.receptacles_to_check.remove(self.curr_recep)

                # if holding something irrelavant, then discard it from where it was pickedup
                if len(self.inventory) > 0 and not self.is_agent_holding_right_object:
                    if self.curr_recep == self.got_inventory_from_recep:
                        return "move {} to {}".format(self.inventory[0], self.got_inventory_from_recep)
                    else:
                        return "go to {}".format(self.got_inventory_from_recep)

                # open the current receptacle if you can
                if (sub_param not in self.obj_cls_to_receptacle_map) and "closed" in obs and not self.checked_inside_curr_recep and last_action.split()[0] not in {'heat', 'cool', 'clean', 'slice'}:
                    return "open {}".format(self.curr_recep)

                # go to next receptacle
                else:
                    if len(self.action_backlog) == 0:
                        receptacle_to_check = self.receptacles_to_check[-1]
                        return "go to {}".format(receptacle_to_check)
                    else:
                        return self.action_backlog.pop()

        # if find succeded, then no receptacles to check
        self.receptacles_to_check = []

        # GOTO
        if sub_action == 'goto':
            target_receps = self.get_list_of_receptacles_of_type(sub_param)
            return "go to {}".format(target_receps[0])

        # TAKE
        if sub_action == 'take':
            if "closed" in obs and not self.checked_inside_curr_recep and len(self.visible_objects) == 0:
                return "open {}".format(self.curr_recep)
            elif len(self.visible_objects) == 0 or not any(sub_param in o for o in self.visible_objects):
                return "examine {}".format(self.curr_recep)
            else:
                obj = random.choice(objs_of_interest)
                return "take {} from {}".format(obj, self.curr_recep)

        # PUT
        if sub_action == 'put':
            if "closed" in obs and not self.checked_inside_curr_recep:
                return "open {}".format(self.curr_recep)
            else:
                obj = self.inventory[0]
                return "move {} to {}".format(obj, self.curr_recep)

        # OPEN
        if sub_action == 'open':
            return "open {}".format(self.curr_recep)

        # CLOSE
        if sub_action == 'close':
            return "close {}".format(self.curr_recep)

        # HEAT
        if sub_action == 'heat':
            inv_obj = self.inventory[0]
            return "heat {} with {}".format(inv_obj, self.curr_recep)

        # COOL
        if sub_action == 'cool':
            inv_obj = self.inventory[0]
            return "cool {} with {}".format(inv_obj, self.curr_recep)

        # CLEAN
        if sub_action == 'clean':
            inv_obj = self.inventory[0]
            return "clean {} with {}".format(inv_obj, self.curr_recep)

        # SLICE
        if sub_action == 'slice':
            if len(self.visible_objects) == 0:
                return "examine {}".format(self.curr_recep)
            else:
                obj = random.choice(objs_of_interest)
                knife_obj = self.inventory[0]
                return "slice {} with {}".format(obj, knife_obj)

        # USE
        if sub_action == 'use':
            if len(self.visible_objects) == 0:
                return "examine {}".format(self.curr_recep)
            else:
                obj = random.choice(objs_of_interest)
                return "use {}".format(obj)

        # if holding something irrelavant, then discard it from where it was pickedup
        if len(self.inventory) > 0 and not self.is_agent_holding_right_object:
            if self.curr_recep == self.got_inventory_from_recep:
                return "move {} to {}".format(self.inventory[0], self.got_inventory_from_recep)
            else:
                return "go to {}".format(self.got_inventory_from_recep)

        # examine objects on receptacles by default
        if "closed" in obs:
            return "open {}".format(self.curr_recep)
        else:
            examine_cmd = "examine {}".format(self.curr_recep)
            return examine_cmd


class PickAndPlaceSimplePolicy(BasePolicy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subgoals = [
            {'action': 'find', 'param': self.task_params['object_target']},
            {'action': 'take', 'param': self.task_params['object_target']},
            {'action': 'find', 'param': self.task_params['parent_target']},
            {'action': 'put',  'param': self.task_params['parent_target']}
        ]

    def check_subgoal_completion(self, game_state):
        obj, parent = self.task_params['object_target'], self.task_params['parent_target']
        at_right_recep, can_put_object, can_take_object, is_obj_in_obs = self.get_predicates(game_state, obj, parent)

        if self.is_agent_holding_right_object and can_put_object and at_right_recep:
            return len(self.subgoals)-1
        elif self.is_agent_holding_right_object:
            return len(self.subgoals)-2
        elif not self.is_agent_holding_right_object and can_take_object and is_obj_in_obs:
            return len(self.subgoals)-3
        else:
            return 0

    def get_predicates(self, game_state, obj, parent):
        raise NotImplementedError()


class PickTwoObjAndPlacePolicy(BasePolicy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subgoals = [
            {'action': 'find', 'param': self.task_params['object_target']},
            {'action': 'take', 'param': self.task_params['object_target']},
            {'action': 'find', 'param': self.task_params['parent_target']},
            {'action': 'put',  'param': self.task_params['parent_target']},
            {'action': 'find', 'param': self.task_params['object_target']},
            {'action': 'take', 'param': self.task_params['object_target']},
            {'action': 'find', 'param': self.task_params['parent_target']},
            {'action': 'put',  'param': self.task_params['parent_target']}
        ]

    def check_subgoal_completion(self, game_state):
        obj, parent = self.task_params['object_target'], self.task_params['parent_target']
        at_right_recep, can_put_object, can_take_object, is_obj_in_obs, is_one_object_already_inside_receptacle, trying_to_take_the_same_object = self.get_predicates(game_state, obj, parent)

        if is_one_object_already_inside_receptacle and self.is_agent_holding_right_object and can_put_object and at_right_recep:
            return len(self.subgoals)-1
        elif is_one_object_already_inside_receptacle and self.is_agent_holding_right_object:
            return len(self.subgoals)-2
        elif is_one_object_already_inside_receptacle and not self.is_agent_holding_right_object and not trying_to_take_the_same_object and can_take_object and is_obj_in_obs:
            return len(self.subgoals)-3
        elif is_one_object_already_inside_receptacle:
            return len(self.subgoals)-4
        elif not is_one_object_already_inside_receptacle and self.is_agent_holding_right_object and can_put_object and at_right_recep:
            return len(self.subgoals)-5
        elif not is_one_object_already_inside_receptacle and self.is_agent_holding_right_object:
            return len(self.subgoals)-6
        elif not is_one_object_already_inside_receptacle and not self.is_agent_holding_right_object and can_take_object and is_obj_in_obs:
            return len(self.subgoals)-7
        else:
            return 0

    def get_predicates(self, game_state, obj, parent):
        raise NotImplementedError()

class LookAtObjInLightPolicy(BasePolicy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subgoals = [
            {'action': 'find', 'param': self.task_params['object_target']},
            {'action': 'take', 'param': self.task_params['object_target']},
            {'action': 'find', 'param': self.task_params['toggle_target']},
            {'action': 'use',  'param': self.task_params['toggle_target']}
        ]

    def check_subgoal_completion(self, game_state):
        obj, toggle = self.task_params['object_target'], self.task_params['toggle_target']
        can_take_object, can_toggle_lamp, is_obj_in_obs = self.get_predicates(game_state, obj, toggle)

        if self.is_agent_holding_right_object and can_toggle_lamp:
            return len(self.subgoals)-1
        elif self.is_agent_holding_right_object:
            return len(self.subgoals)-2
        elif not self.is_agent_holding_right_object and can_take_object and is_obj_in_obs:
            return len(self.subgoals)-3
        else:
            return 0

    def get_predicates(self, game_state, obj, toggle):
        raise NotImplementedError()


class PickHeatThenPlaceInRecepPolicy(BasePolicy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subgoals = [
            {'action': 'find', 'param': self.task_params['object_target']},
            {'action': 'take', 'param': self.task_params['object_target']},
            {'action': 'goto', 'param': 'microwave'},
            {'action': 'heat', 'param': self.task_params['object_target']},
            {'action': 'find', 'param': self.task_params['parent_target']},
            {'action': 'put',  'param': self.task_params['parent_target']}
        ]

    def check_subgoal_completion(self, game_state):
        obj, parent = self.task_params['object_target'], self.task_params['parent_target']
        at_right_recep, can_heat_object, can_put_object, can_take_object, is_obj_in_obs, is_the_object_agent_holding_hot = self.get_predicates(game_state, obj, parent)

        if is_the_object_agent_holding_hot and self.is_agent_holding_right_object and can_put_object and at_right_recep:
            return len(self.subgoals)-1
        elif is_the_object_agent_holding_hot and self.is_agent_holding_right_object:
            return len(self.subgoals)-2
        elif self.is_agent_holding_right_object and can_heat_object:
            return len(self.subgoals)-3
        elif self.is_agent_holding_right_object:
            return len(self.subgoals)-4
        elif not self.is_agent_holding_right_object and can_take_object and is_obj_in_obs:
            return len(self.subgoals)-5
        else:
            return 0

    def get_predicates(self, game_state, obj, parent):
        raise NotImplementedError()


class PickCoolThenPlaceInRecepPolicy(BasePolicy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subgoals = [
            {'action': 'find', 'param': self.task_params['object_target']},
            {'action': 'take', 'param': self.task_params['object_target']},
            {'action': 'goto', 'param': 'fridge'},
            {'action': 'cool', 'param': self.task_params['object_target']},
            {'action': 'find', 'param': self.task_params['parent_target']},
            {'action': 'put',  'param': self.task_params['parent_target']}
        ]

    def check_subgoal_completion(self, game_state):
        obj, parent = self.task_params['object_target'], self.task_params['parent_target']
        at_right_recep, can_cool_object, can_put_object, can_take_object, is_obj_in_obs, is_the_object_agent_holding_cool = self.get_predicates(game_state, obj, parent)

        if is_the_object_agent_holding_cool and self.is_agent_holding_right_object and can_put_object and at_right_recep:
            return len(self.subgoals)-1
        elif is_the_object_agent_holding_cool and self.is_agent_holding_right_object:
            return len(self.subgoals)-2
        elif self.is_agent_holding_right_object and can_cool_object:
            return len(self.subgoals)-3
        elif self.is_agent_holding_right_object:
            return len(self.subgoals)-4
        elif not self.is_agent_holding_right_object and can_take_object and is_obj_in_obs:
            return len(self.subgoals)-5
        else:
            return 0

    def get_predicates(self, game_state, obj, parent):
        raise NotImplementedError()

class PickCleanThenPlaceInRecepPolicy(BasePolicy):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subgoals = [
            {'action': 'find', 'param': self.task_params['object_target']},
            {'action': 'take', 'param': self.task_params['object_target']},
            {'action': 'goto', 'param': 'sinkbasin'},
            {'action': 'clean', 'param': self.task_params['object_target']},
            {'action': 'find', 'param': self.task_params['parent_target']},
            {'action': 'put',  'param': self.task_params['parent_target']}
        ]
        self.is_agent_holding_right_object = False

    def check_subgoal_completion(self, game_state):
        obj, parent = self.task_params['object_target'], self.task_params['parent_target']

        at_right_recep, can_clean_object, can_put_object, can_take_object, is_obj_in_obs, is_the_object_agent_holding_isclean = self.get_predicates(
            game_state, obj, parent)

        if is_the_object_agent_holding_isclean and self.is_agent_holding_right_object and can_put_object and at_right_recep:
            return len(self.subgoals)-1
        elif is_the_object_agent_holding_isclean and self.is_agent_holding_right_object:
            return len(self.subgoals)-2
        elif self.is_agent_holding_right_object and can_clean_object:
            return len(self.subgoals)-3
        elif self.is_agent_holding_right_object:
            return len(self.subgoals)-4
        elif not self.is_agent_holding_right_object and can_take_object and is_obj_in_obs:
            return len(self.subgoals)-5
        else:
            return 0

    def get_predicates(self, game_state, obj, parent):
        raise NotImplementedError()

class HandCodedAgent(Agent):
    """ Handcoded expert for solving tasks. Based on a set of heuristics to solve ALFRED tasks.
        Not guaranteed to succeed or be optimal"""

    def __init__(self, max_steps=150):
        self.max_steps = max_steps

    def get_task_policy(self, task_param):
        task_type = task_param['task_type']
        task_class_str = task_type.replace("_", " ").title().replace(" ", '') + "Policy"
        if task_class_str in globals():
            return globals()[task_class_str]
        else:
            raise Exception("Invalid Task Type: %s" % task_type)

    def reset(self, game="INVALID"):
        traj_data_file = os.path.join(os.path.dirname(game), 'traj_data.json')
        with open(traj_data_file, 'r') as f:
            traj_data = json.load(f)

        self.task_params = {**traj_data['pddl_params'],
                            'task_type': traj_data['task_type']}
        self.task_params = dict((k, v.lower() if v in constants.OBJECTS else v) for k, v in self.task_params.items())

        policy_class = self.get_task_policy(self.task_params)
        self.policy = policy_class(self.task_params, max_steps=self.max_steps)

    def act(self, game_state, reward, done, last_action=''):
        action = self.policy.act(game_state, last_action)

        if action == last_action or (last_action == "look" and action not in game_state['admissible_commands']):
            return random.choice(game_state['admissible_commands'])

        return action

    def observe(self, obs):
        self.policy.observe(obs)
