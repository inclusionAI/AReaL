import os
import sys
import json

import copy
import random
import alfworld.gen.constants as constants
from alfworld.gen.utils.image_util import compress_mask, decompress_mask
from alfworld.agents.utils.misc import get_templated_task_desc, get_human_anns_task_desc, NumpyArrayEncoder
from enum import Enum

class BaseAgent(object):
    '''
    Base class for controllers
    '''

    # constants
    RECEPTACLES = set(constants.RECEPTACLES) | {'Sink', 'Bathtub'}
    OBJECTS = (set(constants.OBJECTS_WSLICED) - set(RECEPTACLES)) | set(constants.MOVABLE_RECEPTACLES)
    OBJECTS -= {'Blinds', 'Boots', 'Cart', 'Chair', 'Curtains', 'Footstool', 'Mirror', 'LightSwtich', 'Painting', 'Poster', 'ShowerGlass', 'Window'}
    STATIC_RECEPTACLES = set(RECEPTACLES) - set(constants.MOVABLE_RECEPTACLES)

    # action enum
    class Action(Enum):
        PASS = 0,
        GOTO = 1,
        PICK = 2,
        PUT = 3,
        OPEN = 4,
        CLOSE = 5,
        TOGGLE = 6,
        HEAT = 7,
        CLEAN = 8,
        COOL = 9,
        SLICE = 10,
        INVENTORY = 11,
        EXAMINE = 12,
        LOOK = 13

    def __init__(self, env, traj_data, traj_root,
                 load_receps=False, debug=False,
                 goal_desc_human_anns_prob=0.0,
                 recep_filename='receps.json', exhaustive_exploration=False):
        self.env = env
        self.traj_data = traj_data
        self.debug = debug
        self.traj_root = traj_root
        self.load_receps = load_receps
        self.recep_file = os.path.join(traj_root, recep_filename)
        self.objects = {}
        self.receptacles = {}
        self.visible_objects = []
        self.exhaustive_exploration = exhaustive_exploration
        self.goal_desc_human_anns_prob = goal_desc_human_anns_prob

        self.feedback = ""
        self.curr_loc = {'action': "Pass"}
        self.curr_recep = "nothing"
        self.inventory = []
        self.intro = ""
        self.frame_desc = ""

        self.init_scene(load_receps)
        self.setup_navigator()
        self.print_intro()

    # explore the scene to build receptacle map
    def init_scene(self, load_receps):
        if load_receps and os.path.isfile(self.recep_file):
            with open(self.recep_file, 'r') as f:
                self.receptacles = json.load(f)
            for recep_id, recep in self.receptacles.items():
                if 'mask' in recep:
                    recep['mask'] = decompress_mask(recep['mask'])
        else:
            self.receptacles = {}
            if self.exhaustive_exploration:
                self.explore_scene_exhaustively()
            else:
                self.explore_scene()

    def get_object(self, name, obj_dict):
        for id, obj in obj_dict.items():
            if obj['num_id'] == name:
                return obj
        return None

    def get_next_num_id(self, object_type, obj_dict):
        return len([obj for _, obj in obj_dict.items() if obj['object_type'] == object_type]) + 1

    def fix_and_comma_in_the_end(self, desc):
        sc = desc.split(",")
        if len(sc) > 2:
            return ",".join(sc[:-2]) + ", and%s." % sc[-2]
        elif len(sc) == 2:
            return desc.rstrip(",") + "."
        else:
            return desc

    def explore_scene(self):
        raise NotImplementedError()

    def explore_scene_exhaustively(self):
        raise NotImplementedError()

    def get_admissible_commands(self):
        raise NotImplementedError()

    def get_instance_seg(self):
        raise NotImplementedError()

    def get_object_state(self, object_id):
        raise NotImplementedError()

    # dump receptacle map to disk
    def save_receps(self):
        receptacles = copy.deepcopy(self.receptacles)
        for recep_id, recep in receptacles.items():
            if 'mask' in recep:
                recep['mask'] = compress_mask(recep['mask'])
        with open(self.recep_file, 'w') as f:
            json.dump(receptacles, f) #, cls=NumpyArrayEncoder)

    # display initial observation and task text
    def print_intro(self):
        self.feedback = "-= Welcome to TextWorld, ALFRED! =-\n\nYou are in the middle of a room. Looking quickly around you, you see "
        recep_list = ["a %s," % (recep['num_id']) for id, recep in self.receptacles.items()]
        self.feedback += self.fix_and_comma_in_the_end(" ".join(recep_list)) + "\n\n"

        if random.random() < self.goal_desc_human_anns_prob:
            task = get_human_anns_task_desc(self.traj_data)
        else:
            task = get_templated_task_desc(self.traj_data)

        self.feedback += "Your task is to: %s" % task

        self.intro = str(self.feedback)

    # choose between different navigator available
    def setup_navigator(self):
        self.navigator = self.env  # by default, directly teleport with THOR API

    def print_frame(self, recep, loc):
        raise NotImplementedError()

    # display properties of an object
    def print_object(self, object):
        object_id, object_name = object['object_id'], object['num_id']

        is_clean, is_cool, is_hot, is_sliced = self.get_object_state(object_id)

        # by default, nothing interesting about the object
        feedback = "This is a normal %s" % object_name

        sliced_str = "sliced " if is_sliced else ""
        if is_hot and is_cool and is_clean:
            feedback = "This is a hot/cold and clean %s%s." % (sliced_str, object_name)  # TODO: weird?
        elif is_hot and is_clean:
            feedback = "This is a hot and clean %s%s." % (sliced_str, object_name)
        elif is_cool and is_clean:
            feedback = "This is a cool and clean %s%s." % (sliced_str, object_name)
        elif is_hot and is_cool:
            feedback = "This is a hot/cold %s%s." % (sliced_str, object_name)
        elif is_clean:
            feedback = "This is a clean %s%s." % (sliced_str, object_name)
        elif is_hot:
            feedback = "This is a hot %s%s." % (sliced_str, object_name)
        elif is_cool:
            feedback = "This is a cool %s%s." % (sliced_str, object_name)

        return feedback

    # command parser
    def parse_command(self, action_str):

        def get_triplet(astr, key):
            astr = astr.replace(key, "").split()
            obj, rel, tar = ' '.join(astr[:2]), astr[2], ' '.join(astr[-2:])
            return obj, rel, tar

        action_str = str(action_str).lower().strip()

        if "go to " in action_str:
            tar = action_str.replace("go to ", "")
            return {'action': self.Action.GOTO, 'tar': tar}
        elif "take " in action_str:
            obj, rel, tar = get_triplet(action_str, "take ")
            return {'action': self.Action.PICK, 'obj': obj, 'rel': rel, 'tar': tar}
        elif "put " in action_str:
            obj, rel, tar = get_triplet(action_str, "put ")
            return {'action': self.Action.PUT, 'obj': obj, 'rel': rel, 'tar': tar}
        elif "open " in action_str:
            tar = action_str.replace("open ", "")
            return {'action': self.Action.OPEN, 'tar': tar}
        elif "close " in action_str:
            tar = action_str.replace("close ", "")
            return {'action': self.Action.CLOSE, 'tar': tar}
        elif "use " in action_str:
            tar = action_str.replace("use ", "")
            return {'action': self.Action.TOGGLE, 'tar': tar}
        elif "heat " in action_str:
            obj, rel, tar = get_triplet(action_str, "heat ")
            return {'action': self.Action.HEAT, 'obj': obj, 'rel': rel, 'tar': tar}
        elif "cool " in action_str:
            obj, rel, tar = get_triplet(action_str, "cool ")
            return {'action': self.Action.COOL, 'obj': obj, 'rel': rel, 'tar': tar}
        elif "clean " in action_str:
            obj, rel, tar = get_triplet(action_str, "clean ")
            return {'action': self.Action.CLEAN, 'obj': obj, 'rel': rel, 'tar': tar}
        elif "slice " in action_str:
            obj, rel, tar = get_triplet(action_str, "slice ")
            return {'action': self.Action.SLICE, 'obj': obj, 'rel': rel, 'tar': tar}
        elif "inventory" in action_str:
            return {'action': self.Action.INVENTORY}
        elif "examine " in action_str:
            tar = action_str.replace("examine ", "")
            return {'action': self.Action.EXAMINE, 'tar': tar}
        elif "look" in action_str:
            return {'action': self.Action.LOOK}
        else:
            return {'action': self.Action.PASS}

    def navigate(self, teleport_loc):
        return self.navigator.step(teleport_loc)

    def step(self, action_str):
        self.feedback = "Nothing happens."
        return self.feedback



