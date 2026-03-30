import hashlib
import random
import string

from collections import Counter
from textworld.logic import Proposition, Variable

import os
import sys
import alfworld.gen.goal_library as glib
import alfworld.gen.constants as constants

import numpy as np
from json import JSONEncoder


class Demangler(object):

    def __init__(self, obj_dict=None, game_infos=None, shuffle=False):
        if obj_dict is None:
            self.obj_count = Counter()
        else:
            self.obj_count = obj_dict

        self.obj_names = {}
        if game_infos:
            ids = sorted([info.id for info in game_infos.values()])
            if shuffle:
                random.shuffle(ids)

            # count the number of instances
            for id in ids:
                splits = id.split("_bar_", 1)
                if len(splits) > 1:
                    name, rest = splits
                    if "basin" in id:
                        name += "basin"
                    self.obj_count[name] += 1

            # make list of num ids for each object (shuffle the ids if required)
            obj_num_ids = {}
            for obj, count in self.obj_count.most_common():
                num_ids = list(range(count+1)[1:])  # start from index 1
                obj_num_ids[obj] = num_ids

            # assign unique num id for each object based on precomputed list
            for id in ids:
                text = id
                text = text.replace("_bar_", "|")
                text = text.replace("_minus_", "-")
                text = text.replace("_plus_", "+")
                text = text.replace("_dot_", ".")
                text = text.replace("_comma_", ",")

                splits = text.split("|", 1)
                if len(splits) == 1:
                    self.obj_names[id] = {'name': text, 'id': 0}
                else:
                    name, rest = splits
                    if "basin" in id:
                        name += "basin"
                    self.obj_names[id] = {'name': name, 'id': obj_num_ids[name].pop()}

    def demangle_alfred_name(self, text):
        assert(text in self.obj_names)
        name, id = self.obj_names[text].values()
        id = str(id) if id > 0 else ""
        res = "{} {}".format(name, id)
        return res


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


def get_templated_task_desc(traj_data):

    pddl_params = traj_data['pddl_params']
    goal_str = traj_data['task_type']
    if pddl_params['object_sliced']:
        goal_str += "_slice"

    template = random.choice(glib.gdict[goal_str]['templates'])
    obj = pddl_params['object_target'].lower()
    recep = pddl_params['parent_target'].lower()
    toggle = pddl_params['toggle_target'].lower()
    mrecep = pddl_params['mrecep_target'].lower()
    filled_in_str = template.format(obj=obj, recep=recep, toggle=toggle, mrecep=mrecep)
    return filled_in_str


def clean_alfred_facts(facts):
    # TODO: fix this?
    demangler = Demangler()

    def _clean_fact(fact: Proposition):
        args = [Variable(demangler.demangle_alfred_name(arg.name), arg.type) for arg in fact.arguments]
        return Proposition(fact.name, args)

    facts = [_clean_fact(fact) for fact in facts if not fact.name.startswith("new-axiom@")]
    return facts


def add_task_to_grammar(grammar_str, traj_data, goal_desc_human_anns_prob=0.0, rng=random):
    if rng.random() < goal_desc_human_anns_prob:
        task_desc = get_human_anns_task_desc(traj_data, rng)
    else:
        task_desc = get_templated_task_desc(traj_data)
    return grammar_str.replace("UNKNOWN GOAL", task_desc)


def get_human_anns_task_desc(traj_data, rng=random):
    return rng.choice(traj_data['turk_annotations']['anns'])['task_desc']


def parse_objects(desc):
    '''
    extract objects after "you see" and before "your task is to:"
    '''
    # TODO: replace with a noun-phrase extractor
    objs = []
    if "you see" in desc.lower():
        obj_str = desc.lower().replace(" and", ",")
        obj_str = obj_str.split("you see ", 1)[1]
        obj_str = obj_str.partition("your task is to:")[0]
        for s in obj_str.split(","):
            item = s.translate(str.maketrans('', '', string.punctuation))
            item = item.strip().replace("a ", "")
            if item:
                objs.append(item)
    return objs


def extract_admissible_commands_with_heuristics(intro, frame_desc, feedback,
                                                curr_recep, inventory):
    '''
    Heavily engineered admissible commands extraction. Lots of commonsense and heuristics used to prune list.
    '''

    at_recep = str(curr_recep) if curr_recep != "nothing" else ""
    at_recep_type = at_recep.split()[0] if at_recep else ""
    in_inv = str(inventory[0]).lower() if len(inventory) > 0 else ""

    OPENABLE_RECEPTACLES = [r.lower() for r in constants.OPENABLE_CLASS_LIST]

    assert intro  # intro should be non-empty

    admissible_commands = []
    templates = [
        "go to {recep}",
        "open {recep}",
        "close {recep}",
        "take {obj} from {recep}",
        "move {obj} to {recep}",
        "use {lamp}",
        "heat {obj} with {microwave}",
        "cool {obj} with {fridge}",
        "clean {obj} with {cleaner}",
        "slice {obj} with {knife}",
        "inventory",
        "look",
        "help",
        "examine {obj}",
        "examine {recep}"
    ]

    # parse interactable and navigable objects
    receps = parse_objects(intro)
    objects = parse_objects(frame_desc) if frame_desc else []
    lamps = [obj for obj in objects if 'lamp' in obj]
    microwaves = [obj for obj in receps if 'microwave' in obj]
    cleaners = [obj for obj in receps if 'sink' in obj or 'bathtub' in obj]
    fridges = [obj for obj in receps if 'fridge' in obj]
    knives = [obj for obj in receps if 'knife' in obj]

    # populate templates
    for t in templates:
        if 'take {obj} from {recep}' in t:
            if at_recep and not in_inv:
                for obj in objects:
                    if 'desklamp' not in obj and 'floorlamp' not in obj:
                        admissible_commands.append(t.format(recep=at_recep, obj=obj))
        elif 'move {obj} to {recep}' in t:
            if in_inv and at_recep:
                admissible_commands.append(t.format(recep=at_recep, obj=in_inv))
        elif '{obj}' in t and '{microwave}' in t:
            if 'microwave' in at_recep and in_inv:
                for microwave in microwaves:
                    admissible_commands.append(t.format(microwave=microwave, obj=in_inv))
        elif '{obj}' in t and '{fridge}' in t:
            if 'fridge' in at_recep and in_inv:
                for fridge in fridges:
                    admissible_commands.append(t.format(fridge=fridge, obj=in_inv))
        elif '{obj}' in t and '{cleaner}' in t:
            if ('sink' in at_recep or 'bathtub' in at_recep) and in_inv:
                for cleaner in cleaners:
                    admissible_commands.append(t.format(cleaner=cleaner, obj=in_inv))
        elif '{obj}' in t and '{knife}' in t:
            if 'knife' in in_inv:
                for knife in knives:
                    for obj in objects:
                        admissible_commands.append(t.format(knife=knife, obj=obj))
        elif 'open {recep}' in t:
            if at_recep:
                if at_recep_type in OPENABLE_RECEPTACLES: # and ("is closed" in feedback or "You close" in feedback):
                    admissible_commands.append(t.format(recep=at_recep))
        elif 'close {recep}' in t:
            if at_recep:
                if at_recep_type in OPENABLE_RECEPTACLES: # and "is open" in feedback:
                    admissible_commands.append(t.format(recep=at_recep))
        elif 'examine {recep}' in t:
            if at_recep:
                admissible_commands.append(t.format(recep=at_recep))
        elif 'examine {obj}' in t:
            if in_inv:
                admissible_commands.append(t.format(obj=in_inv))
        elif 'go to {recep}' in t:
            for recep in receps:
                if recep != at_recep:
                    admissible_commands.append(t.format(recep=recep))
        elif '{recep}' in t:
            for recep in receps:
                admissible_commands.append(t.format(recep=recep))
        elif '{obj}' in t:
            for obj in objects:
                admissible_commands.append(t.format(obj=obj))
        elif '{lamp}' in t:
            for lamp in lamps:
                admissible_commands.append(t.format(lamp=lamp))
        else:
            admissible_commands.append(t)

    return admissible_commands


def extract_admissible_commands(intro, frame_desc):
    '''
    exhaustive list of admissible commands
    '''

    admissible_commands = []
    templates = [
        "go to {recep}",
        "open {recep}",
        "close {recep}",
        "take {obj} from {recep}",
        "move {obj} to {recep}",
        "use {lamp}",
        "heat {obj} with {microwave}",
        "cool {obj} with {fridge}",
        "clean {obj} with {cleaner}",
        "slice {obj} with {knife}",
        "inventory",
        "look",
        "help",
        "examine {obj}",
        "examine {recep}"
    ]

    # parse interactable and navigable objects
    receps = parse_objects(intro)
    objects = parse_objects(frame_desc) if frame_desc else []
    lamps = [obj for obj in objects if 'lamp' in obj]
    microwaves = [obj for obj in receps if 'microwave' in obj]
    cleaners = [obj for obj in receps if 'sink' in obj or 'bathtub' in obj]
    fridges = [obj for obj in receps if 'fridge' in obj]
    knives = [obj for obj in receps if 'knife' in obj]

    # populate templates
    for t in templates:
        if '{recep}' in t and '{obj}' in t:
            for recep in receps:
                for obj in objects:
                    admissible_commands.append(t.format(recep=recep, obj=obj))
        elif '{obj}' in t and '{microwave}' in t:
            for microwave in microwaves:
                for obj in objects:
                    admissible_commands.append(t.format(microwave=microwave, obj=obj))
        elif '{obj}' in t and '{fridge}' in t:
            for fridge in fridges:
                for obj in objects:
                    admissible_commands.append(t.format(fridge=fridge, obj=obj))
        elif '{obj}' in t and '{cleaner}' in t:
            for cleaner in cleaners:
                for obj in objects:
                    admissible_commands.append(t.format(cleaner=cleaner, obj=obj))
        elif '{obj}' in t and '{knife}' in t:
            for knife in knives:
                for obj in objects:
                    admissible_commands.append(t.format(knife=knife, obj=obj))
        elif '{recep}' in t:
            for recep in receps:
                admissible_commands.append(t.format(recep=recep))
        elif '{obj}' in t:
            for obj in objects:
                admissible_commands.append(t.format(obj=obj))
        elif '{lamp}' in t:
            for lamp in lamps:
                admissible_commands.append(t.format(lamp=lamp))
        else:
            admissible_commands.append(t)

    return admissible_commands
