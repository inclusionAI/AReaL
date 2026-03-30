import os
import sys

import time
import multiprocessing as mp
import json
import random
import shutil
import argparse
import numpy as np
import pandas as pd
from collections import OrderedDict
from datetime import datetime

import alfworld.gen.constants
from alfworld.gen.agents.deterministic_planner_agent import DeterministicPlannerAgent
from alfworld.env.thor_env import ThorEnv
from alfworld.gen.game_states.task_game_state_full_knowledge import TaskGameStateFullKnowledge
from alfworld.gen.utils.video_util import VideoSaver
from alfworld.gen.utils.dataset_management_util import load_successes_from_disk, load_fails_from_disk

# params
RAW_IMAGES_FOLDER = 'raw_images/'
DATA_JSON_FILENAME = 'traj_data.json'

# video saver
video_saver = VideoSaver()

# structures to help with constraint enforcement.
goal_to_required_variables = {"pick_and_place_simple": {"pickup", "receptacle", "scene"},
                              "pick_two_obj_and_place": {"pickup", "receptacle", "scene"},
                              "look_at_obj_in_light": {"pickup", "receptacle", "scene"},
                              "pick_clean_then_place_in_recep": {"pickup", "receptacle", "scene"},
                              "pick_heat_then_place_in_recep": {"pickup", "receptacle", "scene"},
                              "pick_cool_then_place_in_recep": {"pickup", "receptacle", "scene"},
                              "pick_and_place_with_movable_recep": {"pickup", "movable", "receptacle", "scene"}}
goal_to_pickup_type = {'pick_heat_then_place_in_recep': 'Heatable',
                       'pick_cool_then_place_in_recep': 'Coolable',
                       'pick_clean_then_place_in_recep': 'Cleanable'}
goal_to_receptacle_type = {'look_at_obj_in_light': "Toggleable"}
goal_to_invalid_receptacle = {'pick_heat_then_place_in_recep': {'Microwave'},
                              'pick_cool_then_place_in_recep': {'Fridge'},
                              'pick_clean_then_place_in_recep': {'SinkBasin'},
                              'pick_two_obj_and_place': {'CoffeeMachine', 'ToiletPaperHanger', 'HandTowelHolder'}}

scene_id_to_objs = {}
obj_to_scene_ids = {}
scenes_for_goal = {g: [] for g in constants.GOALS}
scene_to_type = {}


def sample_task_params(succ_traj, full_traj, fail_traj,
                       goal_candidates, pickup_candidates, movable_candidates, receptacle_candidates, scene_candidates,
                       inject_noise=10):
    # Get the current conditional distributions of all variables (goal/pickup/receptacle/scene).
    goal_weight = [(1 / (1 + np.random.randint(0, inject_noise + 1) + succ_traj.loc[
        (succ_traj['pickup'].isin(pickup_candidates) if 'pickup' in goal_to_required_variables[c] else True) &
        (succ_traj['movable'].isin(movable_candidates) if 'movable' in goal_to_required_variables[c] else True) &
        (succ_traj['receptacle'].isin(receptacle_candidates) if 'receptacle' in goal_to_required_variables[c] else True)
        & (succ_traj['scene'].isin(scene_candidates) if 'scene' in goal_to_required_variables[c] else True)]
            ['goal'].tolist().count(c)))  # Conditional.
                   * (1 / (1 + succ_traj['goal'].tolist().count(c)))  # Prior.
                   for c in goal_candidates]
    goal_probs = [w / sum(goal_weight) for w in goal_weight]

    pickup_weight = [(1 / (1 + np.random.randint(0, inject_noise + 1) +
                           sum([succ_traj.loc[
                                    succ_traj['goal'].isin([g]) &
                                    (succ_traj['movable'].isin(movable_candidates)
                                     if 'movable' in goal_to_required_variables[g] else True) &
                                    (succ_traj['receptacle'].isin(receptacle_candidates)
                                     if 'receptacle' in goal_to_required_variables[g] else True) &
                                    (succ_traj['scene'].isin(scene_candidates)
                                     if 'scene' in goal_to_required_variables[g] else True)]
                                ['pickup'].tolist().count(c) for g in goal_candidates])))
                     * (1 / (1 + succ_traj['pickup'].tolist().count(c)))
                     for c in pickup_candidates]
    pickup_probs = [w / sum(pickup_weight) for w in pickup_weight]

    movable_weight = [(1 / (1 + np.random.randint(0, inject_noise + 1) +
                            sum([succ_traj.loc[
                                     succ_traj['goal'].isin([g]) &
                                     (succ_traj['pickup'].isin(pickup_candidates)
                                      if 'pickup' in goal_to_required_variables[g] else True) &
                                     (succ_traj['receptacle'].isin(receptacle_candidates)
                                      if 'receptacle' in goal_to_required_variables[g] else True) &
                                     (succ_traj['scene'].isin(scene_candidates)
                                      if 'scene' in goal_to_required_variables[g] else True)]
                                 ['movable'].tolist().count(c) for g in goal_candidates])))
                      * (1 / (1 + succ_traj['movable'].tolist().count(c)))
                      for c in movable_candidates]
    movable_probs = [w / sum(movable_weight) for w in movable_weight]

    receptacle_weight = [(1 / (1 + np.random.randint(0, inject_noise + 1) +
                               sum([succ_traj.loc[
                                        succ_traj['goal'].isin([g]) &
                                        (succ_traj['pickup'].isin(pickup_candidates)
                                         if 'pickup' in goal_to_required_variables[g] else True) &
                                        (succ_traj['movable'].isin(movable_candidates)
                                         if 'movable' in goal_to_required_variables[g] else True) &
                                        (succ_traj['scene'].isin(scene_candidates)
                                         if 'scene' in goal_to_required_variables[g] else True)]
                                    ['receptacle'].tolist().count(c) for g in goal_candidates])))
                         * (1 / (1 + succ_traj['receptacle'].tolist().count(c)))
                         for c in receptacle_candidates]
    receptacle_probs = [w / sum(receptacle_weight) for w in receptacle_weight]
    scene_weight = [(1 / (1 + np.random.randint(0, inject_noise + 1) +
                          sum([succ_traj.loc[
                                   succ_traj['goal'].isin([g]) &
                                   (succ_traj['pickup'].isin(pickup_candidates)
                                    if 'pickup' in goal_to_required_variables[g] else True) &
                                   (succ_traj['movable'].isin(movable_candidates)
                                    if 'movable' in goal_to_required_variables[g] else True) &
                                   (succ_traj['receptacle'].isin(receptacle_candidates)
                                    if 'receptacle' in goal_to_required_variables[g] else True)]
                               ['scene'].tolist().count(c) for g in goal_candidates])))
                    * (1 / (1 + succ_traj['scene'].tolist().count(c)))
                    for c in scene_candidates]
    scene_probs = [w / sum(scene_weight) for w in scene_weight]

    # Calculate the probability difference between each value and the maximum so we can iterate over them to find a
    # next-best candidate to sample subject to the constraints of knowing which will fail.
    diffs = [("goal", goal_candidates[idx], goal_probs[idx] - min(goal_probs))
             for idx in range(len(goal_candidates)) if len(goal_candidates) > 1]
    diffs.extend([("pickup", pickup_candidates[idx], pickup_probs[idx] - min(pickup_probs))
                  for idx in range(len(pickup_candidates)) if len(pickup_candidates) > 1])
    diffs.extend([("movable", movable_candidates[idx], movable_probs[idx] - min(movable_probs))
                  for idx in range(len(movable_candidates)) if len(movable_candidates) > 1])
    diffs.extend([("receptacle", receptacle_candidates[idx], receptacle_probs[idx] - min(receptacle_probs))
                  for idx in range(len(receptacle_candidates)) if len(receptacle_candidates) > 1])
    diffs.extend([("scene", scene_candidates[idx], scene_probs[idx] - min(scene_probs))
                  for idx in range(len(scene_candidates)) if len(scene_candidates) > 1])

    # Iteratively pop the next biggest difference until we find a combination that is valid (e.g., not already
    # flagged as impossible by the simulator).
    variable_value_by_diff = {}
    diffs_as_keys = []  # list of diffs; index into list will be used as key values.
    for _, _, diff in diffs:
        already_keyed = False
        for existing_diff in diffs_as_keys:
            if np.isclose(existing_diff, diff):
                already_keyed = True
                break
        if not already_keyed:
            diffs_as_keys.append(diff)
    for variable, value, diff in diffs:
        key = None
        for kidx in range(len(diffs_as_keys)):
            if np.isclose(diffs_as_keys[kidx], diff):
                key = kidx
        if key not in variable_value_by_diff:
            variable_value_by_diff[key] = []
        variable_value_by_diff[key].append((variable, value))

    for key, diff in sorted(enumerate(diffs_as_keys), key=lambda x: x[1], reverse=True):
        variable_value = variable_value_by_diff[key]
        random.shuffle(variable_value)
        for variable, value in variable_value:

            # Select a goal.
            if variable == "goal":
                gtype = value
                # print("sampled goal '%s' with prob %.4f" % (gtype, goal_probs[goal_candidates.index(gtype)]))
                _goal_candidates = [gtype]

                _pickup_candidates = pickup_candidates[:]
                _movable_candidates = movable_candidates[:]
                _receptacle_candidates = receptacle_candidates[:]
                _scene_candidates = scene_candidates[:]

            # Select a pickup object.
            elif variable == "pickup":
                pickup_obj = value
                # print("sampled pickup object '%s' with prob %.4f" %
                #       (pickup_obj,  pickup_probs[pickup_candidates.index(pickup_obj)]))
                _pickup_candidates = [pickup_obj]

                _goal_candidates = goal_candidates[:]
                _movable_candidates = movable_candidates[:]
                _receptacle_candidates = receptacle_candidates[:]
                _scene_candidates = scene_candidates[:]

            # Select a movable object.
            elif variable == "movable":
                movable_obj = value
                # print("sampled movable object '%s' with prob %.4f" %
                #       (movable_obj,  movable_probs[movable_candidates.index(movable_obj)]))
                _movable_candidates = [movable_obj]
                _goal_candidates = [g for g in goal_candidates if g == 'pick_and_place_with_movable_recep']

                _pickup_candidates = pickup_candidates[:]
                _receptacle_candidates = receptacle_candidates[:]
                _scene_candidates = scene_candidates[:]

            # Select a receptacle.
            elif variable == "receptacle":
                receptacle_obj = value
                # print("sampled receptacle object '%s' with prob %.4f" %
                #       (receptacle_obj, receptacle_probs[receptacle_candidates.index(receptacle_obj)]))
                _receptacle_candidates = [receptacle_obj]

                _goal_candidates = goal_candidates[:]
                _pickup_candidates = pickup_candidates[:]
                _movable_candidates = movable_candidates[:]
                _scene_candidates = scene_candidates[:]

            # Select a scene.
            else:
                sampled_scene = value
                # print("sampled scene %s with prob %.4f" %
                #       (sampled_scene, scene_probs[scene_candidates.index(sampled_scene)]))
                _scene_candidates = [sampled_scene]

                _goal_candidates = goal_candidates[:]
                _pickup_candidates = pickup_candidates[:]
                _movable_candidates = movable_candidates[:]
                _receptacle_candidates = receptacle_candidates[:]
            # Perform constraint propagation to determine whether this is a valid assignment.
            propagation_finished = False
            while not propagation_finished:
                assignment_lens = (len(_goal_candidates), len(_pickup_candidates), len(_movable_candidates),
                                   len(_receptacle_candidates), len(_scene_candidates))
                # Constraints on goal.
                _goal_candidates = [g for g in _goal_candidates if
                                    (g not in goal_to_pickup_type or
                                     len(set(_pickup_candidates).intersection(  # Pickup constraint.
                                        constants.VAL_ACTION_OBJECTS[goal_to_pickup_type[g]])) > 0)
                                    and (g not in goal_to_receptacle_type or
                                         np.any([r in constants.VAL_ACTION_OBJECTS[goal_to_receptacle_type[g]]
                                                for r in _receptacle_candidates]))  # Valid by goal receptacle const.
                                    and (g not in goal_to_invalid_receptacle or
                                         len(set(_receptacle_candidates).difference(
                                            goal_to_invalid_receptacle[g])) > 0)  # Invalid by goal receptacle const.
                                    and len(set(_scene_candidates).intersection(
                                        scenes_for_goal[g])) > 0  # Scene constraint
                                    ]

                # Define whether to consider constraints for each role based on current set of candidate goals.
                pickup_constrained = np.any(["pickup" in goal_to_required_variables[g] for g in _goal_candidates])
                movable_constrained = np.any(["movable" in goal_to_required_variables[g] for g in _goal_candidates])
                receptacle_constrained = np.any(["receptacle" in goal_to_required_variables[g]
                                                 for g in _goal_candidates])
                scene_constrained = np.any(["scene" in goal_to_required_variables[g] for g in _goal_candidates])

                # Constraints on pickup obj.
                _pickup_candidates = [p for p in _pickup_candidates if
                                      np.any([g not in goal_to_pickup_type or
                                              p in constants.VAL_ACTION_OBJECTS[goal_to_pickup_type[g]]
                                              for g in _goal_candidates])  # Goal constraint.
                                      and (not movable_constrained or
                                           np.any([p in constants.VAL_RECEPTACLE_OBJECTS[m]
                                                  for m in _movable_candidates]))  # Movable constraint.
                                      and (not receptacle_constrained or
                                           np.any([r in constants.VAL_ACTION_OBJECTS["Toggleable"] or
                                                  p in constants.VAL_RECEPTACLE_OBJECTS[r]
                                                  for r in _receptacle_candidates]))  # Receptacle constraint.
                                      and (not scene_constrained or
                                           np.any([s in obj_to_scene_ids[constants.OBJ_PARENTS[p]]
                                                   for s in _scene_candidates])) # Scene constraint
                                      ]
                # Constraints on movable obj.
                _movable_candidates = [m for m in _movable_candidates if
                                       'pick_and_place_with_movable_recep' in _goal_candidates  # Goal constraint
                                       and (not pickup_constrained or
                                            np.any([p in constants.VAL_RECEPTACLE_OBJECTS[m]
                                                   for p in _pickup_candidates]))  # Pickup constraint.
                                       and (not receptacle_constrained or
                                            np.any([r in constants.VAL_RECEPTACLE_OBJECTS and
                                                   m in constants.VAL_RECEPTACLE_OBJECTS[r]
                                                   for r in _receptacle_candidates]))  # Receptacle constraint.
                                       and (not scene_constrained or
                                            np.any([s in obj_to_scene_ids[constants.OBJ_PARENTS[m]]
                                                    for s in _scene_candidates]))  # Scene constraint
                                       ]
                # Constraints on receptacle obj.
                _receptacle_candidates = [r for r in _receptacle_candidates if
                                          np.any([(g not in goal_to_receptacle_type or
                                                   r in constants.VAL_ACTION_OBJECTS[goal_to_receptacle_type[g]]) and
                                                  (g not in goal_to_invalid_receptacle or
                                                  r not in goal_to_invalid_receptacle[g])
                                                  for g in _goal_candidates])  # Goal constraint.
                                          and (not receptacle_constrained or
                                               r in constants.VAL_ACTION_OBJECTS["Toggleable"] or
                                               np.any([p in constants.VAL_RECEPTACLE_OBJECTS[r]
                                                       for p in _pickup_candidates]))  # Pickup constraint.
                                          and (not movable_constrained or
                                               r in constants.VAL_ACTION_OBJECTS["Toggleable"] or
                                               np.any([m in constants.VAL_RECEPTACLE_OBJECTS[r]
                                                       for m in _movable_candidates]))  # Movable constraint.
                                          and (not scene_constrained or
                                               np.any([s in obj_to_scene_ids[constants.OBJ_PARENTS[r]]
                                                       for s in _scene_candidates]))  # Scene constraint
                                          ]
                # Constraints on scene.
                _scene_candidates = [s for s in _scene_candidates if
                                     np.any([s in scenes_for_goal[g]
                                             for g in _goal_candidates])  # Goal constraint.
                                     and (not pickup_constrained or
                                          np.any([obj_to_scene_ids[constants.OBJ_PARENTS[p]]
                                                  for p in _pickup_candidates]))  # Pickup constraint.
                                     and (not movable_constrained or
                                          np.any([obj_to_scene_ids[constants.OBJ_PARENTS[m]]
                                                  for m in _movable_candidates]))  # Movable constraint.
                                     and (not receptacle_constrained or
                                          np.any([obj_to_scene_ids[constants.OBJ_PARENTS[r]]
                                                  for r in _receptacle_candidates]))  # Receptacle constraint.
                                     ]
                if assignment_lens == (len(_goal_candidates), len(_pickup_candidates), len(_movable_candidates),
                                       len(_receptacle_candidates), len(_scene_candidates)):
                    propagation_finished = True

            candidate_lens = {"goal": len(_goal_candidates), "pickup": len(_pickup_candidates),
                              "movable": len(_movable_candidates), "receptacle": len(_receptacle_candidates),
                              "scene": len(_scene_candidates)}
            if candidate_lens["goal"] == 0:
                # print("Goal over-constrained; skipping")
                continue
            if np.all([0 in [candidate_lens[v] for v in goal_to_required_variables[g]] for g in _goal_candidates]):
                continue

            # Ensure some combination of the remaining constraints is not in failures and is not already populated
            # by the target number of repeats.
            failure_ensured = True
            full_ensured = True
            for g in _goal_candidates:
                pickup_iter = _pickup_candidates if "pickup" in goal_to_required_variables[g] else ["None"]
                for p in pickup_iter:
                    movable_iter = _movable_candidates if "movable" in goal_to_required_variables[g] else ["None"]
                    for m in movable_iter:
                        receptacle_iter = _receptacle_candidates if "receptacle" in goal_to_required_variables[g] \
                            else ["None"]
                        for r in receptacle_iter:
                            scene_iter = _scene_candidates if "scene" in goal_to_required_variables[g] else ["None"]
                            for s in scene_iter:
                                if (g, p, m, r, s) not in fail_traj:
                                    failure_ensured = False
                                if (g, p, m, r, s) not in full_traj:
                                    full_ensured = False
                                if not failure_ensured and not full_ensured:
                                    break
                            if not failure_ensured and not full_ensured:
                                break
                        if not failure_ensured and not full_ensured:
                            break
                    if not failure_ensured and not full_ensured:
                        break
                if not failure_ensured and not full_ensured:
                    break
            if failure_ensured:
                continue
            if full_ensured:
                continue

            if candidate_lens["goal"] > 1 or np.any([np.any([candidate_lens[v] > 1
                                                             for v in goal_to_required_variables[g]])
                                                     for g in _goal_candidates]):
                task_sampler = sample_task_params(succ_traj, full_traj, fail_traj,
                                                  _goal_candidates, _pickup_candidates, _movable_candidates,
                                                  _receptacle_candidates, _scene_candidates)
                sampled_task = next(task_sampler)
                if sampled_task is None:
                    continue
            else:
                g = _goal_candidates[0]
                p = _pickup_candidates[0] if "pickup" in goal_to_required_variables[g] else "None"
                m = _movable_candidates[0] if "movable" in goal_to_required_variables[g] else "None"
                r = _receptacle_candidates[0] if "receptacle" in goal_to_required_variables[g] else "None"
                s = _scene_candidates[0] if "scene" in goal_to_required_variables[g] else "None"
                sampled_task = (g, p, m, r, int(s))

            yield sampled_task

    yield None  # Discovered that there are no valid assignments remaining.


def print_successes(succ_traj):
    print("###################################\n")
    print("Successes: ")
    print(succ_traj)
    print("\n##################################")


def main(args):
    # settings
    constants.DATA_SAVE_PATH = args.save_path
    print("Force Unsave Data: %s" % str(args.force_unsave))

    # Set up data structure to track dataset balance and use for selecting next parameters.
    # In actively gathering data, we will try to maximize entropy for each (e.g., uniform spread of goals,
    # uniform spread over patient objects, uniform recipient objects, and uniform scenes).
    succ_traj = pd.DataFrame(columns=["goal", "pickup", "movable", "receptacle", "scene"])

    # objects-to-scene and scene-to-objects database
    for scene_type, ids in constants.SCENE_TYPE.items():
        for id in ids:
            obj_json_file = os.path.join('layouts', 'FloorPlan%d-objects.json' % id)
            with open(obj_json_file, 'r') as of:
                scene_objs = json.load(of)

            id_str = str(id)
            scene_id_to_objs[id_str] = scene_objs
            for obj in scene_objs:
                if obj not in obj_to_scene_ids:
                    obj_to_scene_ids[obj] = set()
                obj_to_scene_ids[obj].add(id_str)

    # scene-goal database
    for g in constants.GOALS:
        for st in constants.GOALS_VALID[g]:
            scenes_for_goal[g].extend([str(s) for s in constants.SCENE_TYPE[st]])
        scenes_for_goal[g] = set(scenes_for_goal[g])

    # scene-type database
    for st in constants.SCENE_TYPE:
        for s in constants.SCENE_TYPE[st]:
            scene_to_type[str(s)] = st

    # pre-populate counts in this structure using saved trajectories path.
    succ_traj, full_traj = load_successes_from_disk(args.save_path, succ_traj, args.just_examine, args.repeats_per_cond)
    if args.just_examine:
        print_successes(succ_traj)
        return

    # pre-populate failed trajectories.
    fail_traj = load_fails_from_disk(args.save_path)
    print("Loaded %d known failed tuples" % len(fail_traj))

    # create env and agent
    env = ThorEnv()

    game_state = TaskGameStateFullKnowledge(env)
    agent = DeterministicPlannerAgent(thread_id=0, game_state=game_state)

    errors = {}  # map from error strings to counts, to be shown after every failure.
    goal_candidates = constants.GOALS[:]
    pickup_candidates = list(set().union(*[constants.VAL_RECEPTACLE_OBJECTS[obj]  # Union objects that can be placed.
                                           for obj in constants.VAL_RECEPTACLE_OBJECTS]))
    pickup_candidates = [p for p in pickup_candidates if constants.OBJ_PARENTS[p] in obj_to_scene_ids]
    movable_candidates = list(set(constants.MOVABLE_RECEPTACLES).intersection(obj_to_scene_ids.keys()))
    receptacle_candidates = [obj for obj in constants.VAL_RECEPTACLE_OBJECTS
                             if obj not in constants.MOVABLE_RECEPTACLES and obj in obj_to_scene_ids] + \
                            [obj for obj in constants.VAL_ACTION_OBJECTS["Toggleable"]
                             if obj in obj_to_scene_ids]

    # toaster isn't interesting in terms of producing linguistic diversity
    receptacle_candidates.remove('Toaster')
    receptacle_candidates.sort()

    scene_candidates = list(scene_id_to_objs.keys())

    n_until_load_successes = args.async_load_every_n_samples
    print_successes(succ_traj)
    task_sampler = sample_task_params(succ_traj, full_traj, fail_traj,
                                      goal_candidates, pickup_candidates, movable_candidates,
                                      receptacle_candidates, scene_candidates)

    # main generation loop
    # keeps trying out new task tuples as trajectories either fail or suceed
    while True:

        sampled_task = next(task_sampler)
        print(sampled_task)  # DEBUG
        if sampled_task is None:
            sys.exit("No valid tuples left to sample (all are known to fail or already have %d trajectories" %
                     args.repeats_per_cond)
        gtype, pickup_obj, movable_obj, receptacle_obj, sampled_scene = sampled_task
        print("sampled tuple: " + str((gtype, pickup_obj, movable_obj, receptacle_obj, sampled_scene)))

        tries_remaining = args.trials_before_fail
        # only try to get the number of trajectories left to make this tuple full.
        target_remaining = args.repeats_per_cond - len(succ_traj.loc[(succ_traj['goal'] == gtype) &
                                                                (succ_traj['pickup'] == pickup_obj) &
                                                                (succ_traj['movable'] == movable_obj) &
                                                                (succ_traj['receptacle'] == receptacle_obj) &
                                                                (succ_traj['scene'] == str(sampled_scene))])
        num_place_fails = 0  # count of errors related to placement failure for no valid positions.

        # continue until we're (out of tries + have never succeeded) or (have gathered the target number of instances)
        while tries_remaining > 0 and target_remaining > 0:

            # environment setup
            constants.pddl_goal_type = gtype
            print("PDDLGoalType: " + constants.pddl_goal_type)
            task_id = create_dirs(gtype, pickup_obj, movable_obj, receptacle_obj, sampled_scene)

            # setup data dictionary
            setup_data_dict()
            constants.data_dict['task_id'] = task_id
            constants.data_dict['task_type'] = constants.pddl_goal_type
            constants.data_dict['dataset_params']['video_frame_rate'] = constants.VIDEO_FRAME_RATE

            # plan & execute
            try:
                # Agent reset to new scene.
                constraint_objs = {'repeat': [(constants.OBJ_PARENTS[pickup_obj],  # Generate multiple parent objs.
                                               np.random.randint(2 if gtype == "pick_two_obj_and_place" else 1,
                                                                 constants.PICKUP_REPEAT_MAX + 1))],
                                   'sparse': [(receptacle_obj.replace('Basin', ''),
                                               num_place_fails * constants.RECEPTACLE_SPARSE_POINTS)]}
                if movable_obj != "None":
                    constraint_objs['repeat'].append((movable_obj,
                                                      np.random.randint(1, constants.PICKUP_REPEAT_MAX + 1)))
                for obj_type in scene_id_to_objs[str(sampled_scene)]:
                    if (obj_type in pickup_candidates and
                            obj_type != constants.OBJ_PARENTS[pickup_obj] and obj_type != movable_obj):
                        constraint_objs['repeat'].append((obj_type,
                                                          np.random.randint(1, constants.MAX_NUM_OF_OBJ_INSTANCES + 1)))
                if gtype in goal_to_invalid_receptacle:
                    constraint_objs['empty'] = [(r.replace('Basin', ''), num_place_fails * constants.RECEPTACLE_EMPTY_POINTS)
                                                for r in goal_to_invalid_receptacle[gtype]]
                constraint_objs['seton'] = []
                if gtype == 'look_at_obj_in_light':
                    constraint_objs['seton'].append((receptacle_obj, False))
                if num_place_fails > 0:
                    print("Failed %d placements in the past; increased free point constraints: " % num_place_fails
                          + str(constraint_objs))
                scene_info = {'scene_num': sampled_scene, 'random_seed': random.randint(0, 2 ** 32)}
                info = agent.reset(scene=scene_info,
                                   objs=constraint_objs)

                # Problem initialization with given constraints.
                task_objs = {'pickup': pickup_obj}
                if movable_obj != "None":
                    task_objs['mrecep'] = movable_obj
                if gtype == "look_at_obj_in_light":
                    task_objs['toggle'] = receptacle_obj
                else:
                    task_objs['receptacle'] = receptacle_obj
                agent.setup_problem({'info': info}, scene=scene_info, objs=task_objs)

                # Now that objects are in their initial places, record them.
                object_poses = [{'objectName': obj['name'].split('(Clone)')[0],
                                 'position': obj['position'],
                                 'rotation': obj['rotation']}
                                for obj in env.last_event.metadata['objects'] if obj['pickupable']]
                dirty_and_empty = gtype == 'pick_clean_then_place_in_recep'
                object_toggles = [{'objectType': o, 'isOn': v}
                                  for o, v in constraint_objs['seton']]
                constants.data_dict['scene']['object_poses'] = object_poses
                constants.data_dict['scene']['dirty_and_empty'] = dirty_and_empty
                constants.data_dict['scene']['object_toggles'] = object_toggles

                # Pre-restore the scene to cause objects to "jitter" like they will when the episode is replayed
                # based on stored object and toggle info. This should put objects closer to the final positions they'll
                # be inlay at inference time (e.g., mugs fallen and broken, knives fallen over, etc.).
                print("Performing reset via thor_env API")
                env.reset(sampled_scene)
                print("Performing restore via thor_env API")
                env.restore_scene(object_poses, object_toggles, dirty_and_empty)
                event = env.step(dict(constants.data_dict['scene']['init_action']))

                terminal = False
                while not terminal and agent.current_frame_count <= constants.MAX_EPISODE_LENGTH:
                    action_dict = agent.get_action(None)
                    agent.step(action_dict)
                    reward, terminal = agent.get_reward()

                dump_data_dict()
                save_video()

            except Exception as e:
                import traceback
                traceback.print_exc()
                print("Error: " + repr(e))
                print("Invalid Task: skipping...")
                if args.debug:
                    print(traceback.format_exc())

                deleted = delete_save(args.in_parallel)
                if not deleted:  # another thread is filling this task successfully, so leave it alone.
                    target_remaining = 0  # stop trying to do this task.
                else:
                    if str(e) == "API Action Failed: No valid positions to place object found":
                        # Try increasing the space available on sparse and empty flagged objects.
                        num_place_fails += 1
                        tries_remaining -= 1
                    else:  # generic error
                        tries_remaining -= 1

                estr = str(e)
                if len(estr) > 120:
                    estr = estr[:120]
                if estr not in errors:
                    errors[estr] = 0
                errors[estr] += 1
                print("%%%%%%%%%%")
                es = sum([errors[er] for er in errors])
                print("\terrors (%d):" % es)
                for er, v in sorted(errors.items(), key=lambda kv: kv[1], reverse=True):
                    if v / es < 0.01:  # stop showing below 1% of errors.
                        break
                    print("\t(%.2f) (%d)\t%s" % (v / es, v, er))
                print("%%%%%%%%%%")

                continue

            if args.force_unsave:
                delete_save(args.in_parallel)

            # add to save structure.
            succ_traj = succ_traj.append({
                "goal": gtype,
                "movable": movable_obj,
                "pickup": pickup_obj,
                "receptacle": receptacle_obj,
                "scene": str(sampled_scene)}, ignore_index=True)
            target_remaining -= 1
            tries_remaining += args.trials_before_fail  # on success, add more tries for future successes

        # if this combination resulted in a certain number of failures with no successes, flag it as not possible.
        if tries_remaining == 0 and target_remaining == args.repeats_per_cond:
            new_fails = [(gtype, pickup_obj, movable_obj, receptacle_obj, str(sampled_scene))]
            fail_traj = load_fails_from_disk(args.save_path, to_write=new_fails)
            print("%%%%%%%%%%")
            print("failures (%d)" % len(fail_traj))
            # print("\t" + "\n\t".join([str(ft) for ft in fail_traj]))
            print("%%%%%%%%%%")

        # if this combination gave us the repeats we wanted, note it as filled.
        if target_remaining == 0:
            full_traj.add((gtype, pickup_obj, movable_obj, receptacle_obj, sampled_scene))

        # if we're sharing with other processes, reload successes from disk to update local copy with others' additions.
        if args.in_parallel:
            if n_until_load_successes > 0:
                n_until_load_successes -= 1
            else:
                print("Reloading trajectories from disk because of parallel processes...")
                succ_traj = pd.DataFrame(columns=succ_traj.columns)  # Drop all rows.
                succ_traj, full_traj = load_successes_from_disk(args.save_path, succ_traj, False, args.repeats_per_cond)
                print("... Loaded %d trajectories" % len(succ_traj.index))
                n_until_load_successes = args.async_load_every_n_samples
                print_successes(succ_traj)
                task_sampler = sample_task_params(succ_traj, full_traj, fail_traj,
                                                  goal_candidates, pickup_candidates, movable_candidates,
                                                  receptacle_candidates, scene_candidates)
                print("... Created fresh instance of sample_task_params generator")


def create_dirs(gtype, pickup_obj, movable_obj, receptacle_obj, scene_num):
    task_id = 'trial_T' + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    save_name = '%s-%s-%s-%s-%d' % (gtype, pickup_obj, movable_obj, receptacle_obj, scene_num) + '/' + task_id

    constants.save_path = os.path.join(constants.DATA_SAVE_PATH, save_name, RAW_IMAGES_FOLDER)
    if not os.path.exists(constants.save_path):
        os.makedirs(constants.save_path)

    print("Saving images to: " + constants.save_path)
    return task_id


def save_video():
    images_path = constants.save_path + '*.png'
    video_path = os.path.join(constants.save_path.replace(RAW_IMAGES_FOLDER, ''), 'video.mp4')
    video_saver.save(images_path, video_path)


def setup_data_dict():
    constants.data_dict = OrderedDict()
    constants.data_dict['task_id'] = ""
    constants.data_dict['task_type'] = ""
    constants.data_dict['scene'] = {'floor_plan': "", 'random_seed': -1, 'scene_num': -1, 'init_action': [],
                                    'object_poses': [], 'dirty_and_empty': None, 'object_toggles': []}
    constants.data_dict['plan'] = {'high_pddl': [], 'low_actions': []}
    constants.data_dict['images'] = []
    constants.data_dict['template'] = {'task_desc': "", 'high_descs': []}
    constants.data_dict['pddl_params'] = {'object_target': -1, 'object_sliced': -1,
                                          'parent_target': -1, 'toggle_target': -1,
                                          'mrecep_target': -1}
    constants.data_dict['dataset_params'] = {'video_frame_rate': -1}
    constants.data_dict['pddl_state'] = []


def dump_data_dict():
    data_save_path = constants.save_path.replace(RAW_IMAGES_FOLDER, '')
    with open(os.path.join(data_save_path, DATA_JSON_FILENAME), 'w') as fp:
        json.dump(constants.data_dict, fp, sort_keys=True, indent=4)


def delete_save(in_parallel):
    save_folder = constants.save_path.replace(RAW_IMAGES_FOLDER, '')
    if os.path.exists(save_folder):
        try:
            shutil.rmtree(save_folder)
        except OSError as e:
            if in_parallel:  # another thread succeeded at this task while this one failed.
                return False
            else:
                raise e  # if we're not running in parallel, this is an actual.
    return True


def parallel_main(args):
    procs = [mp.Process(target=main, args=(args,)) for _ in range(args.num_threads)]
    try:
        for proc in procs:
            proc.start()
            time.sleep(0.1)
    finally:
        for proc in procs:
            proc.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # settings
    parser.add_argument('--force_unsave', action='store_true', help="don't save any data (for debugging purposes)")
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--save_path', type=str, default="dataset/new_trajectories", help="where to save the generated data")
    parser.add_argument('--x_display', type=str, required=False, default=constants.X_DISPLAY, help="x_display id")
    parser.add_argument("--just_examine", action='store_true', help="just examine what data is gathered; don't gather more")
    parser.add_argument("--in_parallel", action='store_true', help="this collection will run in parallel with others, so load from disk on every new sample")
    parser.add_argument("-n", "--num_threads", type=int, default=0, help="number of processes for parallel mode")
    parser.add_argument('--json_file', type=str, default="", help="path to json file with trajectory dump")

    # params
    parser.add_argument("--repeats_per_cond", type=int, default=3)
    parser.add_argument("--trials_before_fail", type=int, default=5)
    parser.add_argument("--async_load_every_n_samples", type=int, default=10)

    parse_args = parser.parse_args()

    if parse_args.in_parallel and parse_args.num_threads > 1:
        parallel_main(parse_args)
    else:
        main(parse_args)
