import copy
import json
import os
from abc import ABC

import alfworld.gen.constants as constants
from alfworld.gen.game_states.game_state_base import GameStateBase
from alfworld.gen.planner import ff_planner_handler
from alfworld.gen.utils import game_util
from alfworld.gen.utils import py_util


class PlannedGameState(GameStateBase, ABC):
    @staticmethod
    def fix_pddl_str_chars(input_str):
        return py_util.multireplace(input_str,
                                    {'-': '_minus_',
                                     '#': '-',
                                     '|': '_bar_',
                                     '+': '_plus_',
                                     '.': '_dot_',
                                     ',': '_comma_'})

    action_space = [
        {'action': 'Explore'},
        {'action': 'Scan'},
        {'action': 'Plan'},
        {'action': 'End'},
    ]

    action_to_ind = {action['action']: ii for ii, action in enumerate(action_space)}

    def __init__(self, env, seed=None, action_space=None, domain=None, domain_path=None):
        super(PlannedGameState, self).__init__(env, seed, action_space)
        self.planner = ff_planner_handler.PlanParser(domain_path)
        self.domain = domain
        self.terminal = False
        self.problem_id = -1
        self.in_receptacle_ids = {}
        self.was_in_receptacle_ids = {}
        self.need_plan_update = True
        self.pddl_start = None
        self.pddl_init = None
        self.pddl_goal = None
        self.scene_seed = None
        self.object_target = None
        self.parent_target = None
        self.receptacle_to_point = {}
        self.point_to_receptacle = {}
        self.object_to_point = {}
        self.point_to_object = {}
        self.plan = None
        self.next_action = None
        self.failed_plan_action = False
        self.placed_items = set()
        self.openable_object_to_point = None

    def get_goal_pddl(self):
        raise NotImplementedError

    def state_to_pddl(self):
        object_dict = game_util.get_object_dict(self.env.last_event.metadata)
        if self.pddl_start is None:
            # Not previously written to file
            self.planner.problem_id = self.problem_id
            receptacle_types = copy.deepcopy(constants.RECEPTACLES) - set(constants.MOVABLE_RECEPTACLES)
            objects = copy.deepcopy(constants.OBJECTS_SET) - receptacle_types
            object_str = '\n        '.join([obj + ' # object' for obj in objects])

            self.knife_obj = {'ButterKnife', 'Knife'} if constants.data_dict['pddl_params']['object_sliced'] else {}

            otype_str = '\n        '.join([obj + 'Type # otype' for obj in objects])
            rtype_str = '\n        '.join([obj + 'Type # rtype' for obj in receptacle_types])

            self.pddl_goal = self.get_goal_pddl()

            self.pddl_start = '''
(define (problem plan_%s)
    (:domain %s)
    (:metric minimize (totalCost))
    (:objects
        agent1 # agent
        %s
        %s
        %s
''' % (
                self.problem_id,
                self.domain,

                object_str,
                otype_str,
                rtype_str,
                )

            self.pddl_init = '''
    (:init
        (= (totalCost) 0)
'''

            self.pddl_start = PlannedGameState.fix_pddl_str_chars(self.pddl_start)
            self.pddl_init = PlannedGameState.fix_pddl_str_chars(self.pddl_init)
            self.pddl_goal = PlannedGameState.fix_pddl_str_chars(self.pddl_goal)

        # pddl_mid section
        agent_location = 'loc|%d|%d|%d|%d' % (self.pose[0], self.pose[1], self.pose[2], self.pose[3])

        agent_location_str = '\n        (atLocation agent1 %s)' % agent_location
        opened_receptacle_str = '\n        '.join(['(opened %s)' % obj
                                                   for obj in self.currently_opened_object_ids])

        movable_recep_cls_with_knife = []
        in_receptacle_strs = []
        was_in_receptacle_strs = []
        for key, val in self.in_receptacle_ids.items():
            if len(val) == 0:
                continue
            key_cls = object_dict[key]['objectType']
            if key_cls in constants.MOVABLE_RECEPTACLES_SET:
                recep_str = 'inReceptacleObject'
            else:
                recep_str = 'inReceptacle'
            for vv in val:
                vv_cls = object_dict[vv]['objectType']
                if (vv_cls == constants.OBJECTS[self.object_target] or
                        (self.mrecep_target is not None and vv_cls == constants.OBJECTS[self.mrecep_target]) or
                        (len(self.knife_obj) > 0 and vv_cls in self.knife_obj)):

                    # if knife is inside a movable receptacle, make sure to add it to the object list
                    if recep_str == 'inReceptacleObject':
                        movable_recep_cls_with_knife.append(key_cls)

                    in_receptacle_strs.append('(%s %s %s)' % (
                        recep_str,
                        vv,
                        key)
                    )
                if key_cls not in constants.MOVABLE_RECEPTACLES_SET and vv_cls == constants.OBJECTS[self.object_target]:
                    was_in_receptacle_strs.append('(wasInReceptacle  %s %s)' % (vv, key))

        in_receptacle_str = '\n        '.join(in_receptacle_strs)
        was_in_receptacle_str = '\n        '.join(was_in_receptacle_strs)

        # Note which openable receptacles we can safely open (precomputed).
        openable_objects = self.openable_object_to_point.keys()

        metadata_objects = self.env.last_event.metadata['objects']
        receptacles = set({obj['objectId'] for obj in metadata_objects
                           if obj['objectType'] in constants.RECEPTACLES and obj['objectType'] not in constants.MOVABLE_RECEPTACLES_SET})

        objects = set({obj['objectId'] for obj in metadata_objects if
                       (obj['objectType'] == constants.OBJECTS[self.object_target]
                        or obj['objectType'] in constants.MOVABLE_RECEPTACLES_SET
                        or (self.mrecep_target is not None and obj['objectType'] == constants.OBJECTS[self.mrecep_target])
                        or ((self.toggle_target is not None and obj['objectType'] == constants.OBJECTS[self.toggle_target])
                            or ((len(self.knife_obj) > 0 and
                                 (obj['objectType'] in self.knife_obj or
                                  obj['objectType'] in movable_recep_cls_with_knife)))))})

        if len(self.inventory_ids) > 0:
            objects = objects | self.inventory_ids
        if len(self.placed_items) > 0:
            objects = objects | self.placed_items

        receptacle_str = '\n        '.join(sorted([receptacle + ' # receptacle'
                                            for receptacle in receptacles]))

        object_str = '\n        '.join(sorted([obj + ' # object' for obj in objects]))

        locations = set()
        for key, val in self.receptacle_to_point.items():
            key_cls = object_dict[key]['objectType']
            if key_cls not in constants.MOVABLE_RECEPTACLES_SET:
                locations.add(tuple(val.tolist()))
        for obj, loc in self.object_to_point.items():
            obj_cls = object_dict[obj]['objectType']
            if (obj_cls == constants.OBJECTS[self.object_target] or
                    (self.toggle_target is not None and obj_cls == constants.OBJECTS[self.toggle_target]) or
                    (len(self.knife_obj) > 0 and obj_cls in self.knife_obj) or
                    (obj_cls in constants.MOVABLE_RECEPTACLES_SET)):
                locations.add(tuple(loc))

        location_str = ('\n        '.join(['loc|%d|%d|%d|%d # location' % (*loc,)
                                          for loc in locations]) +
                        '\n        %s # location' % agent_location)

        if constants.PRUNE_UNREACHABLE_POINTS:
            # don't flag problematic receptacleTypes for the planner.
            receptacle_type_str = '\n        '.join(['(receptacleType %s %sType)' % (
                receptacle, object_dict[receptacle]['objectType']) for receptacle in receptacles
                                                     if object_dict[receptacle]['objectType'] not in constants.OPENABLE_CLASS_SET or
                                                        receptacle in openable_objects])
        else:
            receptacle_type_str = '\n        '.join(['(receptacleType %s %sType)' % (
                receptacle, object_dict[receptacle]['objectType']) for receptacle in receptacles])

        object_type_str = '\n        '.join(['(objectType %s %sType)' % (
            obj, object_dict[obj]['objectType']) for obj in objects])

        receptacle_objects_str = '\n        '.join(['(isReceptacleObject %s)' % (
            obj) for obj in objects if object_dict[obj]['objectType'] in constants.MOVABLE_RECEPTACLES_SET])

        if constants.PRUNE_UNREACHABLE_POINTS:
            openable_str = '\n        '.join(['(openable %s)' % receptacle for receptacle in receptacles
                                              if object_dict[receptacle]['objectType'] in constants.OPENABLE_CLASS_SET])
        else:
            # don't flag problematic open objects as openable for the planner.
            openable_str = '\n        '.join(['(openable %s)' % receptacle for receptacle in receptacles
                                              if object_dict[receptacle]['objectType'] in constants.OPENABLE_CLASS_SET and
                                              receptacle in openable_objects])

        dists = []
        dist_points = list(locations | {(self.pose[0], self.pose[1], self.pose[2], self.pose[3])})
        for dd, l_start in enumerate(dist_points[:-1]):
            for l_end in dist_points[dd + 1:]:
                actions, path = self.gt_graph.get_shortest_path_unweighted(l_start, l_end)
                # Should cost one more for the trouble of going there at all. Discourages waypoints.
                dist = len(actions) + 1
                dists.append('(= (distance loc|%d|%d|%d|%d loc|%d|%d|%d|%d) %d)' % (
                    l_start[0], l_start[1], l_start[2], l_start[3],
                    l_end[0], l_end[1], l_end[2], l_end[3], dist))
                dists.append('(= (distance loc|%d|%d|%d|%d loc|%d|%d|%d|%d) %d)' % (
                    l_end[0], l_end[1], l_end[2], l_end[3],
                    l_start[0], l_start[1], l_start[2], l_start[3], dist))
        location_distance_str = '\n        '.join(dists)


        # clean objects
        cleanable_str = '\n        '.join(['(cleanable %s)' % obj for obj in objects
                                          if object_dict[obj]['objectType'] in constants.VAL_ACTION_OBJECTS['Cleanable']])

        is_clean_str = '\n        '.join((['(isClean %s)' % obj
                                           for obj in self.cleaned_object_ids if object_dict[obj]['objectType'] == constants.OBJECTS[self.object_target]]))

        # heat objects
        heatable_str = '\n        '.join(['(heatable %s)' % obj for obj in objects
                                          if object_dict[obj]['objectType'] in constants.VAL_ACTION_OBJECTS['Heatable']])

        is_hot_str = '\n        '.join((['(isHot %s)' % obj
                                         for obj in self.hot_object_ids if object_dict[obj]['objectType'] == constants.OBJECTS[self.object_target]]))

        # cool objects
        coolable_str = '\n        '.join(['(coolable %s)' % obj for obj in objects
                                          if object_dict[obj]['objectType'] in constants.VAL_ACTION_OBJECTS['Coolable']])

        # toggleable objects
        toggleable_str = '\n        '.join(['(toggleable %s)' % obj
                                            for obj in self.toggleable_object_ids
                                            if (self.toggle_target is not None
                                                and object_dict[obj]['objectType'] == constants.OBJECTS[self.toggle_target])])

        is_on_str = '\n        '.join(['(isOn %s)' % obj
                                       for obj in self.on_object_ids
                                       if (self.toggle_target is not None
                                           and object_dict[obj]['objectType'] == constants.OBJECTS[self.toggle_target])])

        # sliceable objects
        sliceable_str = '\n        '.join(['(sliceable %s)' % obj for obj in objects
                                          if (object_dict[obj]['objectType'] in constants.VAL_ACTION_OBJECTS['Sliceable'])])

        # sliced objects
        # TODO cleanup: sliced_object_ids is never added to. Does that matter?
        is_sliced_str = '\n        '.join((['(isSliced %s)' % obj
                                            for obj in self.sliced_object_ids
                                            if object_dict[obj]['objectType'] == constants.OBJECTS[self.object_target]]))

        # look for objects that are already cool
        for (key, val) in self.was_in_receptacle_ids.items():
            if 'Fridge' in key:
                for vv in val:
                    self.cool_object_ids.add(vv)

        is_cool_str = '\n        '.join((['(isCool %s)' % obj
                                          for obj in self.cool_object_ids if object_dict[obj]['objectType'] == constants.OBJECTS[self.object_target]]))

        # Receptacle Objects
        recep_obj_str = '\n        '.join(['(isReceptacleObject %s)' % obj for obj in receptacles
                                          if (object_dict[obj]['objectType'] in constants.MOVABLE_RECEPTACLES_SET and
                                              (self.mrecep_target is not None and object_dict[obj]['objectType'] == constants.OBJECTS[self.mrecep_target]))])

        receptacle_nearest_point_strs = sorted(
            ['(receptacleAtLocation %s loc|%d|%d|%d|%d)' % (obj_id, *point)
             for obj_id, point in self.receptacle_to_point.items()
             if (object_dict[obj_id]['objectType'] in constants.RECEPTACLES and
                 object_dict[obj_id]['objectType'] not in constants.MOVABLE_RECEPTACLES_SET)
             ])
        receptacle_at_location_str = '\n        '.join(receptacle_nearest_point_strs)
        extra_facts = self.get_extra_facts()

        pddl_mid_start = '''
        %s
        %s
        %s
        )
''' % (
            object_str,
            receptacle_str,
            location_str,
            )
        pddl_mid_init = '''
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        )
''' % (
            receptacle_type_str,
            object_type_str,
            receptacle_objects_str,
            openable_str,
            agent_location_str,
            opened_receptacle_str,
            cleanable_str,
            is_clean_str,
            heatable_str,
            coolable_str,
            is_hot_str,
            is_cool_str,
            toggleable_str,
            is_on_str,
            recep_obj_str,
            sliceable_str,
            is_sliced_str,
            in_receptacle_str,
            was_in_receptacle_str,
            location_distance_str,
            receptacle_at_location_str,
            extra_facts,
            )

        pddl_mid_start = PlannedGameState.fix_pddl_str_chars(pddl_mid_start)
        pddl_mid_init = PlannedGameState.fix_pddl_str_chars(pddl_mid_init)

        pddl_str = (self.pddl_start + '\n' +
                    pddl_mid_start + '\n' +
                    self.pddl_init + '\n' +
                    pddl_mid_init + '\n' +
                    self.pddl_goal)


        state_save_path = constants.save_path.replace('/raw_images', '/pddl_states')
        if not os.path.exists(state_save_path):
            os.makedirs(state_save_path)
        pddl_state_next_idx = len(constants.data_dict['pddl_state'])
        state_save_file = state_save_path + ('/problem_%s.pddl' % pddl_state_next_idx)

        with open(state_save_file, 'w') as fid:
            fid.write(pddl_str)
            fid.flush()
        constants.data_dict['pddl_state'].append('problem_%s.pddl' % pddl_state_next_idx)

        with open('%s/planner/generated_problems/problem_%s.pddl' % (self.dname, self.problem_id), 'w') as fid:
            fid.write(pddl_str)
            fid.flush()

        return pddl_str

    def get_extra_facts(self):
        raise NotImplementedError

    def get_teleport_action(self, action):
        nearest_point = tuple(map(int, action['location'].split('|')[1:]))
        next_action = {'action': 'TeleportFull',
                       'x': nearest_point[0] * constants.AGENT_STEP_SIZE,
                       'y': self.agent_height,
                       'z': nearest_point[1] * constants.AGENT_STEP_SIZE,
                       'rotateOnTeleport': True,
                       'rotation': nearest_point[2] * 90,
                       'horizon': nearest_point[3]
                       }
        return next_action

    def get_plan_action(self, action):
        if action['action'] == 'GotoLocation':
            action = self.get_teleport_action(action)
        return action

    def get_next_plan_action(self, force_update=False):
        if force_update:
            self.need_plan_update = True
        if self.need_plan_update:
            self.plan = self.get_current_plan()
            self.next_action = self.plan[0]
            if self.next_action['action'] == 'GotoLocation':
                self.next_action = self.get_teleport_action(self.next_action)
        if constants.DEBUG:
            print('\nnew plan\n' + '\n'.join(['%03d %s' % (ii, game_util.get_action_str(pl))
                                              for ii, pl in enumerate(self.plan)]), '\n')
        return self.next_action

    def get_current_plan(self, force_update=False):
        if self.failed_plan_action:
            self.plan = [{'action': 'End', 'value': 0}]
            self.failed_plan_action = False
            return self.plan
        if force_update:
            self.need_plan_update = True
        if self.need_plan_update:
            self.update_receptacle_nearest_points()
            self.plan = []
            # When there are no receptacles, there's nothing to plan.
            # Only happens if called too early (before room exploration).
            if len(self.receptacle_to_point) > 0:
                self.state_to_pddl()
                self.plan = self.planner.get_plan()
            self.need_plan_update = False
            if len(self.plan) == 0:
                # Problem is solved, plan is empty
                self.plan = [{'action': 'End', 'value': 1}]
        return self.plan

    def get_setup_info(self, info=None):
        raise NotImplementedError

    def reset(self, seed=None, info=None, scene=None, objs=None):
        if self.problem_id is not None:
            # clean up old problem
            if (not constants.EVAL and not constants.DEBUG and
                    os.path.exists('%s/planner/generated_problems/problem_%s.pddl' % (self.dname, self.problem_id))):
                os.remove('%s/planner/generated_problems/problem_%s.pddl' % (self.dname, self.problem_id))

        self.terminal = False
        self.problem_id = -1
        self.in_receptacle_ids = {}
        self.was_in_receptacle_ids = {}
        self.need_plan_update = True
        self.pddl_start = None
        self.pddl_init = None
        self.pddl_goal = None
        self.scene_seed = seed
        self.scene_num = None
        self.object_target = None
        self.parent_target = None
        self.receptacle_to_point = {}
        self.point_to_receptacle = {}
        self.object_to_point = {}
        self.point_to_object = {}
        self.plan = None
        self.failed_plan_action = False
        self.placed_items = set()

        if seed is not None:
            print('set seed in planned_game_state', seed)
            self.local_random.seed(seed)

        if not os.path.exists('%s/planner/generated_problems' % self.dname):
            os.makedirs('%s/planner/generated_problems' % self.dname)

        info, max_num_repeats, remove_prob = self.get_setup_info(info)
        super(PlannedGameState, self).reset(self.scene_num, False, self.scene_seed, max_num_repeats, remove_prob,
                                            scene=scene, objs=objs)
        self.gt_graph.clear()

        points_source = 'layouts/%s-openable.json' % self.scene_name
        with open(points_source, 'r') as f:
            openable_object_to_point = json.load(f)
        self.openable_object_to_point = openable_object_to_point

        return info

    def should_keep_door_open(self):
        next_plan_action_idx = len(constants.data_dict['plan']['high_pddl'])
        if next_plan_action_idx < len(self.plan):
            next_action = self.plan[next_plan_action_idx]
            return not next_action['action'] in {'GotoLocation', 'End'}
        else:
            return False

    def close_recep(self, recep):
        if recep['openable'] and recep['isOpen']:
            # check if door should be left open for next action
            if self.should_keep_door_open():
                return
            super(PlannedGameState, self).close_recep(recep)
