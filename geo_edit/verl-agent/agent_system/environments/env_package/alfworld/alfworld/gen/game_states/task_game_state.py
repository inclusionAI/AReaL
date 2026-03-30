import os
import random

import alfworld.gen.constants as constants
import alfworld.gen.goal_library as glib
from alfworld.gen.game_states.planned_game_state import PlannedGameState
from alfworld.gen.utils import game_util


class TaskGameState(PlannedGameState):
    action_space = [
        {'action': 'Explore'},
        {'action': 'Scan'},
        {'action': 'Plan'},
        {'action': 'End'},
    ]

    def __init__(self, env, seed=None, action_space=None):
        if action_space is None:
            action_space = TaskGameState.action_space
        super(TaskGameState, self).__init__(env, seed, action_space,
                                            'put_task', 'planner/domains/PutTaskExtended_domain.pddl')
        self.task_target = None
        self.success = False

    def get_task_str(self):
        return game_util.get_task_str(self.object_target, self.parent_target, self.toggle_target, self.mrecep_target)

    def get_goal_pddl(self):
        goal_type = constants.pddl_goal_type
        if constants.data_dict['pddl_params']['object_sliced']:
            goal_type += "_slice"
        goal_str = glib.gdict[goal_type]['pddl']
        goal_str = goal_str.format(obj=constants.OBJECTS[self.object_target]
                                    if self.object_target is not None else "",
                                   recep=constants.OBJECTS[self.parent_target]
                                    if self.parent_target is not None else "",
                                   toggle=constants.OBJECTS[self.toggle_target]
                                    if self.toggle_target is not None else "",
                                   mrecep=constants.OBJECTS[self.mrecep_target]
                                    if self.mrecep_target is not None else "")
        return goal_str

    def get_current_plan(self, force_update=False):
        plan = super(TaskGameState, self).get_current_plan(force_update)
        self.plan = plan
        return self.plan

    def get_random_task_vals(self, scene=None):
        dataset_type, max_num_repeats, remove_prob, task_row = \
            self.initialize_random_scene(scene=scene)
        return (dataset_type, task_row), max_num_repeats, remove_prob

    def initialize_random_scene(self, scene=None):
        if self.problem_id is not None:
            # clean up old problem
            if not constants.DEBUG and os.path.exists('%s/planner/generated_problems/problem_%s.pddl' % (
                    self.dname, self.problem_id)):
                os.remove('%s/planner/generated_problems/problem_%s.pddl' % (self.dname, self.problem_id))

        # Do equal number of each task type in train.
        dataset_type = 'train'
        task_row = self.local_random.randint(0, 1e5)

        self.problem_id = str(task_row)

        if scene is None:
            # setup random scene with random object initialization
            self.scene_num = random.choice(constants.TRAIN_SCENE_NUMBERS)
            self.scene_seed = self.local_random.randint(0, 2 ** 32)
        else:
            # load a pre-specified sscene
            self.scene_num = scene['scene_num']
            self.scene_seed = scene['random_seed']

        self.object_target = -1
        self.parent_target = -1

        max_num_repeats = constants.MAX_NUM_OF_OBJ_INSTANCES
        remove_prob = 0.0
        return dataset_type, max_num_repeats, remove_prob, task_row

    def get_filter_crit(self, goal_type):

        # helper functions
        def is_obj_pickupable(x):
            return x['pickupable'] and x['objectType'] in constants.OBJECTS

        def is_receptacle(x):
            return x['receptacle'] and x['objectType'] in constants.OBJECTS

        def does_have_duplicates(x):
            num_instances = len(game_util.get_objects_of_type(x['objectType'], self.env.last_event.metadata))
            return num_instances > 1

        def does_have_atleast_n_duplicates(x, n):
            num_instances = len(game_util.get_objects_of_type(x['objectType'], self.env.last_event.metadata))
            return num_instances >= n

        def does_any_recep_type_have_obj_of_type(r, ot):
            all_recep_of_type = [o for o in self.env.last_event.metadata['objects'] if o['objectType'] == r['objectType']]
            valid_recep = []
            for recep in all_recep_of_type:
                receptacle_obj_ids = recep['receptacleObjectIds']
                if receptacle_obj_ids is not None:
                    objs_of_type = [obj_id for obj_id in receptacle_obj_ids if ot in obj_id]
                    if len(objs_of_type) > 0:
                        valid_recep.append(recep)
            return len(valid_recep) > 0

        def does_recep_have_enough_space(r):
            # TODO: how to check this with THOR 2.1.0?
            return True

        def is_recep_full(r):
            # TODO: how to check this with THOR 2.1.0?
            return False

        def can_recep_hold_obj(o, r):
            return o in constants.VAL_RECEPTACLE_OBJECTS[r['objectType']]

        def is_of_type(x, type):
            return x['objectType'] == type

        def is_movable_recep(o):
            return o['objectType'] in constants.MOVABLE_RECEPTACLES

        def is_obj_type_in_recep(x, recep):
            obj_types_in_recep = [o['objectType'] for o in self.env.last_event.metadata['objects']
                                  if (o['parentReceptacles'] is not None and
                                      any(recep in r for r in o['parentReceptacles']))]
            return x['objectType'] in obj_types_in_recep

        def is_obj_in_recep(o, r):
            return o['objectId'] in r['receptacleObjectIds']

        def is_obj_prop(x, prop):
            return x['objectType'] in constants.VAL_ACTION_OBJECTS[prop]

        def is_obj_off(x):
            return x['objectType'] in constants.VAL_ACTION_OBJECTS['Toggleable'] and \
                   x['toggleable'] \
                   and (not x['isToggled'])

        def is_obj_on(x):
            return x['objectType'] in constants.VAL_ACTION_OBJECTS['Toggleable'] and \
                   x['toggleable'] and \
                   x['isToggled']

        # specialized filters for each specific task
        if goal_type == "init":
            return lambda o: is_obj_pickupable(o), \
                   lambda r: is_receptacle(r) and \
                             (can_recep_hold_obj(self.rand_chosen_object_class, r)) and \
                             (not is_recep_full(r)) and \
                             (is_obj_in_recep(self.rand_chosen_object, r))
        elif goal_type == "place_all_obj_type_into_recep":
            return lambda o: does_have_duplicates(o), \
                   lambda r: does_recep_have_enough_space(r)
        elif goal_type == "pick_two_obj_and_place":
            return lambda o: does_have_atleast_n_duplicates(o, 2), \
                   lambda r: does_recep_have_enough_space(r) and \
                             (not does_any_recep_type_have_obj_of_type(r, self.rand_chosen_object_class))
        elif goal_type == "pick_clean_then_place_in_recep":
            return lambda o: (not is_obj_type_in_recep(o, "SinkBasin")) and \
                             (is_obj_prop(o, "Cleanable")), \
                   lambda r: (not is_of_type(r, "SinkBasin"))
        elif goal_type == "pick_heat_then_place_in_recep":
            return lambda o: (not is_obj_type_in_recep(o, "Microwave")) and \
                             (is_obj_prop(o, "Heatable")), \
                   lambda r: (not is_of_type(r, "Microwave"))
        elif goal_type == "pick_cool_then_place_in_recep":
            return lambda o: (not is_obj_type_in_recep(o, "Fridge")) and \
                             (is_obj_prop(o, "Coolable")), \
                   lambda r: (not is_of_type(r, "Fridge"))
        elif goal_type == "look_at_obj_in_light":
            return lambda o: is_obj_pickupable(o), \
                   lambda r: r
        elif goal_type == "pick_and_place_with_movable_recep":
            return lambda o: is_obj_pickupable(o) and \
                             (not is_movable_recep(o)), \
                   lambda r: is_receptacle(r) and \
                             (not is_movable_recep(r)) and \
                             (not does_any_recep_type_have_obj_of_type(r, self.rand_chosen_val_moveable_recep_class))
        elif goal_type == "pick_heat_and_place_with_movable_recep":
            return lambda o: is_obj_pickupable(o) and \
                             (is_obj_prop(o, "Heatable")) and \
                             (not is_movable_recep(o)), \
                   lambda r: is_receptacle(r) and \
                             (not is_movable_recep(r)) and \
                             (not is_of_type(r, "Microwave")) and \
                             (not does_any_recep_type_have_obj_of_type(r, self.rand_chosen_val_moveable_recep_class))
        else:
            return lambda o: is_obj_pickupable(o), \
                   lambda r: is_receptacle(r)


    def setup_problem(self, seed=None, info=None, scene=None, objs=None):
        '''
        setup goal with sampled objects or with the objects specified
        note: must be used with `initialize_random_scene`
        '''
        self.terminal = False
        print('setup random goal ----------------------------------------------')
        print('seed', seed)
        print('info', info)
        print('--------------------------------------------------------------------\n\n')
        super(TaskGameState, self).setup_problem(seed)
        info, max_num_repeats, remove_prob = self.get_random_task_vals(scene=scene)
        dataset_type, task_row = info

        print('Type:', dataset_type, 'Row: ', task_row, 'Scene', self.scene_name, 'seed', self.scene_seed)
        perform_sanity_check = True

        # pickupable, receptacle, toggleable candidates
        pickupable_objects = [o for o in self.env.last_event.metadata['objects']
                              if (o['pickupable'] and o['objectType'] in constants.OBJECTS)]
        receptacle_objects = [o for o in self.env.last_event.metadata['objects']
                              if (o['receptacle'] and o['objectType'] in constants.RECEPTACLES
                                  and o['objectType'] not in constants.MOVABLE_RECEPTACLES_SET)]
        toggle_objects = [o for o in self.env.last_event.metadata['objects']
                          if o['toggleable'] and not o['isToggled']  # must be one we can turn on.
                          and o['objectType'] in constants.VAL_ACTION_OBJECTS["Toggleable"]]

        if len(pickupable_objects) == 0:
            print("Task Failed - %s" % constants.pddl_goal_type)
            raise Exception("No pickupable objects in the scene")

        # filter based on crit
        obj_crit, _ = self.get_filter_crit(constants.pddl_goal_type)
        pickupable_objects = list(filter(obj_crit, pickupable_objects))
        if len(pickupable_objects) == 0:
            print("Task Failed - %s" % constants.pddl_goal_type)
            raise Exception("No pickupable objects related to the goal '%s'" % constants.pddl_goal_type)

        # choose a pickupable object
        if constants.FORCED_SAMPLING or objs is None:
            self.rand_chosen_object = random.choice(pickupable_objects)
        else:
            chosen_class_available = [obj for obj in pickupable_objects
                                      if obj['objectType'] == constants.OBJ_PARENTS[objs['pickup']]]
            if len(chosen_class_available) == 0:
                print("Task Failed - %s" % constants.pddl_goal_type)
                raise Exception("Couldn't find a valid parent '%s' for pickup object with class '%s'" %
                                 (constants.OBJ_PARENTS[objs['pickup']], objs['pickup']))
            self.rand_chosen_object = random.choice(chosen_class_available)
        self.rand_chosen_object_class = self.rand_chosen_object['objectType']
        self.object_target = constants.OBJECTS.index(self.rand_chosen_object_class)

        # for now, any obj differing from its parent is obtained via slicing, but that may change in the future.
        if constants.OBJ_PARENTS[objs['pickup']] != objs['pickup']:
            requires_slicing = True
        else:
            requires_slicing = False

        # choose a movable receptacle
        if "movable_recep" in constants.pddl_goal_type:
            val_movable_receps = [o for o in self.env.last_event.metadata['objects']
                                  if (o['objectType'] in constants.MOVABLE_RECEPTACLES_SET) and
                                     (self.rand_chosen_object_class in constants.VAL_RECEPTACLE_OBJECTS[o['objectType']])]

            if len(val_movable_receps) == 0:
                print("Task Failed - %s" % constants.pddl_goal_type)
                raise Exception("Couldn't find a valid moveable receptacle for the chosen object class")

            if objs is not None:
                val_movable_receps = [o for o in val_movable_receps if o['objectType'] == objs['mrecep']]

            self.rand_chosen_val_moveable_recep_class = random.choice(val_movable_receps)['objectType']
            self.mrecep_target = constants.OBJECTS.index(self.rand_chosen_val_moveable_recep_class)
        else:
            self.mrecep_target = None

        # if slicing is required, make sure a knife is available in the scene
        if requires_slicing:
            knife_objs = [o for o in self.env.last_event.metadata['objects']
                          if ("Knife" in o['objectType'])]
            if len(knife_objs) == 0:
                print("Task Failed - %s" % constants.pddl_goal_type)
                raise Exception("Couldn't find knife in the scene to cut with")

        if constants.pddl_goal_type == "look_at_obj_in_light":
            # choose a toggleable object
            self.parent_target = None

            if constants.FORCED_SAMPLING or objs is None:
                rand_chosen_toggle_object = random.choice(toggle_objects)
            else:
                toggle_class_available = [obj for obj in toggle_objects
                                          if obj['objectType'] == objs['toggle']]
                if len(toggle_class_available) == 0:
                    print("Task Failed - %s" % constants.pddl_goal_type)
                    raise Exception("Couldn't find a valid toggle object of class '%s'" % objs['toggle'])
                rand_chosen_toggle_object = random.choice(toggle_class_available)

            rand_chosen_toggle_class = rand_chosen_toggle_object['objectType']
            self.toggle_target = constants.OBJECTS.index(rand_chosen_toggle_class)
        else:
            self.toggle_target = None

            # find all valid receptacles in which rand_chosen object or rand chosen moveable receptacle can be placed
            val_receptacle_objects_orig = [r for r in receptacle_objects if (self.rand_chosen_val_moveable_recep_class if self.mrecep_target != None else self.rand_chosen_object_class)
                                      in constants.VAL_RECEPTACLE_OBJECTS[r['objectType']]]
            _, recep_crit = self.get_filter_crit(constants.pddl_goal_type)
            val_receptacle_objects = list(filter(recep_crit, val_receptacle_objects_orig))

            if len(val_receptacle_objects) == 0:
                print("Task Failed - %s" % constants.pddl_goal_type)
                raise Exception("Couldn't find a valid receptacle object according to constraints specified")

            # choose a receptacle object
            if constants.FORCED_SAMPLING or objs is None:
                rand_chosen_receptacle_object = random.choice(val_receptacle_objects)
            else:
                receptacle_class_available = [obj for obj in val_receptacle_objects
                                              if obj['objectType'] == objs['receptacle']]
                if len(receptacle_class_available) == 0:
                    print("Task Failed - %s" % constants.pddl_goal_type)
                    raise Exception("Couldn't find a valid receptacle object of class '%s'" % objs['receptacle'])
                rand_chosen_receptacle_object = random.choice(receptacle_class_available)
            rand_chosen_receptacle_object_class = rand_chosen_receptacle_object['objectType']
            self.parent_target = constants.OBJECTS.index(rand_chosen_receptacle_object_class)

        # pddl_param dict
        constants.data_dict['pddl_params']['object_target'] = constants.OBJECTS[self.object_target] if self.object_target is not None else ""
        constants.data_dict['pddl_params']['object_sliced'] = requires_slicing
        constants.data_dict['pddl_params']['parent_target'] = constants.OBJECTS[self.parent_target] if self.parent_target is not None else ""
        constants.data_dict['pddl_params']['toggle_target'] = constants.OBJECTS[self.toggle_target] if self.toggle_target is not None else ""
        constants.data_dict['pddl_params']['mrecep_target'] = constants.OBJECTS[self.mrecep_target] if self.mrecep_target is not None else ""
        self.task_target = (self.object_target, self.parent_target, self.toggle_target, self.mrecep_target)

        if self.parent_target is None:
            validity_check = True
        else:
            objs = game_util.get_objects_of_type(constants.OBJECTS[self.object_target], self.env.last_event.metadata)
            available_receptacles = game_util.get_objects_of_type(
                constants.OBJECTS[self.parent_target], self.env.last_event.metadata)
            validity_check = (len(objs) > 0 and
                              len([obj for obj in available_receptacles]))

        if perform_sanity_check:
            try:
                assert validity_check, 'Task does not seem valid according to scene metadata'
            except AssertionError:
                raise Exception(str(('Row: ', task_row, 'Scene', self.scene_name,
                                      'seed', self.scene_seed)))

        templated_task_desc = self.get_task_str()
        print('problem id', self.problem_id)
        print('Task:', templated_task_desc)
        constants.data_dict['template']['task_desc'] = templated_task_desc

    def get_setup_info(self, info=None, scene=None):
        return self.get_random_task_vals(scene=scene)

    def reset(self, seed=None, info=None, scene=None, objs=None):
        info = super(TaskGameState, self).reset(seed, info, scene=scene, objs=objs)
        self.receptacle_to_point = None
        self.task_target = None
        self.success = False
        return info

    def step(self, action_or_ind):
        action, _ = self.get_action(action_or_ind)
        super(TaskGameState, self).step(action_or_ind)
        self.success = True

    def get_success(self):
        return self.success
