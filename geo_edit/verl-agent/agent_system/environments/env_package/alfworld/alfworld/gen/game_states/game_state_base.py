import copy
import glob
import random
import time

import cv2
import numpy as np

import alfworld.gen.constants as constants
from alfworld.gen.graph import graph_obj
from alfworld.gen.utils import game_util
from alfworld.gen.utils.py_util import SetWithGet
from alfworld.gen.utils.image_util import compress_mask


class GameStateBase(object):
    static_action_space = [
        {'action': 'MoveAhead', 'moveMagnitude': constants.AGENT_STEP_SIZE},
        {'action': 'MoveBack', 'moveMagnitude': constants.AGENT_STEP_SIZE},
        {'action': 'RotateLeft'},
        {'action': 'RotateRight'},
        {'action': 'LookUp'},
        {'action': 'LookDown'},
        {'action': 'OpenObject'},
        {'action': 'CloseObject'},
        {'action': 'Terminate'},
    ]

    def __init__(self, env, seed=None, action_space=None):
        self.env = env
        if seed is not None:
            self.local_random = random.Random(seed)
        else:
            self.local_random = random.Random()
        if action_space is None:
            self.action_space = GameStateBase.static_action_space
        else:
            self.action_space = action_space
        self.dname = constants.LOG_FILE
        self.total_frame_count = 0
        self.current_frame_count = 0
        self.timers = np.zeros((2, 2))
        self.board = None
        self.original_board = None
        self.currently_opened_object_ids = SetWithGet()
        self.inventory_ids = SetWithGet()

        self.cleaned_object_ids = SetWithGet()
        self.hot_object_ids = SetWithGet()
        self.cool_object_ids = SetWithGet()

        self.on_object_ids = set()
        self.toggleable_object_ids = set()
        self.sliced_object_ids = set()

        self.scene_num = None
        self.scene_name = None
        self.bounds = None
        self.gt_graph = None
        self.start_point = None
        self.event = None
        self.s_t = None
        self.s_t_orig = None
        self.s_t_depth = None
        self.pose = None
        self.agent_height = None
        self.camera_height = None

    def setup_problem(self, seed=None):
        # This should basically always be overloaded then super'd.
        if seed is not None:
            print('set seed in game_state_base setup', seed)
            self.local_random.seed(seed)
        self.process_frame()

    def reset(self, scene_num=None, use_gt=True, seed=None, max_num_repeats=constants.MAX_NUM_OF_OBJ_INSTANCES,
              remove_prob=None, scene=None, objs=None):
        # Reset should be called only when all information from a scene should be cleared.
        self.current_frame_count = 0
        self.timers = np.zeros((2, 2))
        self.board = None
        self.original_board = None
        self.currently_opened_object_ids = SetWithGet()

        self.cleaned_object_ids = SetWithGet()
        self.hot_object_ids = SetWithGet()
        self.cool_object_ids = SetWithGet()

        self.on_object_ids = set()
        self.toggleable_object_ids = set()
        self.sliced_object_ids = set()

        self.inventory_ids = SetWithGet()
        self.scene_name = None
        self.bounds = None
        self.start_point = None
        self.event = None
        self.s_t = None
        self.s_t_orig = None
        self.s_t_depth = None
        self.pose = None
        self.agent_height = None
        self.camera_height = None

        new_scene = (self.gt_graph is None or self.gt_graph.scene_id is None or self.gt_graph.scene_id == scene_num)
        self.scene_num = scene_num

        if self.scene_num is None:
            self.scene_num = self.local_random.choice(constants.SCENE_NUMBERS)
        self.scene_num = self.scene_num

        if scene is not None:
            self.scene_num = scene['scene_num']
            seed = scene['random_seed']

        self.scene_name = 'FloorPlan%d' % self.scene_num
        self.event = self.env.reset(self.scene_name)
        if max_num_repeats is None:
            self.event = self.env.random_initialize(seed)
        else:
            self.env.step(dict(
                action='Initialize',
                gridSize=constants.AGENT_STEP_SIZE / constants.RECORD_SMOOTHING_FACTOR,
                cameraY=constants.CAMERA_HEIGHT_OFFSET,
                renderImage=constants.RENDER_IMAGE,
                renderDepthImage=constants.RENDER_DEPTH_IMAGE,
                renderClassImage=constants.RENDER_CLASS_IMAGE,
                renderObjectImage=constants.RENDER_OBJECT_IMAGE,
                visibility_distance=constants.VISIBILITY_DISTANCE,
                makeAgentsVisible=False,
            ))

            free_per_receptacle = []
            if objs is not None:
                if 'sparse' in objs:
                    for o, c in objs['sparse']:
                        free_per_receptacle.append({'objectType': o, 'count': c})
                if 'empty' in objs:
                    for o, c in objs['empty']:
                        free_per_receptacle.append({'objectType': o, 'count': c})
            self.env.step(dict(action='InitialRandomSpawn', randomSeed=seed, forceVisible=False,
                               numRepeats=[{'objectType': o, 'count': c}
                                           for o, c in objs['repeat']]
                               if objs is not None and 'repeat' in objs else None,
                               minFreePerReceptacleType=free_per_receptacle if objs is not None else None
                               ))

            # if 'clean' action, make everything dirty and empty out fillable things
            if constants.pddl_goal_type == "pick_clean_then_place_in_recep":
                self.env.step(dict(action='SetStateOfAllObjects',
                                   StateChange="CanBeDirty",
                                   forceAction=True))

                self.env.step(dict(action='SetStateOfAllObjects',
                                   StateChange="CanBeFilled",
                                   forceAction=False))

            if objs is not None and ('seton' in objs and len(objs['seton']) > 0):
                self.env.step(dict(action='SetObjectToggles',
                                   objectToggles=[{'objectType': o, 'isOn': v}
                                                  for o, v in objs['seton']]))

        self.gt_graph = graph_obj.Graph(use_gt=True, construct_graph=True, scene_id=self.scene_num)

        if seed is not None:
            self.local_random.seed(seed)
            print('set seed in game_state_base reset', seed)

        self.bounds = np.array([self.gt_graph.xMin, self.gt_graph.yMin,
                                self.gt_graph.xMax - self.gt_graph.xMin + 1,
                                self.gt_graph.yMax - self.gt_graph.yMin + 1])

        self.agent_height = self.env.last_event.metadata['agent']['position']['y']
        self.camera_height = self.agent_height + constants.CAMERA_HEIGHT_OFFSET
        start_point = self.local_random.randint(0, self.gt_graph.points.shape[0] - 1)
        start_point = self.gt_graph.points[start_point, :].copy()
        self.start_point = (start_point[0], start_point[1], self.local_random.randint(0, 3))

        action = {'action': 'TeleportFull',
                  'x': self.start_point[0] * constants.AGENT_STEP_SIZE,
                  'y': self.agent_height,
                  'z': self.start_point[1] * constants.AGENT_STEP_SIZE,
                  'rotateOnTeleport': True,
                  'horizon': 30,
                  'rotation': self.start_point[2] * 90,
                  }
        self.event = self.env.step(action)

        constants.data_dict['scene']['scene_num'] = self.scene_num
        constants.data_dict['scene']['init_action'] = action
        constants.data_dict['scene']['floor_plan'] = self.scene_name
        constants.data_dict['scene']['random_seed'] = seed

        self.pose = game_util.get_pose(self.event)

        # Manually populate parentReceptacles data based on existing ReceptableObjectIds.
        objects = self.env.last_event.metadata['objects']
        for idx in range(len(objects)):
            if objects[idx]['parentReceptacles'] is None:
                objects[idx]['parentReceptacles'] = []
                for jdx in range(len(objects)):
                    if (objects[jdx]['receptacleObjectIds'] is not None
                            and objects[idx]['objectId'] in objects[jdx]['receptacleObjectIds']):
                        objects[idx]['parentReceptacles'].append(objects[jdx]['objectId'])

    def get_action(self, action_or_ind):
        if type(action_or_ind) == int or isinstance(action_or_ind, np.generic):
            action = copy.deepcopy(self.action_space[action_or_ind])
        else:
            action = action_or_ind
        return action, False

    def store_image_name(self, name):
        constants.data_dict['images'].append({"high_idx": game_util.get_last_hl_action_index(),
                                              "low_idx": game_util.get_last_ll_action_index(),
                                              "image_name": name})

    # def store_ll_action(self, action):
        constants.data_dict['plan']['low_actions'].append({"high_idx": len(constants.data_dict['plan']['high_pddl']) - 1,
                                                           "api_action": action,
                                                           "discrete_action": self.get_ll_discrete_action(action)})

    def get_ll_discrete_action(self, action):
        a_type = action['action']
        discrete_action = {'action': "", 'args': {}}

        if 'TeleportFull' in a_type:
            print("WARNING: Low-level actions should not use TeleportFull. Instead use LookUp or LookDown")
            look_str = "Look"
            if action['horizon'] == 0:
                look_str += "Ahead"
            elif action['horizon'] > 0:
                look_str += "Down_%d" % (action['horizon'])
            elif action['horizon'] < 0:
                look_str += "Up_%d" % (action['horizon'])

            discrete_action['action'] = look_str

        elif 'RotateLeft' in a_type:
            discrete_action['action'] = "RotateLeft_90" # API can only do 90 deg
        elif 'RotateRight' in a_type:
            discrete_action['action'] = "RotateRight_90" # API can only do 90 deg
        elif 'MoveAhead' in a_type:
            discrete_action['action'] = "MoveAhead_%d" % int(constants.AGENT_STEP_SIZE * 100)
        elif 'LookUp' in a_type:
            discrete_action['action'] = "LookUp_%d" % constants.AGENT_HORIZON_ADJ
        elif 'LookDown' in a_type:
            discrete_action['action'] = "LookDown_%d" % constants.AGENT_HORIZON_ADJ

        elif 'OpenObject' in a_type:
            bbox, point, mask = self.get_bbox_point_mask(action['objectId'])
            discrete_action['action'] = "OpenObject"
            discrete_action['args']['bbox'] = bbox
            discrete_action['args']['point'] = point
            discrete_action['args']['mask'] = mask
        elif 'CloseObject' in a_type:
            bbox, point, mask = self.get_bbox_point_mask(action['objectId'])
            discrete_action['action'] = "CloseObject"
            discrete_action['args']['bbox'] = bbox
            discrete_action['args']['point'] = point
            discrete_action['args']['mask'] = mask
        elif 'PickupObject' in a_type:
            bbox, point, mask = self.get_bbox_point_mask(action['objectId'])
            discrete_action['action'] = "PickupObject"
            discrete_action['args']['bbox'] = bbox
            discrete_action['args']['point'] = point
            discrete_action['args']['mask'] = mask
        elif 'PutObject' in a_type:
            bbox, point, mask = self.get_bbox_point_mask(action['receptacleObjectId'])
            discrete_action['action'] = "PutObject"
            discrete_action['args']['bbox'] = bbox
            discrete_action['args']['point'] = point
            discrete_action['args']['mask'] = mask
        elif 'ToggleObjectOn' in a_type:
            bbox, point, mask = self.get_bbox_point_mask(action['objectId'])
            discrete_action['action'] = "ToggleObjectOn"
            discrete_action['args']['bbox'] = bbox
            discrete_action['args']['point'] = point
            discrete_action['args']['mask'] = mask
        elif 'ToggleObjectOff' in a_type:
            bbox, point, mask = self.get_bbox_point_mask(action['objectId'])
            discrete_action['action'] = "ToggleObjectOff"
            discrete_action['args']['bbox'] = bbox
            discrete_action['args']['point'] = point
            discrete_action['args']['mask'] = mask
        elif 'SliceObject' in a_type:
            bbox, point, mask = self.get_bbox_point_mask(action['objectId'])
            discrete_action['action'] = "SliceObject"
            discrete_action['args']['bbox'] = bbox
            discrete_action['args']['point'] = point
            discrete_action['args']['mask'] = mask

        return discrete_action

    def get_bbox_of_obj(self, object_id):
        instance_detections2D = self.env.last_event.instance_detections2D if self.env.last_event.instance_detections2D != None else []

        if object_id in instance_detections2D:
            np_box = instance_detections2D[object_id]
            return list([int(np_box[0]), int(np_box[1]), int(np_box[2]), int(np_box[3])])  # weird json formatting issue
        else:
            raise Exception("No bounding boxes on screen for %s" % object_id)

    def get_point_of_obj(self, object_id, centroid_type="box_center", visualize_debug=False):
        instance_detections2D = self.env.last_event.instance_detections2D
        instance_seg_frame = np.array(self.env.last_event.instance_segmentation_frame)

        if object_id in instance_detections2D:
            color = self.env.last_event.object_id_to_color[object_id]
            seg_mask = cv2.inRange(instance_seg_frame, color, color)
            nz_rows, nz_cols = np.nonzero(seg_mask)

            if centroid_type == "moment":
                _, contours, hierarchy = cv2.findContours(seg_mask, 1, 2)
                moment = cv2.moments(contours[0])
                if moment["m00"] == 0:
                    return list(["NULL", "NULL"])

                cx = int(moment['m10']/moment['m00'])
                cy = int(moment['m01']/moment['m00'])
            else:
                x0, y0, xmax, ymax = min(nz_cols), min(nz_rows), max(nz_cols), max(nz_rows)
                cx = int((xmax - x0) / 2 + x0)
                cy = int((ymax - y0) / 2 + y0)

            # if centroid not on seg mask (non-convex hull), find the closest point that is on the mask
            if seg_mask[cy, cx] == 0:
                non_zero_coords = zip(nz_cols, nz_rows)
                dist = lambda a, b: (a[0]-b[0])**2 + (a[1]-b[1])**2
                closest_point = min(non_zero_coords, key=lambda co: dist(co, (cx, cy)))
                cx, cy = closest_point

            # plot point on seg mask
            if visualize_debug:
                seg_mask = cv2.cvtColor(seg_mask, cv2.COLOR_GRAY2RGB)
                cv2.imshow('img', instance_seg_frame)
                cv2.imshow('seg', seg_mask)
                cv2.waitKey(0)

                cv2.circle(seg_mask, (cx, cy), 15, (0,0,255), 2)
                cv2.circle(seg_mask, (cx, cy), 2, (0, 0, 255), -1)
                cv2.imshow('seg', seg_mask)
                cv2.waitKey(0)

            return list([int(cx), int(cy)])
        else:
            raise Exception("No point for ", object_id)

    def get_mask_of_obj(self, object_id):
        instance_detections2D = self.env.last_event.instance_detections2D
        instance_seg_frame = np.array(self.env.last_event.instance_segmentation_frame)

        if object_id in instance_detections2D:
            color = self.env.last_event.object_id_to_color[object_id]
            seg_mask = cv2.inRange(instance_seg_frame, color, color)
            seg_mask = np.array(seg_mask) / 255
            seg_mask = seg_mask.astype(int)
            # Compress the segmentation mask using simple run-length compression.
            run_len_compressed = compress_mask(seg_mask)
            return run_len_compressed
        else:
            raise Exception("No mask for ", object_id)

    def get_bbox_point_mask(self, object_id):
        bbox = self.get_bbox_of_obj(object_id)
        point = self.get_point_of_obj(object_id)
        mask = self.get_mask_of_obj(object_id)
        return bbox, point, mask

    def get_some_visible_obj_of_name(self, name):
        objects = [obj for obj in self.env.last_event.metadata['objects'] if obj['objectType'] == name and obj['visible']]
        if len(objects) == 0:
            raise Exception("No visible %s found!" % name)
        return objects[0]  # return the first visible instance

    def step(self, action_or_ind, process_frame=True):
        if self.event is None:
            self.event = self.env.last_event
        self.current_frame_count += 1
        self.total_frame_count += 1
        action, should_fail = self.get_action(action_or_ind)
        if should_fail:
            self.env.last_event.metadata['lastActionSuccess'] = False
        else:
            t_start = time.time()
            start_pose = game_util.get_pose(self.event)
            if 'action' not in action or action['action'] is None or action['action'] == 'None':
                self.env.last_event.metadata['lastActionSuccess'] = True
            else:
                if constants.RECORD_VIDEO_IMAGES:
                    im_ind = len(glob.glob(constants.save_path + '/*.png'))
                    if 'Teleport' in action['action']:
                        position = self.env.last_event.metadata['agent']['position']
                        rotation = self.env.last_event.metadata['agent']['rotation']
                        start_horizon = self.env.last_event.metadata['agent']['cameraHorizon']
                        start_rotation = rotation['y']
                        if (np.abs(action['x'] - position['x']) > 0.001 or
                            np.abs(action['z'] - position['z']) > 0.001):
                            # Movement
                            for xx in np.arange(.1, 1, .1):
                                new_action = copy.deepcopy(action)
                                new_action['x'] = np.round(position['x'] * (1 - xx) + action['x'] * xx, 5)
                                new_action['z'] = np.round(position['z'] * (1 - xx) + action['z'] * xx, 5)
                                new_action['rotation'] = start_rotation
                                new_action['horizon'] = start_horizon
                                self.event = self.env.step(new_action)
                                cv2.imwrite(constants.save_path + '/%09d.png' % im_ind,
                                            self.event.frame[:, :, ::-1])
                                # #game_util.store_image_name('%09d.png' % im_ind)  # eww... seriously need to clean this up
                                im_ind += 1
                        if np.abs(action['horizon'] - self.env.last_event.metadata['agent']['cameraHorizon']) > 0.001:
                            end_horizon = action['horizon']
                            for xx in np.arange(.1, 1, .1):
                                new_action = copy.deepcopy(action)
                                new_action['horizon'] = np.round(start_horizon * (1 - xx) + end_horizon * xx, 3)
                                new_action['rotation'] = start_rotation
                                self.event = self.env.step(new_action)
                                cv2.imwrite(constants.save_path + '/%09d.png' % im_ind,
                                            self.event.frame[:, :, ::-1])
                                #game_util.store_image_name('%09d.png' % im_ind)
                                im_ind += 1
                        if np.abs(action['rotation'] - rotation['y']) > 0.001:
                            end_rotation = action['rotation']
                            for xx in np.arange(.1, 1, .1):
                                new_action = copy.deepcopy(action)
                                new_action['rotation'] = np.round(start_rotation * (1 - xx) + end_rotation * xx, 3)
                                self.event = self.env.step(new_action)
                                cv2.imwrite(constants.save_path + '/%09d.png' % im_ind,
                                            self.event.frame[:, :, ::-1])
                                #game_util.store_image_name('%09d.png' % im_ind)
                                im_ind += 1

                        self.event = self.env.step(action)

                    # elif 'MoveAhead' in action['action']:
                    #     # self.store_ll_action(action)
                    #     # self.save_image(1)
                    #     events = self.env.smooth_move_ahead(action)
                    #     # for event in events:
                    #     #     im_ind = len(glob.glob(constants.save_path + '/*.png'))
                    #     #     cv2.imwrite(constants.save_path + '/%09d.png' % im_ind, event.frame[:, :, ::-1])
                    #     #     #game_util.store_image_name('%09d.png' % im_ind)
                    #
                    # elif 'Rotate' in action['action']:
                    #     # self.store_ll_action(action)
                    #     # self.save_image(1)
                    #     events = self.env.smooth_rotate(action)
                    #     # for event in events:
                    #     #     im_ind = len(glob.glob(constants.save_path + '/*.png'))
                    #     #     cv2.imwrite(constants.save_path + '/%09d.png' % im_ind, event.frame[:, :, ::-1])
                    #     #     #game_util.store_image_name('%09d.png' % im_ind)
                    #
                    # elif 'Look' in action['action']:
                    #     # self.store_ll_action(action)
                    #     # self.save_image(1)
                    #     events = self.env.smooth_look(action)
                    #     # for event in events:
                    #     #     im_ind = len(glob.glob(constants.save_path + '/*.png'))
                    #     #     cv2.imwrite(constants.save_path + '/%09d.png' % im_ind, event.frame[:, :, ::-1])
                    #     #     #game_util.store_image_name('%09d.png' % im_ind)

                    elif 'OpenObject' in action['action']:
                        open_action = dict(action=action['action'],
                                           objectId=action['objectId'],
                                           moveMagnitude=1.0)
                        # self.store_ll_action(open_action)
                        # self.save_act_image(open_action, dir=constants.BEFORE)

                        self.event = self.env.step(open_action)

                        # self.save_act_image(open_action, dir=constants.AFTER)
                        self.check_action_success(self.event)

                    elif 'CloseObject' in action['action']:
                        close_action = dict(action=action['action'],
                                            objectId=action['objectId'])
                        # self.store_ll_action(close_action)
                        # self.save_act_image(close_action, dir=constants.BEFORE)

                        self.event = self.env.step(close_action)

                        # self.save_act_image(close_action, dir=constants.AFTER)
                        self.check_action_success(self.event)

                    elif 'PickupObject' in action['action']:
                        # [hack] correct object ids of slices
                        action['objectId'] = self.correct_slice_id(action['objectId'])

                        # open the receptacle if needed
                        parent_recep = self.get_parent_receps(action['objectId'])
                        if parent_recep is not None and parent_recep['objectType'] in constants.OPENABLE_CLASS_SET:
                            self.open_recep(parent_recep)  # stores LL action

                        # close/open the object if needed
                        pickup_obj = game_util.get_object(action['objectId'], self.env.last_event.metadata)
                        if pickup_obj['objectType'] in constants.FORCED_OPEN_STATE_ON_PICKUP:
                            if pickup_obj['isOpen'] != constants.FORCED_OPEN_STATE_ON_PICKUP[pickup_obj['objectType']]:
                                if pickup_obj['isOpen']:
                                    self.close_recep(pickup_obj)  # stores LL action
                                else:
                                    self.open_recep(pickup_obj)  # stores LL action

                        # pick up the object
                        self.check_obj_visibility(action, min_pixels=10)
                        pickup_action = dict(action=action['action'],
                                             objectId=action['objectId'])
                        # self.store_ll_action(pickup_action)
                        # self.save_act_image(pickup_action, dir=constants.BEFORE)

                        self.event = self.env.step(pickup_action)

                        # self.save_act_image(pickup_action, dir=constants.AFTER)
                        self.check_action_success(self.event)

                        # close the receptacle if needed
                        if parent_recep is not None:
                            parent_recep = game_util.get_object(parent_recep['objectId'], self.env.last_event.metadata)
                            self.close_recep(parent_recep)  # stores LL action

                    elif 'PutObject' in action['action']:
                        if len(self.env.last_event.metadata['inventoryObjects']) > 0:
                            inv_obj = self.env.last_event.metadata['inventoryObjects'][0]['objectId']
                        else:
                            raise RuntimeError("Taking 'PutObject' action with no held inventory object")
                        action['objectId'] = inv_obj

                        # open the receptacle if needed
                        parent_recep = game_util.get_object(action['receptacleObjectId'], self.env.last_event.metadata)
                        if parent_recep is not None and parent_recep['objectType'] in constants.OPENABLE_CLASS_SET:
                            self.open_recep(parent_recep)  # stores LL action

                        # Open the parent receptacle of the movable receptacle target.
                        elif parent_recep['objectType'] in constants.MOVABLE_RECEPTACLES_SET:
                            movable_parent_recep_ids = parent_recep['parentReceptacles']
                            if movable_parent_recep_ids is not None and len(movable_parent_recep_ids) > 0:
                                print(parent_recep['objectId'], movable_parent_recep_ids)  # DEBUG
                                movable_parent_recep = game_util.get_object(movable_parent_recep_ids[0],
                                                                            self.env.last_event.metadata)
                                if movable_parent_recep['objectType'] in constants.OPENABLE_CLASS_SET:
                                    self.open_recep(movable_parent_recep)  # stores LL action

                        # put the object
                        put_action = dict(action=action['action'],
                                          objectId=action['objectId'],
                                          receptacleObjectId=action['receptacleObjectId'],
                                          forceAction=True,
                                          placeStationary=True)
                        # self.store_ll_action(put_action)
                        # self.save_act_image(put_action, dir=constants.BEFORE)

                        self.event = self.env.step(put_action)

                        # self.save_act_image(put_action, dir=constants.AFTER)
                        self.check_obj_visibility(action)
                        self.check_action_success(self.event)

                        # close the receptacle if needed
                        if parent_recep is not None:
                            parent_recep = game_util.get_object(parent_recep['objectId'], self.env.last_event.metadata)
                            self.close_recep(parent_recep)  # stores LL action

                    elif 'CleanObject' in action['action']:
                        # put the object in the sink
                        sink_obj_id = self.get_some_visible_obj_of_name('SinkBasin')['objectId']
                        inv_obj = self.env.last_event.metadata['inventoryObjects'][0]
                        put_action = dict(action='PutObject',
                                          objectId=inv_obj['objectId'],
                                          receptacleObjectId=sink_obj_id,
                                          forceAction=True,
                                          placeStationary=True)
                        # self.store_ll_action(put_action)
                        # self.save_act_image(put_action, dir=constants.BEFORE)
                        self.event = self.env.step(put_action)
                        # self.save_act_image(put_action, dir=constants.AFTER)
                        self.check_obj_visibility(put_action)
                        self.check_action_success(self.event)

                        # turn on the tap
                        clean_action = copy.deepcopy(action)
                        clean_action['action'] = 'ToggleObjectOn'
                        clean_action['objectId'] = game_util.get_obj_of_type_closest_to_obj(
                            "Faucet", inv_obj['objectId'], self.env.last_event.metadata)['objectId']
                        # self.store_ll_action(clean_action)
                        # self.save_act_image(clean_action, dir=constants.BEFORE)
                        self.event = self.env.step({k: clean_action[k]
                                                    for k in ['action', 'objectId']})
                        # self.save_act_image(clean_action, dir=constants.AFTER)
                        self.check_action_success(self.event)
                        # Need to delay one frame to let `isDirty` update on stream-affected.
                        self.env.noop()

                        # Call built-in 'CleanObject' THOR action on every object in the SinkBasin.
                        # This means we clean everything in the sink, rather than just the things that happen to touch
                        # the water stream, which is the default simulator behavior but looks weird for our purposes.
                        sink_basin_obj = game_util.get_obj_of_type_closest_to_obj(
                            "SinkBasin", clean_action['objectId'], self.env.last_event.metadata)
                        for in_sink_obj_id in sink_basin_obj['receptacleObjectIds']:
                            if (game_util.get_object(in_sink_obj_id, self.env.last_event.metadata)['dirtyable']
                                    and game_util.get_object(in_sink_obj_id, self.env.last_event.metadata)['isDirty']):
                                self.event = self.env.step({'action': 'CleanObject',
                                                            'objectId': in_sink_obj_id})

                        # turn off the tap
                        close_action = copy.deepcopy(clean_action)
                        close_action['action'] = 'ToggleObjectOff'
                        # self.store_ll_action(close_action)
                        # self.save_act_image(close_action, dir=constants.BEFORE)
                        self.event = self.env.step({k: close_action[k]
                                                    for k in ['action', 'objectId']})
                        # self.save_act_image(action, dir=constants.AFTER)
                        self.check_action_success(self.event)

                        # pick up the object from the sink
                        pickup_action = dict(action='PickupObject',
                                             objectId=inv_obj['objectId'])
                        # self.store_ll_action(pickup_action)
                        # self.save_act_image(pickup_action, dir=constants.BEFORE)
                        self.event = self.env.step(pickup_action)
                        # self.save_act_image(pickup_action, dir=constants.AFTER)
                        self.check_obj_visibility(pickup_action)
                        self.check_action_success(self.event)

                    elif 'HeatObject' in action['action']:
                        # open the microwave
                        microwave_obj_id = self.get_some_visible_obj_of_name('Microwave')['objectId']
                        microwave_obj = game_util.get_object(microwave_obj_id, self.env.last_event.metadata)
                        self.open_recep(microwave_obj)

                        # put the object in the microwave
                        inv_obj = self.env.last_event.metadata['inventoryObjects'][0]
                        put_action = dict(action='PutObject',
                                          objectId=inv_obj['objectId'],
                                          receptacleObjectId=microwave_obj_id,
                                          forceAction=True,
                                          placeStationary=True)
                        # self.store_ll_action(put_action)
                        # self.save_act_image(put_action, dir=constants.BEFORE)
                        self.event = self.env.step(put_action)
                        # self.save_act_image(put_action, dir=constants.AFTER)
                        self.check_obj_visibility(put_action)
                        self.check_action_success(self.event)

                        # close the microwave
                        microwave_obj = game_util.get_object(microwave_obj_id, self.env.last_event.metadata)
                        self.close_recep(microwave_obj)

                        # turn on the microwave
                        heat_action = copy.deepcopy(action)
                        heat_action['action'] = 'ToggleObjectOn'
                        heat_action['objectId'] = microwave_obj_id
                        # self.store_ll_action(heat_action)
                        # self.save_act_image(heat_action, dir=constants.BEFORE)
                        self.event = self.env.step({k: heat_action[k]
                                                    for k in ['action', 'objectId']})
                        # self.save_act_image(heat_action, dir=constants.AFTER)

                        # turn off the microwave
                        stop_action = copy.deepcopy(heat_action)
                        stop_action['action'] = 'ToggleObjectOff'
                        # self.store_ll_action(stop_action)
                        # self.save_act_image(stop_action, dir=constants.BEFORE)
                        self.event = self.env.step({k: stop_action[k]
                                                    for k in ['action', 'objectId']})
                        # self.save_act_image(stop_action, dir=constants.AFTER)

                        # open the microwave
                        microwave_obj = game_util.get_object(microwave_obj_id, self.env.last_event.metadata)
                        self.open_recep(microwave_obj)

                        # pick up the object from the microwave
                        pickup_action = dict(action='PickupObject',
                                             objectId=inv_obj['objectId'])
                        # self.store_ll_action(pickup_action)
                        # self.save_act_image(pickup_action, dir=constants.BEFORE)
                        self.event = self.env.step(pickup_action)
                        # self.save_act_image(pickup_action, dir=constants.AFTER)
                        self.check_obj_visibility(pickup_action)
                        self.check_action_success(self.event)

                        # close the microwave again
                        microwave_obj = game_util.get_object(microwave_obj_id, self.env.last_event.metadata)
                        self.close_recep(microwave_obj)

                    elif 'CoolObject' in action['action']:
                        # open the fridge
                        fridge_obj_id = self.get_some_visible_obj_of_name('Fridge')['objectId']
                        fridge_obj = game_util.get_object(fridge_obj_id, self.env.last_event.metadata)
                        self.open_recep(fridge_obj)

                        # put the object in the fridge
                        inv_obj = self.env.last_event.metadata['inventoryObjects'][0]
                        put_action = dict(action='PutObject',
                                          objectId=inv_obj['objectId'],
                                          receptacleObjectId=fridge_obj_id,
                                          forceAction=True,
                                          placeStationary=True)
                        # self.store_ll_action(put_action)
                        # self.save_act_image(put_action, dir=constants.BEFORE)
                        self.event = self.env.step(put_action)
                        # self.save_act_image(put_action, dir=constants.AFTER)
                        self.check_obj_visibility(put_action)
                        self.check_action_success(self.event)

                        # close and cool the object inside the frige
                        cool_action = dict(action='CloseObject',
                                           objectId=action['objectId'])
                        # self.store_ll_action(cool_action)
                        # self.save_act_image(action, dir=constants.BEFORE)
                        self.event = self.env.step(cool_action)
                        # self.save_act_image(action, dir=constants.MIDDLE)
                        # self.save_act_image(action, dir=constants.AFTER)

                        # open the fridge again
                        fridge_obj = game_util.get_object(fridge_obj_id, self.env.last_event.metadata)
                        self.open_recep(fridge_obj)

                        # pick up the object from the fridge
                        pickup_action = dict(action='PickupObject',
                                             objectId=inv_obj['objectId'])
                        # self.store_ll_action(pickup_action)
                        # self.save_act_image(pickup_action, dir=constants.BEFORE)
                        self.event = self.env.step(pickup_action)
                        # self.save_act_image(pickup_action, dir=constants.AFTER)
                        self.check_obj_visibility(pickup_action)
                        self.check_action_success(self.event)

                        # close the fridge again
                        fridge_obj = game_util.get_object(fridge_obj_id, self.env.last_event.metadata)
                        self.close_recep(fridge_obj)

                    elif 'ToggleObject' in action['action']:
                        on_action = dict(action=action['action'],
                                         objectId=action['objectId'])

                        toggle_obj = game_util.get_object(action['objectId'], self.env.last_event.metadata)
                        on_action['action'] = 'ToggleObjectOff' if toggle_obj['isToggled'] else 'ToggleObjectOn'
                        # self.store_ll_action(on_action)
                        # self.save_act_image(on_action, dir=constants.BEFORE)
                        self.event = self.env.step(on_action)
                        # self.save_act_image(on_action, dir=constants.AFTER)
                        self.check_action_success(self.event)

                    elif 'SliceObject' in action['action']:
                        # open the receptacle if needed
                        parent_recep = self.get_parent_receps(action['objectId'])
                        if parent_recep is not None and parent_recep['objectType'] in constants.OPENABLE_CLASS_SET:
                            self.open_recep(parent_recep)

                        # slice the object
                        slice_action = dict(action=action['action'],
                                            objectId=action['objectId'])
                        # self.store_ll_action(slice_action)
                        # self.save_act_image(slice_action, dir=constants.BEFORE)
                        self.event = self.env.step(slice_action)
                        # self.save_act_image(action, dir=constants.AFTER)
                        self.check_action_success(self.event)

                        # close the receptacle if needed
                        if parent_recep is not None:
                            parent_recep = game_util.get_object(parent_recep['objectId'], self.env.last_event.metadata)
                            self.close_recep(parent_recep)  # stores LL action

                    else:
                        # check that the object to pick is visible in the camera frame
                        if action['action'] == 'PickupObject':
                            self.check_obj_visibility(action)

                        # self.store_ll_action(action)
                        # self.save_act_image(action, dir=constants.BEFORE)

                        self.event = self.env.step(action)

                        # self.save_act_image(action, dir=constants.AFTER)

                        if action['action'] == 'PutObject':
                            self.check_obj_visibility(action)
                else:
                    self.event = self.env.step(action)
                new_pose = game_util.get_pose(self.event)
                point_dists = np.sum(np.abs(self.gt_graph.points - np.array(new_pose)[:2]), axis=1)
                if np.min(point_dists) > 0.0001:
                    # print('Point teleport failure')
                    self.event = self.env.step({
                        'action': 'Teleport',
                        'x': start_pose[0] * constants.AGENT_STEP_SIZE,
                        'y': self.agent_height,
                        'z': start_pose[1] * constants.AGENT_STEP_SIZE,
                        'rotateOnTeleport': True,
                        'rotation': new_pose[2] * 90,
                    })
                    self.env.last_event.metadata['lastActionSuccess'] = False

            self.timers[0, 0] += time.time() - t_start
            self.timers[0, 1] += 1
            if self.timers[0, 1] % 100 == 0:
                # print('env step time %.3f' % (self.timers[0, 0] / self.timers[0, 1]))
                self.timers[0, :] = 0

        if self.env.last_event.metadata['lastActionSuccess']:
            if action['action'] == 'OpenObject':
                self.currently_opened_object_ids.add(action['objectId'])
            elif action['action'] == 'CloseObject':
                self.currently_opened_object_ids.remove(action['objectId'])
            elif action['action'] == 'PickupObject':
                self.inventory_ids.add(action['objectId'])
            elif action['action'] == 'PutObject':
                self.inventory_ids.remove(action['objectId'])

        if self.env.last_event.metadata['lastActionSuccess'] and process_frame:
            self.process_frame()

    def correct_slice_id(self, object_id):
        main_obj = game_util.get_object(object_id, self.env.last_event.metadata)
        if (main_obj is not None and
                main_obj['objectType'] in constants.VAL_ACTION_OBJECTS['Sliceable'] and main_obj['isSliced']):
            slice_obj_type = main_obj['objectType'] + "Sliced"

            # Iterate through slices and pick up the one closest to the agent according to the 'distance' metadata.
            min_d = None
            best_slice_id = None
            sidx = 1
            while True:
                slice_obj_id = main_obj['objectId'] + "|" + slice_obj_type + "_%d" % sidx
                slice_obj = game_util.get_object(slice_obj_id, self.env.last_event.metadata)
                if slice_obj is None:
                    break
                if min_d is None or slice_obj['distance'] < min_d:
                    min_d = slice_obj['distance']
                    best_slice_id = slice_obj_id
                sidx += 1
            return best_slice_id
        return object_id

    def open_recep(self, recep):
        if recep['openable'] and not recep['isOpen']:
            open_action = dict(action='OpenObject',
                               objectId=recep['objectId'])

            # self.store_ll_action(open_action)
            # self.save_act_image(open_action, dir=constants.BEFORE)
            self.event = self.env.step(open_action)
            # self.save_act_image(open_action, dir=constants.AFTER)
            self.check_action_success(self.event)

    def close_recep(self, recep):
        if recep['openable'] and recep['isOpen']:
            close_action = dict(action='CloseObject',
                                objectId=recep['objectId'])
            # self.store_ll_action(close_action)
            # self.save_act_image(close_action, dir=constants.BEFORE)
            self.event = self.env.step(close_action)
            # self.save_act_image(close_action, dir=constants.AFTER)
            self.check_action_success(self.event)

    def get_parent_receps(self, object_id):
        obj = game_util.get_object(object_id, self.env.last_event.metadata)
        if obj is None or obj['parentReceptacles'] is None:
            return None

        parent_receps = [game_util.get_object(recep, self.env.last_event.metadata) for recep in
                         obj['parentReceptacles']]
        parent_receps = [recep for recep in parent_receps if recep['objectType'] in constants.OPENABLE_CLASS_SET]

        if len(parent_receps) > 0:
            return parent_receps[0]
        else:
            return None

    def check_obj_visibility(self, action, min_pixels=constants.MIN_VISIBLE_PIXELS):
        for o in self.env.last_event.metadata['objects']:
            if o['objectId'] == action['objectId'] and \
                    "Sliced" not in action['objectId']: # NOTE: ignore slice objects for this check
                instance_masks = self.env.last_event.instance_masks
                total_obj_pixels = np.sum(instance_masks[o['objectId']] if o['objectId'] in instance_masks else [0])
                # print("\tNum. Object Pixels: " + str(total_obj_pixels))
                if not o['visible'] or (total_obj_pixels < min_pixels):
                    raise Exception("Pickup item not visible! Visible property: " + str(o['visible']) +
                                     "; visible pixes: " + str(total_obj_pixels))

    def check_action_success(self, event):
        if not event.metadata['lastActionSuccess']:
            raise Exception("API Action Failed: %s" % event.metadata['errorMessage'])

    def save_act_image(self, action, dir=constants.BEFORE):
        self.save_image(constants.SAVE_FRAME_BEFORE_AND_AFTER_COUNTS[action['action']][dir])

    def save_image(self, count):
        if constants.RECORD_VIDEO_IMAGES:
            im_ind = -1
            for i in range(count):
                im_ind = len(glob.glob(constants.save_path + '/*.png'))
                cv2.imwrite(constants.save_path + '/%09d.png' % im_ind, self.env.last_event.frame[:, :, ::-1])
                #game_util.store_image_name('%09d.png' % im_ind)

                self.env.noop()
            return im_ind
        else:
            return -1

    def process_frame(self):
        self.event = self.env.last_event
        self.pose = game_util.get_pose(self.event)

        self.s_t_orig = self.event.frame
        self.s_t = game_util.imresize(self.event.frame,
                                      (constants.SCREEN_HEIGHT, constants.SCREEN_WIDTH), rescale=False)

        self.s_t_depth = game_util.imresize(self.event.depth_frame,
                                            (constants.SCREEN_HEIGHT, constants.SCREEN_WIDTH), rescale=False)
