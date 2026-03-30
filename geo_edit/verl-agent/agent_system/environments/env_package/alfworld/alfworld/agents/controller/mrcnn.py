############
# Reference: https://haochen23.github.io/2020/05/instance-segmentation-mask-rcnn.html#.XyiLqfhKg40

import os
import cv2
import sys
import json
import re
import copy
import random
import traceback
from collections import Counter

from PIL import Image
import numpy as np

import alfworld.gen
import alfworld.gen.constants as constants

from alfworld.agents.controller.base import BaseAgent
from alfworld.agents.utils.misc import extract_admissible_commands_with_heuristics

try:
    import torchvision.transforms as T
except ImportError:
    raise ImportError("torchvision not found. Please install them via `pip install alfworld[full]`.")

class MaskRCNNAgent(BaseAgent):

    def __init__(self, env, traj_data, traj_root,
                 pretrained_model=None,
                 load_receps=False, debug=False,
                 goal_desc_human_anns_prob=0.0,
                 classes=constants.OBJECTS_DETECTOR,
                 save_detections_to_disk=False, save_detections_path='./'):

        self.openable_points = self.get_openable_points(traj_data)

        # pre-trained MaskRCNN model
        assert pretrained_model
        self.mask_rcnn = pretrained_model
        self.mask_rcnn.eval()
        self.mask_rcnn.cuda()
        self.transform = T.Compose([T.ToTensor()])
        self.classes = classes

        # object state tracking
        self.cleaned_objects = set()
        self.cooled_objects = set()
        self.heated_objects = set()

        # recording settings
        self.save_detections_to_disk = save_detections_to_disk
        self.save_detections_path = save_detections_path

        super().__init__(env, traj_data, traj_root,
                         load_receps=load_receps, debug=debug,
                         goal_desc_human_anns_prob=goal_desc_human_anns_prob,
                         recep_filename='receps_mrcnn.json', exhaustive_exploration=False)

    def get_openable_points(self, traj_data):
        scene_num = traj_data['scene']['scene_num']
        openable_json_file = os.path.join(alfworld.gen.__path__[0], 'layouts/FloorPlan%d-openable.json' % scene_num)
        with open(openable_json_file, 'r') as f:
            openable_points = json.load(f)
        return openable_points

    # use pre-computed openable points from ALFRED to store receptacle locations
    def explore_scene(self):
        agent_height = self.env.last_event.metadata['agent']['position']['y']
        for object_id, point in self.openable_points.items():
            action = {'action': 'TeleportFull',
                      'x': point[0],
                      'y': agent_height,
                      'z': point[1],
                      'rotateOnTeleport': False,
                      'rotation': point[2],
                      'horizon': point[3]}
            event = self.env.step(action)

            if event and event.metadata['lastActionSuccess']:
                instance_segs = np.array(self.env.last_event.instance_segmentation_frame)
                color_to_object_id = self.env.last_event.color_to_object_id

                # find unique instance segs
                color_count = Counter()
                for x in range(instance_segs.shape[0]):
                    for y in range(instance_segs.shape[1]):
                        color = instance_segs[x, y]
                        color_count[tuple(color)] += 1

                for color, num_pixels in color_count.most_common():
                    if color in color_to_object_id:
                        object_id = color_to_object_id[color]
                        object_type = object_id.split('|')[0]
                        if "Basin" in object_id:
                            object_type += "Basin"

                        if object_type in self.STATIC_RECEPTACLES:
                            if object_id not in self.receptacles:
                                self.receptacles[object_id] = {
                                    'object_id': object_id,
                                    'object_type': object_type,
                                    'locs': action,
                                    'num_pixels': num_pixels,
                                    'mask': np.array(np.all(instance_segs == np.array(color), axis=-1), dtype=int),
                                    'num_id': "%s %d" % (object_type.lower(), self.get_next_num_id(object_type, self.receptacles)),
                                    'closed': True if object_type in constants.OPENABLE_CLASS_LIST else None
                                }
                            elif object_id in self.receptacles and num_pixels > self.receptacles[object_id]['num_pixels']:
                                self.receptacles[object_id]['locs'] = action  # .append(action)
                                self.receptacles[object_id]['num_pixels'] = num_pixels

        # self.save_receps()

    # explore each scene exhaustively with grid-search
    def explore_scene_exhaustively(self):
        event = self.env.step(dict(action='GetReachablePositions',
                                   gridSize=constants.AGENT_STEP_SIZE))

        reachable_points = set()
        for point in event.metadata['actionReturn']:
            reachable_points.add((point['x'], point['z']))

        agent_height = event.metadata['agent']['position']['y']

        # teleport to points
        for p_idx, point in enumerate(reachable_points):
            if (p_idx+1) % 10 == 0:
                print("Checking %d/%d..." % (p_idx+1, len(reachable_points)))

            action = {'action': 'TeleportFull',
                      'x': point[0],
                      'y': agent_height,
                      'z': point[1]}
            event = self.env.step(action)

            # look up and down
            if event and event.metadata['lastActionSuccess']:
                for horizon in [-30, 0, 30]:
                    action = {'action': 'TeleportFull',
                              'x': point[0],
                              'y': agent_height,
                              'z': point[1],
                              'rotateOnTeleport': True,
                              'rotation': 0,
                              'horizon': horizon}
                    event = self.env.step(action)

                    # look around
                    if event and event.metadata['lastActionSuccess']:
                        for r in range(4):
                            rotation = 90 * r
                            action = {'action': 'TeleportFull',
                                      'x': point[0],
                                      'y': agent_height,
                                      'z': point[1],
                                      'rotateOnTeleport': True,
                                      'rotation': rotation,
                                      'horizon': horizon}
                            event = self.env.step(action)

                            if event and event.metadata['lastActionSuccess']:
                                masks, boxes, pred_cls = self.get_instance_seg()

                                for i in range(len(masks)):
                                    object_id = "{}|{}".format(pred_cls[i], str(boxes[i]))
                                    object_type = object_id.split('|')[0]
                                    num_pixels = len(masks[i].nonzero()[0])
                                    if "Basin" in object_id:
                                        object_type += "Basin"
                                    if object_type in self.STATIC_RECEPTACLES: # and object_type == recep_class:
                                        recep_id = object_id
                                        if recep_id not in self.receptacles:
                                            self.receptacles[recep_id] = {
                                                'object_id': recep_id,
                                                'object_type': object_type,
                                                'locs': action,
                                                'mask': masks[i] if masks[i].shape == (300, 300) else np.ones((300, 300)),
                                                'num_pixels': num_pixels,
                                                'num_id': "%s %d" % (object_type.lower(), self.get_next_num_id(object_type, self.receptacles)),
                                                'closed': True if object_type in constants.OPENABLE_CLASS_LIST else None
                                            }
                                        elif recep_id in self.receptacles and num_pixels > self.receptacles[recep_id]['num_pixels']:
                                            self.receptacles[recep_id]['locs'] = action  # .append(action)
                                            self.receptacles[recep_id]['num_pixels'] = num_pixels

        # self.save_receps()

    # display a list of visible objects
    def print_frame(self, recep, loc):
        visible_objects = self.update_detection(recep, loc)

        visible_objects_with_articles = ["a %s," % vo for vo in visible_objects]
        feedback = ""
        if len(visible_objects) > 0:
            feedback = "On the %s, you see %s" % (recep['num_id'], self.fix_and_comma_in_the_end(' '.join(visible_objects_with_articles)))
        elif not recep['closed'] and len(visible_objects) == 0:
            feedback = "On the %s, you see nothing." % (recep['num_id'])

        return visible_objects, feedback

    def update_detection(self, recep, loc):
        masks, boxes, pred_cls = self.get_instance_seg()
        # for each unique seg add to object dictionary if it's more visible than before
        visible_objects = []
        self.objects = {}
        for i in range(len(masks)):
            object_id = "{}|{}".format(pred_cls[i], str(boxes[i]))
            object_type = pred_cls[i]
            num_pixels = len(masks[i].nonzero()[0])
            if object_type in self.OBJECTS:
                if object_id not in self.objects:
                    num_id = "%s %d" % (
                        object_type.lower() if "Sliced" not in object_id else "sliced-%s" % object_type.lower(),
                        self.get_next_num_id(object_type, self.objects))
                    self.objects[object_id] = {
                        'object_id': object_id,
                        'object_type': object_type,
                        'parent': recep['object_id'],
                        'loc': loc,
                        'mask': masks[i],
                        'num_pixels': num_pixels,
                        'num_id': num_id
                    }
                elif object_id in self.objects and num_pixels > self.objects[object_id]['num_pixels']:
                    self.objects[object_id]['loc'] = loc
                    self.objects[object_id]['mask'] = masks[i]
                    self.objects[object_id]['num_pixels'] = num_pixels

                if self.objects[object_id]['num_id'] not in self.inventory:
                    visible_objects.append(self.objects[object_id]['num_id'])
        return visible_objects

    def update_gt_receptacle_mask(self, recep):
        instance_segs = self.env.last_event.instance_segmentation_frame
        object_id_to_color = self.env.last_event.object_id_to_color
        recep_id = recep['object_id']
        if recep_id in object_id_to_color:
            recep_instance_color = object_id_to_color[recep_id]
            recep['mask'] = np.array(np.all(instance_segs == np.array(recep_instance_color), axis=-1), dtype=int)

    def get_most_visible_object_of_type(self, otype, obj_dict):
        max_pixels, best_view_obj = 0, None
        for id, obj in obj_dict.items():
            if otype in obj['object_type'].lower():
                if obj['num_pixels'] > max_pixels:
                    best_view_obj = obj
                    max_pixels = obj['num_pixels']
        return best_view_obj

    def get_object_of_num_id(self, onum, obj_dict):
        obj = None
        for id, o in obj_dict.items():
            if onum == o['num_id']:
                return o
        return obj

    def get_coloured_mask(self, mask):
        """
        random_colour_masks
          parameters:
            - image - predicted masks
          method:
            - the masks of each predicted object is given random colour for visualization
        """
        colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
        r = np.zeros_like(mask).astype(np.uint8)
        g = np.zeros_like(mask).astype(np.uint8)
        b = np.zeros_like(mask).astype(np.uint8)
        r[mask == 1], g[mask == 1], b[mask == 1] = colours[random.randrange(0,10)]
        coloured_mask = np.stack([r, g, b], axis=2)
        return coloured_mask

    def get_maskrcnn_prediction(self, frame, confidence):
        """
        get_prediction
          parameters:
            - img_path - path of the input image
            - confidence - threshold to keep the prediction or not
          method:
            - Image is obtained from the image path
            - the image is converted to image tensor using PyTorch's Transforms
            - image is passed through the model to get the predictions
            - masks, classes and bounding boxes are obtained from the model and soft masks are made binary(0 or 1) on masks
              ie: eg. segment of cat is made 1 and rest of the image is made 0

        """
        img = Image.fromarray(frame)
        img = self.transform(img).cuda()
        pred = self.mask_rcnn([img])
        pred_score = list(pred[0]['scores'].detach().cpu().numpy())
        pred_pruned = [pred_score.index(x) for x in pred_score if x>confidence]

        if len(pred_pruned) > 0:
            pred_t = pred_pruned[-1]
            masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
            pred_class = [self.classes[i] for i in list(pred[0]['labels'].detach().cpu().numpy())]
            pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
            masks = masks[:pred_t+1]
            pred_boxes = pred_boxes[:pred_t+1] if len(pred_pruned) > 1 else pred_boxes
            pred_class = pred_class[:pred_t+1] if len(pred_pruned) > 1 else pred_class
            return masks, pred_boxes, pred_class
        else:
            return [], [], []

    def get_instance_seg(self):
        '''
        Ground-truth instance segemetations (with consistent object IDs) from THOR
        '''

        frame = self.env.last_event.frame[:,:,::-1]
        masks, boxes, pred_cls = self.get_maskrcnn_prediction(frame, confidence=0.5) # TODO: add confidence to config file

        # visualize detections
        if self.debug:
            img = copy.deepcopy(frame)
            for i in range(len(masks)):
                rgb_mask = self.get_coloured_mask(masks[i])
                if rgb_mask.shape == img.shape:
                    img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
                    cv2.rectangle(img, boxes[i][0], boxes[i][1],color=(0, 255, 0), thickness=2)
                    cv2.putText(img, pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),thickness=2)

            cv2.imshow("MaskRCNN", img)
            cv2.waitKey(1)

            if self.save_detections_to_disk:
                if not os.path.exists(self.save_detections_path):
                    os.makedirs(self.save_detections_path)
                for _ in range(10): # save 10 frames
                    img_idx = len(glob.glob(self.save_detections_path + '/*.png'))
                    cv2.imwrite(self.save_detections_path + '/%09d.png' % img_idx, img)

        return masks, boxes, pred_cls

    def get_object_state(self, object_id):
        is_clean = object_id in self.cleaned_objects
        is_hot = object_id in self.heated_objects
        is_cool = object_id in self.cooled_objects
        is_sliced = 'Sliced' in object_id
        return is_clean, is_cool, is_hot, is_sliced

    def get_admissible_commands(self):
        return extract_admissible_commands_with_heuristics(self.intro, self.frame_desc, self.feedback,
                                                           self.curr_recep, self.inventory)

    def interact(self, action, object_name):
        # exception: pass actions
        if action == 'Pass':
            self.env.va_interact(action=action)
            return self.env.last_event, None

        # extract context
        object_type = object_name.split()[0]
        recep, loc = self.get_object(self.curr_recep, self.receptacles), self.curr_loc

        # update MaskRCNN detections
        _ = self.update_detection(recep, self.curr_loc)
        self.update_gt_receptacle_mask(recep)

        # choose which dictionary to look at
        recep_classes = [c.lower() for c in list(self.STATIC_RECEPTACLES)]
        obj_dict = self.receptacles if object_type in recep_classes else self.objects

        # find the object with specific num id, if not, then find some object of same class
        tar_object = self.get_object_of_num_id(object_name, obj_dict)
        if not tar_object:
            tar_object = self.get_most_visible_object_of_type(object_type, obj_dict)

        # use va_interact with ThorEnv
        _, event, _, _, _ = self.env.va_interact(action=action, interact_mask=tar_object['mask'])

        return event, tar_object

    def step(self, action_str):

        event = None
        self.feedback = "Nothing happens."

        try:
            cmd = self.parse_command(action_str)

            if cmd['action'] == self.Action.GOTO:
                target = cmd['tar']
                recep = self.get_object(target, self.receptacles)
                if recep and recep['num_id'] == self.curr_recep:
                    return self.feedback
                self.curr_loc = recep['locs']
                event = self.navigate(self.curr_loc)
                self.curr_recep = target
                self.visible_objects, self.feedback = self.print_frame(recep, self.curr_loc)

                # feedback conditions
                if event and event.metadata['lastActionSuccess']:
                    loc_id = list(self.receptacles.keys()).index(recep['object_id'])
                    loc_feedback = "You arrive at loc %s. " % (loc_id)
                    state_feedback = "The {} is {}. ".format(self.curr_recep, "closed" if recep['closed'] else "open") if recep['closed'] is not None else ""
                    loc_state_feedback = loc_feedback + state_feedback
                    self.feedback = loc_state_feedback + self.feedback if "closed" not in state_feedback else loc_state_feedback
                    self.frame_desc = str(self.feedback)

            elif cmd['action'] == self.Action.PICK:
                obj, rel, tar = cmd['obj'], cmd['rel'], cmd['tar']
                if obj in self.visible_objects:
                    event, object = self.interact('PickupObject', obj)
                    if event and event.metadata['lastActionSuccess']:
                        self.inventory.append(object['num_id'])
                        self.feedback = "You pick up the %s from the %s." % (obj, tar)

            elif cmd['action'] == self.Action.PUT:
                obj, rel, tar = cmd['obj'], cmd['rel'], cmd['tar']
                event, recep = self.interact('PutObject', tar)
                if event and event.metadata['lastActionSuccess']:
                    self.inventory.pop()
                    self.feedback = "You put the %s %s %s." % (obj, rel, tar)

            elif cmd['action'] == self.Action.OPEN:
                target = cmd['tar']
                event, recep = self.interact('OpenObject', target)
                if event and event.metadata['lastActionSuccess']:
                    self.receptacles[recep['object_id']]['closed'] = False
                    self.visible_objects, self.feedback = self.print_frame(recep, self.curr_loc)
                    action_feedback = "You open the %s. " % target
                    self.feedback = action_feedback + self.feedback
                    self.frame_desc = str(self.feedback)

            elif cmd['action'] == self.Action.CLOSE:
                target = cmd['tar']
                event, recep = self.interact('CloseObject', target)
                if event and event.metadata['lastActionSuccess']:
                    self.receptacles[recep['object_id']]['closed'] = True
                    self.feedback = "You close the %s." % target
                    _ = self.update_detection(recep, self.curr_loc)

            elif cmd['action'] == self.Action.TOGGLE:
                target = cmd['tar']
                event, object = self.interact('ToggleObjectOn', target)
                if event and event.metadata['lastActionSuccess']:
                    self.feedback = "The %s is on." % object['num_id']

            elif cmd['action'] == self.Action.HEAT:
                obj, rel, tar = cmd['obj'], cmd['rel'], cmd['tar']

                # open the microwave, heat the object, take the object, close the microwave
                events = []
                event, microwave = self.interact('OpenObject', tar)
                events.append(event)
                event, microwave = self.interact('PutObject', tar)
                events.append(event)
                event, microwave = self.interact('CloseObject', tar)
                events.append(event)
                event, microwave = self.interact('ToggleObjectOn', tar)
                events.append(event)
                event, _ = self.interact('Pass', "")
                events.append(event)
                event, microwave = self.interact('ToggleObjectOff', tar)
                events.append(event)
                event, microwave = self.interact('OpenObject', tar)
                events.append(event)
                event, object = self.interact('PickupObject', obj)
                events.append(event)
                event, microwave = self.interact('CloseObject', tar)
                events.append(event)

                if all(e.metadata['lastActionSuccess'] for e in events) and self.curr_recep == tar:
                    self.heated_objects.add(object['object_id'])
                    self.feedback = "You heat the %s using the %s." % (obj, tar)

            elif cmd['action'] == self.Action.CLEAN:
                obj, rel, tar = cmd['obj'], cmd['rel'], cmd['tar']

                # put the object in the sink, turn on the faucet, turn off the faucet, pickup the object
                events = []
                event, sink = self.interact('PutObject', tar)
                events.append(event)
                event, faucet = self.interact('ToggleObjectOn', 'faucet')
                events.append(event)
                event, _ = self.interact('Pass', '')
                events.append(event)
                event, faucet = self.interact('ToggleObjectOff', 'faucet')
                events.append(event)
                event, object = self.interact('PickupObject', obj)
                events.append(event)

                if all(e.metadata['lastActionSuccess'] for e in events) and self.curr_recep == tar:
                    self.cleaned_objects.add(object['object_id'])
                    self.feedback = "You clean the %s using the %s." % (obj, tar)

            elif cmd['action'] == self.Action.COOL:
                obj, rel, tar = cmd['obj'], cmd['rel'], cmd['tar']

                # open the fridge, put the object inside, close the fridge, open the fridge, pickup the object
                events = []
                event, fridge = self.interact('OpenObject', 'fridge')
                events.append(event)
                event, fridge = self.interact('PutObject', 'fridge')
                events.append(event)
                event, fridge = self.interact('CloseObject', 'fridge')
                events.append(event)
                event, _ = self.interact('Pass', '')
                events.append(event)
                event, fridge = self.interact('OpenObject', 'fridge')
                events.append(event)
                event, object = self.interact('PickupObject', obj)
                events.append(event)
                event, fridge = self.interact('CloseObject', 'fridge')
                events.append(event)

                if all(e.metadata['lastActionSuccess'] for e in events) and self.curr_recep == tar:
                    self.cooled_objects.add(object['object_id'])
                    self.feedback = "You cool the %s using the %s." % (obj, tar)

            elif cmd['action'] == self.Action.SLICE:
                obj, rel, tar = cmd['obj'], cmd['rel'], cmd['tar']
                if len(self.inventory) > 0 and 'knife' in self.inventory[0]:
                    event, object = self.interact('SliceObject', obj)
                if event and event.metadata['lastActionSuccess']:
                    self.feedback = "You slice %s with the %s" % (obj, tar)

            elif cmd['action'] == self.Action.INVENTORY:
                if len(self.inventory) > 0:
                    self.feedback = "You are carrying: %s" % (self.inventory[0])
                else:
                    self.feedback = "You are not carrying anything."

            elif cmd['action'] == self.Action.EXAMINE:
                target = cmd['tar']
                receptacle = self.get_object(target, self.receptacles)
                object = self.get_object(target, self.objects)

                if receptacle:
                    self.visible_objects, self.feedback = self.print_frame(receptacle, self.curr_loc)
                    self.frame_desc = str(self.feedback)
                elif object:
                    self.feedback = self.print_object(object)

            elif cmd['action'] == self.Action.LOOK:
                if self.curr_recep == "nothing":
                    self.feedback = "You are in the middle of a room. Looking quickly around you, you see nothing."
                else:
                    self.feedback = "You are facing the %s. Next to it, you see nothing." % self.curr_recep

        except:
            if self.debug:
                print(traceback.format_exc())

        if event and not event.metadata['lastActionSuccess']:
            self.feedback = "Nothing happens."

        if self.debug:
            print(self.feedback)
        return self.feedback




