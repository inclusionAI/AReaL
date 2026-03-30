import json
import os
import threading
import time

import cv2
import numpy as np

import constants
from utils import game_util

N_PROCS = 40

lock = threading.Lock()
all_scene_numbers = sorted(constants.TRAIN_SCENE_NUMBERS + constants.TEST_SCENE_NUMBERS, reverse=True)


def get_obj(env, open_test_objs, reachable_points, agent_height, scene_name, good_obj_point):

    # Reset the scene to put all the objects back where they started.
    game_util.reset(env, scene_name,
                    render_image=False,
                    render_depth_image=False,
                    render_class_image=False,
                    render_object_image=True)

    if good_obj_point is not None:
        search_points = {good_obj_point[0]}
    else:
        search_points = reachable_points

    for point in search_points:
        for rotation in range(4):
            if good_obj_point is not None and rotation != good_obj_point[1]:
                continue
            for horizon in [-30, 0, 30]:
                if good_obj_point is not None and horizon != good_obj_point[2]:
                    continue

                action = {'action': 'TeleportFull',
                          'x': point[0],
                          'y': agent_height,
                          'z': point[1],
                          'rotateOnTeleport': True,
                          'rotation': rotation * 90,
                          'horizon': horizon
                          }
                event = env.step(action)
                if event.metadata['lastActionSuccess']:

                    # Close everything.
                    for obj in event.metadata['objects']:
                        if (obj['visible'] and obj['objectId'] and obj['isOpen'] and
                                obj['objectType'] in constants.VAL_RECEPTACLE_OBJECTS):
                            action = {'action': 'CloseObject',
                                      'objectId': obj['objectId']}
                            event = env.step(action)

                    # Try to pick up a valid inv object.
                    for obj in event.metadata['objects']:
                        if obj['visible'] and obj['objectType'] in open_test_objs:
                            action = {'action': 'PickupObject',
                                      'objectId': obj['objectId']}
                            event = env.step(action)
                            if event.metadata['lastActionSuccess']:
                                good_obj_point = (point, rotation, horizon)
                                return good_obj_point

                    # Open everything.
                    for open_obj in event.metadata['objects']:
                        if (open_obj['visible'] and open_obj['objectId'] and open_obj['openable'] and
                                not open_obj['pickupable'] and
                                open_obj['objectType'] in constants.VAL_RECEPTACLE_OBJECTS):
                            action = {'action': 'OpenObject',
                                      'objectId': open_obj['objectId']}
                            event = env.step(action)
                            if event.metadata['lastActionSuccess']:

                                # Try to pick up a valid inv object.
                                for obj in event.metadata['objects']:
                                    if obj['visible'] and obj['objectType'] in open_test_objs:
                                        action = {'action': 'PickupObject',
                                                  'objectId': obj['objectId']}
                                        event = env.step(action)
                                        if event.metadata['lastActionSuccess']:
                                            good_obj_point = (point, rotation, horizon)
                                            return good_obj_point

                                # Close open_obj.
                                action = {'action': 'CloseObject',
                                          'objectId': open_obj['objectId']}
                                event = env.step(action)
    return None


# Derived from function of the same name in game_states/game_state_base.py
def get_mask_of_obj(env, object_id):
    instance_detections2D = env.last_event.instance_detections2D
    instance_seg_frame = np.array(env.last_event.instance_segmentation_frame)

    if object_id in instance_detections2D:
        color = env.last_event.object_id_to_color[object_id]
        seg_mask = cv2.inRange(instance_seg_frame, color, color)
        seg_mask = np.array(seg_mask) / 255
        seg_mask = seg_mask.astype(int)
        return np.sum(seg_mask)
    else:
        return None


def run():
    print(all_scene_numbers)
    # create env and agent
    env = game_util.create_env(build_path=constants.BUILD_PATH,
                               quality='Low')
    while len(all_scene_numbers) > 0:
        lock.acquire()
        scene_num = all_scene_numbers.pop()
        lock.release()
        fn = os.path.join('layouts', ('FloorPlan%d-layout.npy') % scene_num)
        if os.path.isfile(fn):
            print("file %s already exists; skipping this floorplan" % fn)
            continue

        openable_json_file = os.path.join('layouts', ('FloorPlan%d-openable.json') % scene_num)
        scene_objs_json_file = os.path.join('layouts', ('FloorPlan%d-objects.json') % scene_num)

        scene_name = ('FloorPlan%d') % scene_num
        print('Running ' + scene_name)
        event = game_util.reset(env, scene_name,
                                render_image=False,
                                render_depth_image=False,
                                render_class_image=False,
                                render_object_image=True)
        agent_height = event.metadata['agent']['position']['y']

        scene_objs = list(set([obj['objectType'] for obj in event.metadata['objects']]))
        with open(scene_objs_json_file, 'w') as sof:
            json.dump(scene_objs, sof, sort_keys=True, indent=4)

        # Get all the reachable points through Unity for this step size.
        event = env.step(dict(action='GetReachablePositions',
                              gridSize=constants.AGENT_STEP_SIZE / constants.RECORD_SMOOTHING_FACTOR))
        if event.metadata['actionReturn'] is None:
            print("ERROR: scene %d 'GetReachablePositions' returns None" % scene_num)
        else:
            reachable_points = set()
            for point in event.metadata['actionReturn']:
                reachable_points.add((point['x'], point['z']))
            print("scene %d got %d reachable points, now checking" % (scene_num, len(reachable_points)))

            # Pick up a small object to use in testing whether points are good for openable objects.
            open_test_objs = {'ButterKnife', 'CD', 'CellPhone', 'Cloth', 'CreditCard', 'DishSponge', 'Fork',
                              'KeyChain', 'Pen', 'Pencil', 'SoapBar', 'Spoon', 'Watch'}
            good_obj_point = None
            good_obj_point = get_obj(env, open_test_objs, reachable_points, agent_height, scene_name, good_obj_point)


            best_open_point = {}  # map from object names to the best point from which they can be successfully opened
            best_sem_coverage = {}  # number of pixels in the semantic map of the receptacle at the existing best openpt
            checked_points = set()
            scene_receptacles = set()
            for point in reachable_points:
                point_is_valid = True
                action = {'action': 'TeleportFull',
                          'x': point[0],
                          'y': agent_height,
                          'z': point[1],
                          }
                event = env.step(action)
                if event.metadata['lastActionSuccess']:
                    for horizon in [-30, 0, 30]:
                        action = {'action': 'TeleportFull',
                                  'x': point[0],
                                  'y': agent_height,
                                  'z': point[1],
                                  'rotateOnTeleport': True,
                                  'rotation': 0,
                                  'horizon': horizon
                                  }
                        event = env.step(action)
                        if not event.metadata['lastActionSuccess']:
                            point_is_valid = False
                            break
                        for rotation in range(3):
                            action = {'action': 'RotateLeft'}
                            event = env.step(action)
                            if not event.metadata['lastActionSuccess']:
                                point_is_valid = False
                                break
                        if not point_is_valid:
                            break
                    if point_is_valid:
                        checked_points.add(point)
                    else:
                        continue

                    # Check whether we can open objects from here in any direction with any tilt.
                    for rotation in range(4):
                        # First try up, then down, then return to the horizon before moving again.
                        for horizon in [-30, 0, 30]:

                            action = {'action': 'TeleportFull',
                                      'x': point[0],
                                      'y': agent_height,
                                      'z': point[1],
                                      'rotateOnTeleport': True,
                                      'rotation': rotation * 90,
                                      'horizon': horizon
                                      }
                            event = env.step(action)
                            for obj in event.metadata['objects']:
                                if (obj['visible'] and obj['objectId'] and obj['receptacle'] and not obj['pickupable']
                                        and obj['objectType'] in constants.VAL_RECEPTACLE_OBJECTS):
                                    obj_name = obj['objectId']
                                    obj_point = (obj['position']['x'], obj['position']['y'])
                                    scene_receptacles.add(obj_name)

                                    # Go ahead and attempt to close the object from this position if it's open.
                                    if obj['openable'] and obj['isOpen']:
                                        close_action = {'action': 'CloseObject',
                                                        'objectId': obj['objectId']}
                                        event = env.step(close_action)

                                    point_to_recep = np.linalg.norm(np.array(point) - np.array(obj_point))
                                    if len(env.last_event.metadata['inventoryObjects']) > 0:
                                        inv_obj = env.last_event.metadata['inventoryObjects'][0]['objectId']
                                    else:
                                        inv_obj = None

                                    # Heuristic implemented in task_game_state has agent 0.5 or farther in agent space.
                                    heuristic_far_enough_from_recep = 0.5 < point_to_recep
                                    # Ensure this point affords a larger view according to the semantic segmentation
                                    # of the receptacle than the existing.
                                    point_sem_coverage = get_mask_of_obj(env, obj['objectId'])
                                    if point_sem_coverage is None:
                                        use_sem_heuristic = False
                                        better_sem_covereage = False
                                    else:
                                        use_sem_heuristic = True
                                        better_sem_covereage = (obj_name not in best_sem_coverage or
                                                                best_sem_coverage[obj_name] is None or
                                                                point_sem_coverage > best_sem_coverage[obj_name])
                                    # Ensure that this point is farther away than our existing best candidate.
                                    # We'd like to open each receptacle from as far away as possible while retaining
                                    # the ability to pick/place from it.
                                    farther_than_existing_good_point = (obj_name not in best_open_point or
                                                                        point_to_recep >
                                                                        np.linalg.norm(
                                                                            np.array(point) -
                                                                            np.array(best_open_point[obj_name][:2])))
                                    # If we don't have an inventory object, though, we'll fall back to the heuristic
                                    # of being able to open/close as _close_ as possible.
                                    closer_than_existing_good_point = (obj_name not in best_open_point or
                                                                        point_to_recep <
                                                                        np.linalg.norm(
                                                                            np.array(point) -
                                                                            np.array(best_open_point[obj_name][:2])))
                                    # Semantic segmentation heuristic.
                                    if ((use_sem_heuristic and heuristic_far_enough_from_recep and better_sem_covereage)
                                            or (not use_sem_heuristic and
                                                # Distance heuristics.
                                                (heuristic_far_enough_from_recep and
                                                 (inv_obj and farther_than_existing_good_point) or
                                                 (not inv_obj and closer_than_existing_good_point)))):
                                        if obj['openable']:
                                            action = {'action': 'OpenObject',
                                                      'objectId': obj['objectId']}
                                            event = env.step(action)
                                        if not obj['openable'] or event.metadata['lastActionSuccess']:
                                            # We can open the object, so try placing our small inventory obj inside.
                                            # If it can be placed inside and retrieved, then this is a safe point.
                                            action = {'action': 'PutObject',
                                                      'objectId': inv_obj,
                                                      'receptacleObjectId': obj['objectId'],
                                                      'forceAction': True,
                                                      'placeStationary': True}
                                            if inv_obj:
                                                event = env.step(action)
                                            if inv_obj is None or event.metadata['lastActionSuccess']:
                                                action = {'action': 'PickupObject',
                                                          'objectId': inv_obj}
                                                if inv_obj:
                                                    event = env.step(action)
                                                if inv_obj is None or event.metadata['lastActionSuccess']:

                                                    # Finally, ensure we can also close the receptacle.
                                                    if obj['openable']:
                                                        action = {'action': 'CloseObject',
                                                                  'objectId': obj['objectId']}
                                                        event = env.step(action)
                                                    if not obj['openable'] or event.metadata['lastActionSuccess']:

                                                        # We can put/pick our inv object into the receptacle from here.
                                                        # We have already ensured this point is farther than any
                                                        # existing best, so this is the new best.
                                                        best_open_point[obj_name] = [point[0], point[1], rotation * 90, horizon]
                                                        best_sem_coverage[obj_name] = point_sem_coverage

                                                # We could not retrieve our inv object, so we need to go get another one
                                                else:
                                                    good_obj_point = get_obj(env, open_test_objs, reachable_points,
                                                                             agent_height, scene_name, good_obj_point)
                                                    action = {'action': 'TeleportFull',
                                                              'x': point[0],
                                                              'y': agent_height,
                                                              'z': point[1],
                                                              'rotateOnTeleport': True,
                                                              'rotation': rotation * 90,
                                                              'horizon': horizon
                                                              }
                                                    event = env.step(action)

                                    # Regardless of what happened up there, try to close the receptacle again if
                                    # it remained open.
                                    if obj['isOpen']:
                                        action = {'action': 'CloseObject',
                                                  'objectId': obj['objectId']}
                                        event = env.step(action)

            essential_objs = []
            if scene_num in constants.SCENE_TYPE["Kitchen"]:
                essential_objs.extend(["Microwave", "Fridge"])
            for obj in essential_objs:
                if not np.any([obj in obj_key for obj_key in best_open_point]):
                    print("WARNING: Essential object %s has no open points in scene %d" % (obj, scene_num))

            print("scene %d found open/pick/place/close positions for %d/%d receptacle objects" %
                  (scene_num, len(best_open_point), len(scene_receptacles)))
            with open(openable_json_file, 'w') as f:
                json.dump(best_open_point, f, sort_keys=True, indent=4)

            print("scene %d reachable %d, checked %d; taking intersection" %
                  (scene_num, len(reachable_points), len(checked_points)))

            points = np.array(list(checked_points))[:, :2]
            points = points[np.lexsort((points[:, 0], points[:, 1])), :]
            np.save(fn, points)

    env.stop()
    print('Done')


threads = []
for n in range(N_PROCS):
    thread = threading.Thread(target=run)
    threads.append(thread)
    thread.start()
    time.sleep(1)
