import copy
import random

import cv2
import numpy as np

from alfworld.gen import constants
from alfworld.gen import goal_library as glib


def get_pose(event):
    pose = event.pose
    return (int(np.round(pose[0] / (1000 * constants.AGENT_STEP_SIZE))),
            int(np.round(pose[1] / (1000 * constants.AGENT_STEP_SIZE))),
            int(np.round(pose[2] / (1000 * 90))),
            int(np.round(pose[3] / (1000))))


def get_object_data(metadata):
    return [
        {"objectName": obj["name"].split("(Clone)")[0], "position": obj["position"], "rotation": obj["rotation"]}
        for obj in metadata["objects"]
        if obj["pickupable"]
    ]


def imresize(image, size, rescale=True):
    if image is None:
        return None
    if image.shape[0] != size[0] or image.shape[1] != size[1]:
        image = cv2.resize(image, size)
    if rescale:
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        image /= 255.0
    return image


def depth_imresize(image, size, rescale=True, max_depth=constants.MAX_DEPTH):
    if image is None:
        return None
    if image.shape[0] != size[0] or image.shape[1] != size[1]:
        image = cv2.resize(image, size)
    image[image > max_depth] = max_depth
    if rescale:
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        image /= max_depth
    return image


def get_camera_matrix(pose, camera_height):
    assert(pose[2] in {0, 1, 2, 3})
    sin_x = np.sin(pose[3] * np.pi / 180)
    cos_x = np.cos(pose[3] * np.pi / 180)
    x_rotation = np.matrix([
        [1, 0, 0],
        [0, cos_x, -sin_x],
        [0, sin_x, cos_x]])
    sin_y = np.sin(pose[2] * np.pi / 180)
    cos_y = np.cos(pose[2] * np.pi / 180)
    y_rotation = np.matrix([
        [cos_y, 0, sin_y],
        [0, 1, 0],
        [-sin_y, 0, cos_y]])
    rotation_matrix = np.matmul(x_rotation, y_rotation)
    transformation_matrix = np.matrix([pose[0], camera_height, pose[1], 1]).T
    extrinsic_matrix = np.concatenate((np.concatenate((rotation_matrix, np.matrix([0, 0, 0])), axis=0),
                                       transformation_matrix), axis=1)
    return extrinsic_matrix


def get_rotation_matrix(pose):
    assert(pose[2] in {0, 1, 2, 3}), 'rotation was %s' % str(pose[2])
    sin_x = np.sin(-pose[3] * np.pi / 180)
    cos_x = np.cos(-pose[3] * np.pi / 180)
    x_rotation = np.matrix([
        [1, 0, 0],
        [0, cos_x, -sin_x],
        [0, sin_x, cos_x]], dtype=np.float32)
    sin_y = np.sin((-pose[2] % 4) * 90 * np.pi / 180)
    cos_y = np.cos((-pose[2] % 4) * 90 * np.pi / 180)
    y_rotation = np.matrix([
        [cos_y, 0, sin_y],
        [0, 1, 0],
        [-sin_y, 0, cos_y]], dtype=np.float32)
    rotation_matrix = np.matmul(x_rotation, y_rotation)
    return rotation_matrix


def depth_to_world_coordinates(depth, pose, camera_height):
    x_points = np.arange(-constants.SCREEN_WIDTH / 2, constants.SCREEN_WIDTH / 2, dtype=depth.dtype)
    x_vals = (depth * x_points / constants.FOCAL_LENGTH)

    y_points = np.arange(constants.SCREEN_HEIGHT / 2, -constants.SCREEN_HEIGHT / 2, -1, dtype=depth.dtype)
    y_vals = (depth.T * y_points / constants.FOCAL_LENGTH).T

    z_vals = depth
    xyz = np.stack((x_vals, y_vals, z_vals), axis=2) / (1000 * constants.AGENT_STEP_SIZE)
    rotation_matrix = np.linalg.inv(get_rotation_matrix(pose))
    xyz = np.array(np.dot(rotation_matrix, xyz.reshape(-1, 3).T).T).reshape(
        constants.SCREEN_HEIGHT, constants.SCREEN_WIDTH, 3)
    xzy = xyz[:, :, [0, 2, 1]]
    xzy += np.array([pose[0], pose[1], camera_height])
    return xzy


# coordinates should be [n, (xzy)]
def world_to_camera_coordinates(coordinates, pose, camera_height):
    coordinates = coordinates.copy()
    coordinates -= np.array([pose[0], pose[1], camera_height])
    xyz = coordinates[:, [0, 2, 1]]  # [n, (xyz)]
    rotation_matrix = get_rotation_matrix(pose)
    xyd = np.array(np.dot(rotation_matrix, xyz.T).T)
    xyd *= (1000 * constants.AGENT_STEP_SIZE)
    depth = np.maximum(xyd[:, -1], 0.01)
    x_points = xyd[:, 0] * constants.FOCAL_LENGTH / depth + constants.SCREEN_WIDTH / 2
    y_points = constants.SCREEN_HEIGHT - (xyd[:, 1] * constants.FOCAL_LENGTH / depth + constants.SCREEN_HEIGHT / 2)
    return np.stack((x_points, y_points, depth)).T


def get_templated_action_str(plan, idx=0):
    action = copy.deepcopy(plan[idx])
    object_name, recep_name, prev_object_name, prev_recep_name, next_object_name, next_recep_name = get_relevant_objs(action, plan, idx)

    a_type = action['action']
    templated_str = ""

    if 'GotoLocation' in a_type:
        templated_str = "go to the %s" % (next_recep_name if next_recep_name != "" else prev_object_name)
    elif 'OpenObject' in a_type:
        templated_str = "open the %s" % (object_name)
    elif 'CloseObject' in a_type:
        templated_str = "close the %s" % (object_name)
    elif 'PickupObject' in a_type:
        templated_str = "pick up the %s" % (object_name)
    elif 'PutObject' in a_type:
        templated_str = "put the %s in the %s" % (object_name, recep_name)
    elif 'CleanObject' in a_type:
        templated_str = "wash the %s" % (prev_object_name)
    elif 'HeatObject' in a_type:
        templated_str = "heat the %s" % (prev_object_name)
    elif 'CoolObject' in a_type:
        templated_str = "cool the %s" % (prev_object_name)
    elif 'ToggleObject' in a_type:
        templated_str = "toggle %s" % (object_name)
    elif 'SliceObject' in a_type:
        templated_str = "slice the %s" % (object_name)
    elif 'End' in a_type:
        templated_str = "<<STOP>>"

    return templated_str


def get_discrete_hl_action(plan, idx=0):
    action = copy.deepcopy(plan[idx])
    object_name, recep_name, prev_object_name, prev_recep_name, next_object_name, next_recep_name = get_relevant_objs(action, plan, idx)

    a_type = action['action']
    discrete_action = {'action': "", 'args': []}

    if 'GotoLocation' in a_type:
        discrete_action['action'] = "GotoLocation"
        discrete_action['args'] = [next_recep_name if next_recep_name != "" else next_object_name]
    elif 'OpenObject' in a_type:
        discrete_action['action'] = "OpenObject"
        discrete_action['args'] = [object_name]
    elif 'CloseObject' in a_type:
        discrete_action['action'] = "CloseObject"
        discrete_action['args'] = [object_name]
    elif 'PickupObject' in a_type:
        discrete_action['action'] = "PickupObject"
        discrete_action['args'] = [object_name]
    elif 'PutObject' in a_type:
        discrete_action['action'] = "PutObject"
        discrete_action['args'] = [object_name, recep_name]
    elif 'CleanObject' in a_type:
        discrete_action['action'] = "CleanObject"
        discrete_action['args'] = [prev_object_name]
    elif 'HeatObject' in a_type:
        discrete_action['action'] = "HeatObject"
        discrete_action['args'] = [prev_object_name]
    elif 'CoolObject' in a_type:
        discrete_action['action'] = "CoolObject"
        discrete_action['args'] = [prev_object_name]
    elif 'ToggleObject' in a_type:
        discrete_action['action'] = "ToggleObject"
        discrete_action['args'] = [object_name]
    elif 'SliceObject' in a_type:
        discrete_action['action'] = "SliceObject"
        discrete_action['args'] = [object_name]
    else:
        discrete_action['action'] = "NoOp"
        discrete_action['args'] = []

    return discrete_action


def object_id_to_name(object_id):
    return object_id.split('|')[0]


def get_relevant_objs(action, plan, idx=0):
    object_name = object_id_to_name(action['objectId']).lower() if 'objectId' in action else ""
    recep_name = object_id_to_name(action['receptacleObjectId']).lower() if 'receptacleObjectId' in action else ""
    prev_object_name, prev_recep_name = "", ""
    next_object_name, next_recep_name = "", ""

    prev_idx = idx - 2
    if prev_idx >= 0:
        prev_action = copy.deepcopy(plan[prev_idx])
        prev_object_name = object_id_to_name(prev_action['objectId']).lower() if 'objectId' in prev_action else ""
        prev_recep_name = object_id_to_name(prev_action['receptacleObjectId']).lower() if 'receptacleObjectId' in prev_action else ""

    next_idx = idx + 1
    if next_idx < len(plan):
        next_action = copy.deepcopy(plan[next_idx])
        next_object_name = object_id_to_name(next_action['objectId']).lower() if 'objectId' in next_action else ""
        next_recep_name = object_id_to_name(next_action['receptacleObjectId']).lower() if 'receptacleObjectId' in next_action else ""

    return object_name, recep_name, prev_object_name, prev_recep_name, next_object_name, next_recep_name


def get_action_str(action):
    action = copy.deepcopy(action)
    a_type = action['action']
    action_str = 'Action: ' + a_type
    del action['action']

    if 'Teleport' in a_type:
        action_str = a_type
        if 'x' in action:
            action_str += ' x: %.03f' % action['x']
            del action['x']
        if 'y' in action:
            action_str += ' y: %.03f' % action['y']
            del action['y']
        if 'z' in action:
            action_str += ' z: %.03f' % action['z']
            del action['z']
        if 'rotation' in action and action.get('rotateOnTeleport', False):
            if type(action['rotation']) == dict:
                action_str += ' r: %d' % int(action['rotation']['y'])
            else:
                action_str += ' r: %d' % int(action['rotation'])
            del action['rotation']
            del action['rotateOnTeleport']
        if 'horizon' in action:
            action_str += ' h: %d' % int(action['horizon'])
            del action['horizon']
    elif 'Goto' in a_type:
        action_str = a_type
        if 'location' in action:
            action_str += ' loc: %s' % action['location']
            del action['location']
    elif a_type in {'OpenObject', 'CloseObject', 'PickupObject', 'ToggleObject', 'SliceObject'}:
        if 'objectId' not in action:
            action['objectId'] = 'None'
        action_str = '%s %s' % (a_type, action['objectId'])
    elif a_type in {'RotateByDegree', 'LookByDegree'}:
        if type(action['rotation']) == dict:
            action_str += ' r: %d' % int(action['rotation']['y'])
        else:
            action_str += ' r: %d' % int(action['rotation'])
        action_str = '%s %d' % (a_type, action['rotation']['y'])
        del action['rotation']
    elif a_type == 'PutObject':
        action_str = a_type
        if 'objectId' in action:
            action_str += ' o: %s' % action['objectId']
            del action['objectId']
        if 'receptacleObjectId' in action:
            action_str += ' r: %s' % action['receptacleObjectId']
            del action['receptacleObjectId']

    if len(action) > 0:
        action_str += '\tFull: ' + str(action)
    return action_str


def get_object(object_id, metadata):
    for obj in metadata['objects']:
        if obj['objectId'] == object_id:
            return obj
    return None


def get_object_dict(metadata):
    return {obj['objectId']: obj for obj in metadata['objects']}


def get_objects_of_type(object_type, metadata):
    return [obj for obj in metadata['objects'] if obj['objectType'] == object_type]


def get_obj_of_type_closest_to_obj(object_type, ref_object_id, metadata):
    objs_of_type = [obj for obj in metadata['objects'] if obj['objectType'] == object_type and obj['visible']]
    ref_obj = get_object(ref_object_id, metadata)
    closest_objs_of_type = sorted(objs_of_type, key=lambda o: np.linalg.norm(np.array([o['position']['x'], o['position']['y'], o['position']['z']]) - \
                                                                             np.array([ref_obj['position']['x'], ref_obj['position']['y'], ref_obj['position']['z']])))
    if len(closest_objs_of_type) == 0:
        raise Exception("No closest %s found!" % (ref_obj))
    return closest_objs_of_type[0] # retrun the first closest visible object


def get_objects_with_name_and_prop(name, prop, metadata):
    return [obj for obj in metadata['objects']
            if name in obj['objectId'] and obj[prop]]


def get_visible_objs(objs):
    return [obj for obj in objs if obj['visible']]


def get_object_bounds(obj, scene_bounds):
    # obj_bounds = np.array(obj['bounds3D'])[[0, 2, 3, 5]]  # Get X and Z out
    # Get a 'box' that is a singular point in (x,z) based on object position in place of now-unavailable 'bounds3d'
    obj_bounds = np.array([obj['position']['x'], obj['position']['z'], obj['position']['x'], obj['position']['z']])
    obj_bounds /= constants.AGENT_STEP_SIZE
    obj_bounds = np.round(obj_bounds).astype(np.int32)
    obj_bounds[[2, 3]] = np.maximum(obj_bounds[[2, 3]], obj_bounds[[0, 1]] + 1)
    obj_bounds[[0, 2]] = np.clip(obj_bounds[[0, 2]], scene_bounds[0], scene_bounds[0] + scene_bounds[2])
    obj_bounds[[1, 3]] = np.clip(obj_bounds[[1, 3]], scene_bounds[1], scene_bounds[1] + scene_bounds[3])
    obj_bounds -= np.array(scene_bounds)[[0, 1, 0, 1]]
    return obj_bounds


def get_object_bounds_batch(boxes, scene_bounds):
    obj_bounds = boxes[:, [0, 2, 3, 5]]  # Get X and Z out
    obj_bounds /= constants.AGENT_STEP_SIZE
    obj_bounds = np.round(obj_bounds).astype(np.int32)
    obj_bounds[:, [2, 3]] = np.maximum(obj_bounds[:, [2, 3]], obj_bounds[:, [0, 1]] + 1)
    obj_bounds[:, [0, 2]] = np.clip(obj_bounds[:, [0, 2]], scene_bounds[0], scene_bounds[0] + scene_bounds[2])
    obj_bounds[:, [1, 3]] = np.clip(obj_bounds[:, [1, 3]], scene_bounds[1], scene_bounds[1] + scene_bounds[3])
    obj_bounds -= np.array(scene_bounds)[[0, 1, 0, 1]]
    return obj_bounds


def get_task_str(task_type_ind, object_ind, receptacle_ind=None, toggle_ind=None, mrecep_ind=None):
    goal_str = constants.pddl_goal_type
    if constants.data_dict['pddl_params']['object_sliced']:
        goal_str += "_slice"
    template = random.choice(glib.gdict[goal_str]['templates'])
    obj = constants.OBJECTS[object_ind].lower()
    recep = constants.OBJECTS[receptacle_ind].lower() if receptacle_ind is not None else ""
    tog = constants.OBJECTS[toggle_ind].lower() if toggle_ind is not None else ""
    mrecep = constants.OBJECTS[mrecep_ind].lower() if mrecep_ind is not None else ""
    filled_in_str = template.format(obj=obj, recep=recep, toggle=tog, mrecep=mrecep)
    return filled_in_str


def get_last_hl_action_index():
    return len(constants.data_dict['plan']['high_pddl']) - 1


def get_last_ll_action_index():
    return len(constants.data_dict['plan']['low_actions']) - 1


def store_image_name(name):
    constants.data_dict['images'].append({"high_idx": get_last_hl_action_index(),
                                          "low_idx": get_last_ll_action_index(),
                                          "image_name": name})


