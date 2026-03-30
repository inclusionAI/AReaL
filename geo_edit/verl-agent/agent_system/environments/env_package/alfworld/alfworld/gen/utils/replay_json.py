import json

def replay_json(env, json_file):
    # load json data
    with open(json_file) as f:
        traj_data = json.load(f)

    # setup
    scene_num = traj_data['scene']['scene_num']
    object_poses = traj_data['scene']['object_poses']
    dirty_and_empty = traj_data['scene']['dirty_and_empty']
    object_toggles = traj_data['scene']['object_toggles']

    scene_name = 'FloorPlan%d' % scene_num
    env.reset(scene_name)
    env.restore_scene(object_poses, object_toggles, dirty_and_empty)

    # initialize
    event = env.step(dict(traj_data['scene']['init_action']))
    print("Task: %s" % (traj_data['template']['task_desc']))

    steps_taken = 0
    for ll_action in traj_data['plan']['low_actions']:
        hl_action_idx, traj_api_cmd, traj_discrete_action = \
            ll_action['high_idx'], ll_action['api_action'], ll_action['discrete_action']

        # print templated low-level instructions & discrete action
        print("HL Templ: %s, LL Cmd: %s" % (traj_data['template']['high_descs'][hl_action_idx],
                                            traj_discrete_action['action']))

        # Use the va_interact that modelers will have to use at inference time.
        action_name, action_args = traj_discrete_action['action'], traj_discrete_action['args']

        # three ways to specify object of interest mask
        # 1. create a rectangular mask from bbox
        # mask = env.bbox_to_mask(action_args['bbox']) if 'bbox' in action_args else None  # some commands don't require any arguments
        # 2. create a point mask from bbox
        # mask = env.point_to_mask(action_args['point']) if 'point' in action_args else None
        # 3. use full pixel-wise segmentation mask
        compressed_mask = action_args['mask'] if 'mask' in action_args else None
        if compressed_mask is not None:
            mask = env.decompress_mask(compressed_mask)
        else:
            mask = None

        success, event, target_instance_id, err, _ = env.va_interact(action_name, interact_mask=mask)
        if not success:
            raise RuntimeError(err)

        steps_taken += 1

    return steps_taken
