import os
import io
import re
from sympy import im
import torch
from typing import Any, Tuple
import math
import numpy as np
from PIL import Image

def process_image(image: Any, max_pixels: int = 2048 * 2048, min_pixels: int = 512 * 512):
    import math
    from io import BytesIO
    from PIL import Image

    if isinstance(image, dict):
        image = Image.open(BytesIO(image['bytes']))

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)

    if image.width < 28 or image.height < 28:
        resize_factor = 28 / min(image.width, image.height)
        width, height = int(image.width * resize_factor + 1), int(image.height * resize_factor + 1)
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)

    if image.width / image.height >= 200:
        width, height = image.width, int(image.width / 190 + 1)
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)

    if image.height / image.width >= 200:
        width, height = int(image.height / 190 + 1), image.height
        image = image.resize((width, height), resample=Image.Resampling.NEAREST)

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image

def crop_image(image: Image.Image, coordinates: list, save_path=None):
    # Resize the image

    if coordinates[0] >= coordinates[2] or coordinates[1] >= coordinates[3]:
        print("coordinates: ", coordinates)

    image_crop = image.crop(coordinates)
    # crop_w, crop_h = image_crop.size

    # w, h = image.size
    # target_pixel=w*h
    # resize = np.sqrt(target_pixel / (crop_w * crop_h))
    # target_w, target_h = int(crop_w * resize), int(crop_h * resize)
    # target_w, target_h = max(28, target_w), max(28, target_h)
    # image_crop = image_crop.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)
    # if save_path:
    #     # Save the enlarged image
    #     image.save(save_path)
    #     image_crop.save(save_path.replace(".jpg", "_crop.jpg"))
    return image_crop

def prepare_grounding_inputs_multi_turn(json_objects: list, observations):
    assert len(json_objects) == 1
    obj = json_objects[0]
    assert "bbox_2d" in obj, f"There is no 'bbox_2d' in grounding argument: {obj}"
    assert "source" in obj, f"There is no 'source' in grounding argument: {obj}"
    assert isinstance(obj, dict) and isinstance(obj['bbox_2d'], list) and len(obj['bbox_2d']) == 4
    assert isinstance(obj['source'], str)

    bbox = obj['bbox_2d']
    source = obj['source']
    if source == "original_image":
        observation_id = 0
    else:
        pattern = "observation_([0-9]*)"
        match = re.match(pattern, source, re.DOTALL)
        try:
            observation_id = int(match.group(1))
        except Exception as e:
            raise ValueError(f"The 'source' argument \"{source}\" does not match the pattern 'observation_([0-9]*)'.")

    assert observation_id < len(observations), f"Error in predicting the source observation: \"{source}\"."
    image = observations[observation_id]

    w, h = image.size
    bbox = (bbox[0]*w, bbox[1]*h, bbox[2]*w, bbox[3]*h)
    # if use_relative_coordinates:
    #     w, h = image.size
    #     bbox = (bbox[0]*w, bbox[1]*h, bbox[2]*w, bbox[3]*h)
    # else:
    #     w, h = image.size
    #     w_used, h_used = image_size_used_list[observation_id]
    #     if w != w_used or h != h_used:
    #         multiplier_w = w / w_used
    #         multiplier_h = h / h_used
    #         bbox = (bbox[0]*multiplier_w, bbox[1]*multiplier_h, bbox[2]*multiplier_w, bbox[3]*multiplier_h)
    bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
    bbox = [max(bbox[0], 0), max(bbox[1], 0), min(bbox[2], w-1), min(bbox[3], h-1)]

    assert bbox[0] <= bbox[2] - 1, f"The bounding box is not valid in width: {bbox}"
    assert bbox[1] <= bbox[3] - 1, f"The bounding box is not valid in height: {bbox}"

    width_box = bbox[2] - bbox[0]
    height_box = bbox[3] - bbox[1]
    if width_box / height_box >= 200 or height_box / width_box >= 200:
        raise ValueError("The absolute aspect ratio of the bounding box exceeds 200, which is too large.")

    return image, bbox



def get_valid_mask(uids, multi_modal_data, meta_info, reward_tensor, acc_reward_tensor, format_reward_tensor, invalid_uids, exceed_mask=None):

    statistics_dict = {
        "solve_none": 0,
        "solve_all": 0,
        "solve_acc_none": 0,
        "solve_acc_all": 0,
        "solve_format_none": 0,
        "solve_format_all": 0,
        "totals": 0,
        "success_tool_call_per_batch": 0,
        "success_tool_call_per_sample": 0,
        "use_tool_and_direct_answer_num": 0,
        "wrong_and_use_tool_num": 0,
        "wrong_and_direct_answer_num": 0,
        "wrong_and_use_tool_and_direct_answer_num": 0,
        "only_use_tool_num": 0,
        "only_direct_answer_num": 0,
        "only_wrong_num": 0,
        "num_0_0": 0, # direct 全错，tool 全错,
        "num_0_1": 0, # direct 全错，tool 全对,
        "num_0_2": 0, # direct 全错，tool 有对有错,
        "num_0_k": 0, # direct 全错，tool 不存在,
        "num_1_0": 0,
        "num_1_1": 0,
        "num_1_2": 0,
        "num_1_k": 0,
        "num_2_0": 0,
        "num_2_1": 0,
        "num_2_2": 0,
        "num_2_k": 0,
        "num_k_0": 0,
        "num_k_1": 0,
        "num_k_2": 0,
    }
    unique_uids = np.unique(uids)
    valid_mask = torch.ones(len(uids), dtype=torch.bool)
    invalid_mask = torch.zeros(len(uids), dtype=torch.bool)

    wrong_and_use_tool_uids = []
    wrong_and_direct_answer_uids = []

    for uid in unique_uids:
        uid_mask = uids == uid

        uid_acc_rewards = acc_reward_tensor[uid_mask].sum(-1)
        uid_format_rewards = format_reward_tensor[uid_mask].sum(-1)

        if meta_info['tool_call'] in ['crop']:
            uid_multi_modal_data = multi_modal_data[uid_mask]
            success_tool_call_mask = torch.tensor([len(item['image']) > 1 for item in uid_multi_modal_data]).to(uid_acc_rewards.device)
            success_tool_call_count = success_tool_call_mask.sum().item()
            statistics_dict["success_tool_call_per_batch"] += success_tool_call_count
            statistics_dict["success_tool_call_per_sample"] += (success_tool_call_count > 0)
            # Compute Reward Distribution
            use_tool_correct_answer_mask = (uid_acc_rewards > 0) & success_tool_call_mask
            use_tool_correct_answer_count = use_tool_correct_answer_mask.sum().item()
            direct_answer_correct_answer_mask = (uid_acc_rewards > 0) & (~success_tool_call_mask)
            direct_answer_correct_answer_count = direct_answer_correct_answer_mask.sum().item()

            use_tool_wrong_answer_mask = (uid_acc_rewards == 0) & success_tool_call_mask
            use_tool_wrong_answer_count = use_tool_wrong_answer_mask.sum().item()
            direct_answer_wrong_answer_mask = (uid_acc_rewards == 0) & (~success_tool_call_mask)
            direct_answer_wrong_answer_count = direct_answer_wrong_answer_mask.sum().item()

            wrong_answer_count = (uid_acc_rewards == 0).sum().item()
            
            assert use_tool_wrong_answer_count + direct_answer_wrong_answer_count == wrong_answer_count

            if direct_answer_correct_answer_count == 0 and direct_answer_wrong_answer_count > 0 and use_tool_correct_answer_count == 0 and use_tool_wrong_answer_count > 0:
                statistics_dict["num_0_0"] += 1
            elif direct_answer_correct_answer_count == 0 and direct_answer_wrong_answer_count > 0 and use_tool_correct_answer_count > 0 and use_tool_wrong_answer_count == 0:
                statistics_dict["num_0_1"] += 1
            elif direct_answer_correct_answer_count == 0 and direct_answer_wrong_answer_count > 0 and use_tool_correct_answer_count > 0 and use_tool_wrong_answer_count > 0:
                statistics_dict["num_0_2"] += 1
            elif direct_answer_correct_answer_count == 0 and direct_answer_wrong_answer_count > 0 and use_tool_correct_answer_count == 0 and use_tool_wrong_answer_count == 0:
                statistics_dict["num_0_k"] += 1
            elif direct_answer_correct_answer_count > 0 and direct_answer_wrong_answer_count == 0 and use_tool_correct_answer_count == 0 and use_tool_wrong_answer_count > 0:
                statistics_dict["num_1_0"] += 1
            elif direct_answer_correct_answer_count > 0 and direct_answer_wrong_answer_count == 0 and use_tool_correct_answer_count > 0 and use_tool_wrong_answer_count == 0:
                statistics_dict["num_1_1"] += 1
            elif direct_answer_correct_answer_count > 0 and direct_answer_wrong_answer_count == 0 and use_tool_correct_answer_count > 0 and use_tool_wrong_answer_count > 0:
                statistics_dict["num_1_2"] += 1
            elif direct_answer_correct_answer_count > 0 and direct_answer_wrong_answer_count == 0 and use_tool_correct_answer_count == 0 and use_tool_wrong_answer_count == 0:
                statistics_dict["num_1_k"] += 1
            elif direct_answer_correct_answer_count > 0 and direct_answer_wrong_answer_count > 0 and use_tool_correct_answer_count == 0 and use_tool_wrong_answer_count > 0:
                statistics_dict["num_2_0"] += 1
            elif direct_answer_correct_answer_count > 0 and direct_answer_wrong_answer_count > 0 and use_tool_correct_answer_count > 0 and use_tool_wrong_answer_count == 0:
                statistics_dict["num_2_1"] += 1
            elif direct_answer_correct_answer_count > 0 and direct_answer_wrong_answer_count > 0 and use_tool_correct_answer_count > 0 and use_tool_wrong_answer_count > 0:
                statistics_dict["num_2_2"] += 1
            elif direct_answer_correct_answer_count > 0 and direct_answer_wrong_answer_count > 0 and use_tool_correct_answer_count == 0 and use_tool_wrong_answer_count == 0:
                statistics_dict["num_2_k"] += 1
            elif direct_answer_correct_answer_count == 0 and direct_answer_wrong_answer_count == 0 and use_tool_correct_answer_count == 0 and use_tool_wrong_answer_count > 0:
                statistics_dict["num_k_0"] += 1
            elif direct_answer_correct_answer_count == 0 and direct_answer_wrong_answer_count == 0 and use_tool_correct_answer_count > 0 and use_tool_wrong_answer_count == 0:
                statistics_dict["num_k_1"] += 1
            elif direct_answer_correct_answer_count == 0 and direct_answer_wrong_answer_count == 0 and use_tool_correct_answer_count > 0 and use_tool_wrong_answer_count > 0:
                statistics_dict["num_k_2"] += 1

            if use_tool_correct_answer_count > 0 and direct_answer_correct_answer_count > 0 and wrong_answer_count == 0:
                statistics_dict["use_tool_and_direct_answer_num"] += 1
            elif use_tool_correct_answer_count > 0 and direct_answer_correct_answer_count == 0 and wrong_answer_count > 0:
                statistics_dict["wrong_and_use_tool_num"] += 1
                wrong_and_use_tool_uids.append(uid)
            elif use_tool_correct_answer_count > 0 and direct_answer_correct_answer_count == 0 and wrong_answer_count == 0:
                statistics_dict["only_use_tool_num"] += 1
            elif use_tool_correct_answer_count == 0 and direct_answer_correct_answer_count > 0 and wrong_answer_count > 0:
                statistics_dict["wrong_and_direct_answer_num"] += 1
                wrong_and_direct_answer_uids.append(uid)
            elif use_tool_correct_answer_count == 0 and direct_answer_correct_answer_count > 0 and wrong_answer_count == 0:
                statistics_dict["only_direct_answer_num"] += 1
            elif use_tool_correct_answer_count == 0 and direct_answer_correct_answer_count == 0 and wrong_answer_count > 0:
                statistics_dict["only_wrong_num"] += 1
            elif use_tool_correct_answer_count > 0 and direct_answer_correct_answer_count > 0 and wrong_answer_count > 0:
                statistics_dict["wrong_and_use_tool_and_direct_answer_num"] += 1
            else:
                raise NotImplementedError(f"use_tool_correct_answer_count: {use_tool_correct_answer_count}, direct_answer_correct_answer_count: {direct_answer_correct_answer_count}, wrong_answer_count: {wrong_answer_count}.")

        if uid in invalid_uids:
            valid_mask[uid_mask] = False
            invalid_mask[uid_mask] = True
            continue

        uid_rewards = reward_tensor[uid_mask].sum(-1)  # Sum rewards for each sequence

        if exceed_mask is not None:
            old_uid_rewards = uid_rewards
            exceed_mask_uid = exceed_mask[uid_mask]
            uid_rewards = uid_rewards[~exceed_mask_uid]
            print(f"len(old_uid_rewards): {len(old_uid_rewards)}, len(uid_rewards): {len(uid_rewards)}")

        if uid_rewards.std().item() < 1e-4:
            valid_mask[uid_mask] = False
            invalid_mask[uid_mask] = True

        if (uid_rewards == 0).all():
            statistics_dict["solve_none"] += 1
        elif (uid_rewards == meta_info['acc_reward_weight'] + meta_info['format_reward_weight'] + meta_info['use_tool_reward_weight']).all():
            statistics_dict["solve_all"] += 1

        # Check if all rewards are 0 or all are 1 for this uid
        if (uid_acc_rewards == 0).all():
            statistics_dict["solve_acc_none"] += 1
        elif (uid_acc_rewards == meta_info['acc_reward_weight']).all():
            statistics_dict["solve_acc_all"] += 1

        if (uid_format_rewards == 0).all():
            statistics_dict["solve_format_none"] += 1
        elif (uid_format_rewards == meta_info['format_reward_weight']).all():
            statistics_dict["solve_format_all"] += 1

        statistics_dict["totals"] += 1
    return valid_mask, invalid_mask, wrong_and_use_tool_uids, wrong_and_direct_answer_uids, statistics_dict

