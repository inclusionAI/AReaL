# Copyright ByteDance, The University of Hong Kong

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0


import re

def is_valid_direct_answer(response, direct_answer_format) -> bool:
    """
    对 <think>...</think><answer>...</answer> 的形式进行校验：
      1) 是否整体匹配大体结构
      2) 是否只出现一次 <think>...</think> 和 <answer>...</answer>
      3) 不应包含 <grounding> </grounding>
    """
    pattern = direct_answer_format
    # 1). Structure Matching
    if not re.match(pattern, response, re.DOTALL):
        return False
    # 2). Pattern Count
    if response.count('<think>') != 1 or response.count('</think>') != 1:
        return False
    if response.count('<answer>') != 1 or response.count('</answer>') != 1:
        return False
    # 3). <grounding> </grounding> is not allowed!
    if '<grounding>' in response or '</grounding>' in response:
        return False
    return True

def is_valid_direct_answer_grounding(response, direct_answer_format) -> bool:
    """
    对 <think>...</think><answer>...</answer> 的形式进行校验：
      1) 是否整体匹配大体结构
      2) 是否只出现一次 <think>...</think> 和 <answer>...</answer>
      3) 不应包含 <grounding> </grounding>
    """
    pattern = direct_answer_format
    # 1). Structure Matching
    if not re.match(pattern, response, re.DOTALL):
        return False
    # 2). Pattern Count
    if response.count('<think>') != 1 or response.count('</think>') != 1:
        return False
    if response.count('<answer>') != 1 or response.count('</answer>') != 1:
        return False
    # 3). <tool_call> </tool_call> is not allowed!
    if '<grounding>' in response or '</grounding>' in response:
        return False
    return True

def is_valid_tool_call(response, step_tool_call_format) -> bool:
    """
    对 <think>...</think>...<grounding>...</grounding> 的形式进行校验：
      1) 整体正则匹配
      2) <think>...</think> 各出现一次
      3) <grounding>...</grounding> 只出现一次
      4) 不应出现 <answer> </answer>
    """
    pattern = step_tool_call_format
    # 1). Structure Matching
    if not re.match(pattern, response, re.DOTALL):
        return False
    # 2). <think> Count
    if response.count('<think>') != 1 or response.count('</think>') != 1:
        return False
    # 3). <grounding> </grounding> Count
    if response.count('<grounding>') != 1 and response.count('</grounding>') != 1:
        return False
    # 4). <answer> or </answer> is not allowed!
    if '<answer>' in response or '</answer>' in response:
        return False
    return True

def is_valid_tool_call_grounding(response, step_tool_call_format) -> bool:
    """
    对 <think>...</think>...<grounding>...</grounding> 的形式进行校验：
      1) 整体正则匹配
      2) <think>...</think> 各出现一次
      3) <grounding>...</grounding> 只出现一次
      4) 不应出现 <answer> </answer>
    """
    pattern = step_tool_call_format
    # 1). Structure Matching
    if not re.match(pattern, response, re.DOTALL):
        return False
    # 2). <think> Count
    if response.count('<think>') != 1 or response.count('</think>') != 1:
        return False
    # 3). <grounding> </grounding> Count
    if response.count('<grounding>') != 1 and response.count('</grounding>') != 1:
        return False
    # 4). <answer> or </answer> is not allowed!
    if '<answer>' in response or '</answer>' in response:
        return False
    return True

def format_reward(predict_str_list: list, extra_info: dict = None):
    """
    Check if the model's response follows the required formats and return a reward.
    [1-turn]:
        - Direct Answer
    [2-turn]:
        - Call Image Resize Tool + Answer
    Args:
    - predict_str_list (list): A list of responses, currently, max length of `predict_str_list` is 10 (10-turn), max image num is 2.
    Returns:
    - format_score: float, 1.0 for right format, 0.0 for wrong
    - tool_call_count: int, times of function tools called
    """
    conv_rounds = len(predict_str_list)
    format_score, tool_call_count = 0, 0
    # All allowed formats
    direct_answer_format = r'^<think>.*</think>.*<answer>.*</answer>$'
    step_tool_call_format = r'^<think>.*</think>.*<grounding>.*</grounding>$'
    tool_call_pattern = re.compile(r'<grounding>(.*?)</grounding>', re.DOTALL)
    # HACK/FIXME: We need more flexible judge in the future
    # 1-turn
    if conv_rounds == 1:
        response = predict_str_list[0].strip()
        tool_call_contents = tool_call_pattern.findall(response)
        if len(tool_call_contents) > 0:
            tool_call_count += 1
        # Direct Answer
        if is_valid_direct_answer(response, direct_answer_format):
            format_score = 1
    # multi-turn
    else:
        tool_call_match_flag = True
        for response in predict_str_list[:-1]:
            response = response.strip()
            tool_call_contents = tool_call_pattern.findall(response)
            if len(tool_call_contents) > 0:
                tool_call_count += 1
            # Call Function Tool
            if not is_valid_tool_call(response, step_tool_call_format):
                tool_call_match_flag = False
                break
        final_answer_match_flag = is_valid_direct_answer(predict_str_list[-1], direct_answer_format)
        if tool_call_match_flag and final_answer_match_flag:
            format_score = 1
    return format_score, tool_call_count


def grounding_format_reward(predict_str_list: list, extra_info: dict = None):
    """
    Check if the model's response follows the required formats and return a reward.
    [1-turn]:
        - Direct Answer
    [2-turn]:
        - Call Image Resize Tool + Answer
    Args:
    - predict_str_list (list): A list of responses, currently, max length of `predict_str_list` is 10 (10-turn), max image num is 2.
    Returns:
    - format_score: float, 1.0 for right format, 0.0 for wrong
    - tool_call_count: int, times of function tools called
    """
    conv_rounds = len(predict_str_list)
    format_score, tool_call_count = 0, 0
    # All allowed formats
    direct_answer_format = r'^<think>.*</think>.*<answer>.*</answer>$'
    step_tool_call_format = r'^<think>.*</think>.*<grounding>.*</grounding>$'
    tool_call_pattern = re.compile(r'<grounding>(.*?)</grounding>', re.DOTALL)
    # HACK/FIXME: We need more flexible judge in the future
    # 1-turn

    # print("conv_rounds: ", conv_rounds)

    if conv_rounds == 1:
        response = predict_str_list[0].strip()
        tool_call_contents = tool_call_pattern.findall(response)
        if len(tool_call_contents) > 0:
            tool_call_count += 1
        # Direct Answer
        if is_valid_direct_answer_grounding(response, direct_answer_format):
            format_score = 1
    # multi-turn
    else:
        tool_call_match_flag = True
        for response in predict_str_list[:-1]:
            response = response.strip()
            tool_call_contents = tool_call_pattern.findall(response)
            if len(tool_call_contents) > 0:
                tool_call_count += 1
            # Call Function Tool
            if not is_valid_tool_call_grounding(response, step_tool_call_format):
                tool_call_match_flag = False
                break
        final_answer_match_flag = is_valid_direct_answer_grounding(predict_str_list[-1], direct_answer_format)
        if tool_call_match_flag and final_answer_match_flag:
            format_score = 1

        # import pdb; pdb.set_trace()

    return format_score, tool_call_count


def inner_acc_reward(predict_str_list: list, original_answer: str, use_gpt=False, gpt_extract_answer=False, extra_info=None):

    original_answer = original_answer.strip()
    if isinstance(predict_str_list, list):
        predict_str = predict_str_list[-1].strip()
    else:
        assert isinstance(predict_str_list, str)
        predict_str = predict_str_list.strip()
    
    # extract_answer_pattern = r'<answer>(.*?)</answer>'
    # match = re.search(extract_answer_pattern, predict_str, re.DOTALL)
    extract_answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
    matches = extract_answer_pattern.findall(predict_str)
    if len(matches) > 0:
        predict_str = matches[-1]
    else:
        return 0.0
    
    # print(f"predict_str: {predict_str}, original_answer: {original_answer}")

    # exact match
    if predict_str.strip() == original_answer:
        return 1.0
    
    # match format like "(A) xxx"
    # pattern = re.compile(r'\(([A-Z])\).*', re.DOTALL)
    # all_matches = pattern.findall(predict_str)
    # if len(all_matches) == 1 and all_matches[0].strip() == original_answer:
    #     return 1.0
    pattern = r'^\(([A-Z])\).*$'
    match = re.match(pattern, predict_str, re.DOTALL)
    if match and match.group(1) == original_answer:
        return 1.0

    # match format like "A. xxx"
    pattern = r'^([A-Z])\..*$'
    match = re.match(pattern, predict_str, re.DOTALL)
    if match and match.group(1) == original_answer:
        return 1.0

    return 0.0

def acc_reward(predict_str_list: list, solution: str, extra_info: dict = None) -> float:
    gpt_extract_answer = extra_info.get("gpt_extract_answer", False)
    try:
        reward = inner_acc_reward(predict_str_list, solution, use_gpt=True, gpt_extract_answer=gpt_extract_answer, extra_info=extra_info)
    except Exception as e:
        print(str(e))
        return {"is_filter": True, "info": str(e)}
    return reward

def compute_score(predict_str_list: list, ground_truth: str, extra_info: dict = None) -> dict:
    acc_reward_weight = extra_info.get('acc_reward_weight', 1.0) if extra_info else 1.0
    format_reward_weight = extra_info.get('format_reward_weight', 1.0) if extra_info else 1.0
    tool_call_penalty = 0.1
    if extra_info is not None and 'tool_call_penalty' in extra_info:
        tool_call_penalty = extra_info.get('tool_call_penalty', 0.1)
    acc = acc_reward(predict_str_list, ground_truth, extra_info)
    if isinstance(acc, dict):
        return acc
    format_score, tool_call_count = grounding_format_reward(predict_str_list, extra_info)

    acc_score = acc_reward_weight * acc
    format_score = format_reward_weight * format_score
    # if tool_call_count > 0:
    #     format_score += 1.0 * extra_info.get('use_tool_reward_weight')

    # print("tool_call_penalty: ", tool_call_penalty)

    tool_penalty_factor = (1 - tool_call_penalty) if tool_call_count > 0 else 1.0
    tool_reward = extra_info.get('use_tool_reward_weight', 0.0) if tool_call_count > 0 else 0.0
    score = tool_penalty_factor * acc_score + format_score + tool_reward

    # print(f"tool_penalty_factor: {tool_penalty_factor}, tool_reward: {tool_reward}, acc_score: {acc_score}, format_score: {format_score}, score: {score}")
    reward_dict = {
        "score": score,
        "acc_score": acc_score,
        "format_score": format_score,
        "tool_reward": tool_reward,
    }

    return reward_dict

if __name__ == '__main__':
    question = "Elena Ferrante" #"<image>\nHint: Please answer the question and provide the final answer at the end.\nQuestion: How many states are represented by the lightest color on the map?" #"<image>What is the output score when the first input is 4 and the second input is 5 according to the Hamlet Evaluation System shown in Figure 2?" #"<image>Who wrote this book?\nAnswer the question with a short phrase."
    predict_str = ["""<think>To determine the name of the store with a blue sign, I\'ll need to look closely at the sign. The image shows a building with several signs, and one of them is blue and located to the right of the gray baffle. I\'ll zoom in on that area to read the sign clearly.</think> <grounding>{"bbox_2d": [2761, 715, 3160, 896]}</grounding>""", """<think>To determine the name of the store with a blue sign</think> <answer>The name of the store with a blue sign is "J&optica." </answer>"""]
    ground_truth = "Jptica" #"Martha White" #"china" #"$ 2 $" #"A" #"1:3" #"0.5 cm" #"0.5"
    extra_info = {
        "acc_reward_weight": 1.0,
        "format_reward_weight": 0.5,
        "use_tool_reward_weight": 0.5,
        "gpt_extract_answer": True,
        "extract_answer_tags": "strict",
    }
    # s1 = compute_score(question, predict_str, ground_truth, extra_info)
    # print(s1)

    s2 = grounding_format_reward(predict_str, extra_info)
    print(s2)