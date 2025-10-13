import re
import os
import time
import requests


SYSTEM_PROMPT = "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs.\nYour task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:\n------\n##INSTRUCTIONS:\n- Focus on the meaningful match between the predicted answer and the correct answer.\n- Consider synonyms or paraphrases as valid matches.\n- Evaluate the correctness of the prediction compared to the answer."

QUERY_PROMPT = """I will give you a question related to an image and the following text as inputs:\n\n1. **Question Related to the Image**: {question}\n2. **Ground Truth Answer**: {ground_truth}\n3. **Model Predicted Answer**: {prediction}\n\nYour task is to evaluate the model's predicted answer against the ground truth answer, based on the context provided by the question related to the image. Consider the following criteria for evaluation:\n- **Relevance**: Does the predicted answer directly address the question posed, considering the information provided by the given question?\n- **Accuracy**: Compare the predicted answer to the ground truth answer. You need to evaluate from the following two perspectives:\n(1) If the ground truth answer is open-ended, consider whether the prediction accurately reflects the information given in the ground truth without introducing factual inaccuracies. If it does, the prediction should be considered correct.\n(2) If the ground truth answer is a definitive answer, strictly compare the model's prediction to the actual answer. Pay attention to unit conversions such as length and angle, etc. As long as the results are consistent, the model's prediction should be deemed correct.\n**Output Format**:\nYour response should include an integer score indicating the correctness of the prediction: 1 for correct and 0 for incorrect. Note that 1 means the model's prediction strictly aligns with the ground truth, while 0 means it does not.\nThe format should be \"Score: 0 or 1\""""


API_KEY = os.getenv("API_KEY")


def query(prompt: str, system_prompt: str = None,
            max_retries: int = 5, initial_delay: float = 3.0,
            model: str = "gemini-2.5-pro") -> str:

    if system_prompt is not None:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
        ]
    else:
        messages = []

    messages.append({"role": "user", "content": []})

    messages[-1]["content"].append({"type": "text", "text": prompt})


    attempt = 0
    delay = max(0.0, float(initial_delay))

    last_error: Exception | None = None
    while attempt < max_retries:
        try:
            temperature = min(0.2 * attempt, 1.0)

            url = "https://matrixllm.alipay.com/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}",  
            }
            data = {
                "stream": False,
                "model": model,             
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 16384,
                "timeout": 120,
            }

            resp = requests.post(url, headers=headers, json=data, timeout=120)
            if resp.status_code == 200:
                result = resp.json()
                if not result.get("choices"):
                    raise ValueError(f"Unexpected response: {result}")
                answer_text = result["choices"][0]["message"]["content"]
            elif resp.status_code == 429:
                # Backoff retry on rate limit with capped delay (30s)
                print("Received 429 too many requests. Backing off...")
                time.sleep(min(delay, 30))
                delay = min(delay * 2, 30)
                attempt += 1
                continue
            else:
                raise requests.HTTPError(
                    f"HTTP {resp.status_code}: {resp.text}"
                )

            if 'score:' not in answer_text.lower():
                raise ValueError(f"No 'score:' in response: {answer_text}")

            extract_score = (
                answer_text.lower()
                .split("score:")[-1]
                .strip()
                .split("\n")[0]
                .strip()
                .split(" ")[0]
            )

            if "1" not in extract_score and "0" not in extract_score:
                raise ValueError(f"No '0' nor '1' in response: {extract_score}")

            return extract_score

        except requests.Timeout as e:
            # On timeout, raise immediately as requested
            print("=" * 100)
            print(f"Timeout error: {e}")
            print("messages:", messages)
            print("=" * 100)
            raise
        except requests.ConnectionError as e:
            print("=" * 100)
            print(f"Network error: {e}")
            print("messages:", messages)
            print("=" * 100)
            time.sleep(min(delay, 30))
            delay = min(delay * 2, 30)
            attempt += 1
            last_error = e
            continue
        except Exception as e:
            print("=" * 100)
            print(f"Error: {e}")
            print("messages:", messages)
            print("=" * 100)
            time.sleep(min(delay, 30))
            delay = min(delay * 2, 30)
            attempt += 1
            last_error = e
            continue

    # Exceeded retries: raise with context
    raise RuntimeError(f"Failed after {max_retries} attempts") from last_error



def is_valid_direct_answer(response, direct_answer_format) -> bool:
    """
    对 <think>...</think><answer>...</answer> 的形式进行校验：
      1) 是否整体匹配大体结构
      2) 是否只出现一次 <think>...</think> 和 <answer>...</answer>
      3) 不应包含 <tool_call> </tool_call>
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
    if '<tool_call>' in response or '</tool_call>' in response:
        return False
    return True

def is_valid_direct_answer_grounding(response, direct_answer_format) -> bool:
    """
    对 <think>...</think><answer>...</answer> 的形式进行校验：
      1) 是否整体匹配大体结构
      2) 是否只出现一次 <think>...</think> 和 <answer>...</answer>
      3) 不应包含 <tool_call> </tool_call>
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
    对 <think>...</think>...<tool_call>...</tool_call> 的形式进行校验：
      1) 整体正则匹配
      2) <think>...</think> 各出现一次
      3) <tool_call>...</tool_call> 只出现一次
      4) 不应出现 <answer> </answer>
    """
    pattern = step_tool_call_format
    # 1). Structure Matching
    if not re.match(pattern, response, re.DOTALL):
        return False
    # 2). <think> Count
    if response.count('<think>') != 1 or response.count('</think>') != 1:
        return False
    # 3). <tool_call> </tool_call> Count
    if response.count('<tool_call>') != 1 and response.count('</tool_call>') != 1:
        return False
    # 4). <answer> or </answer> is not allowed!
    if '<answer>' in response or '</answer>' in response:
        return False
    return True

def is_valid_tool_call_grounding(response, step_tool_call_format) -> bool:
    """
    对 <think>...</think>...<tool_call>...</tool_call> 的形式进行校验：
      1) 整体正则匹配
      2) <think>...</think> 各出现一次
      3) <tool_call>...</tool_call> 只出现一次
      4) 不应出现 <answer> </answer>
    """
    pattern = step_tool_call_format
    # 1). Structure Matching
    if not re.match(pattern, response, re.DOTALL):
        return False
    # 2). <think> Count
    if response.count('<think>') != 1 or response.count('</think>') != 1:
        return False
    # 3). <tool_call> </tool_call> Count
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
    step_tool_call_format = r'^<think>.*</think>.*<tool_call>.*</tool_call>$'
    tool_call_pattern = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
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

    return format_score, tool_call_count


def inner_acc_reward(prompt: str, predict_str_list: list, original_answer: str, use_gpt=False, gpt_extract_answer=False, extra_info=None):
    original_predict_str = ' '.join(predict_str_list)
    if gpt_extract_answer:
        if extra_info['extract_answer_tags'] == 'split':
            original_predict_str = original_predict_str.split("<answer>")[-1].split("</answer>")[0].strip()
        elif extra_info['extract_answer_tags'] == 'strict':
            extract_answer_pattern = r'<answer>(.*?)</answer>'
            match = re.search(extract_answer_pattern, original_predict_str, re.DOTALL)
            if match:
                original_predict_str = match.group(1)
            else:
                reward = 0.0
                return reward
        elif extra_info['extract_answer_tags'] == 'strict_v2':
            print("e1")
            extract_answer_pattern = r'<answer>(.*?)</answer>$'
            match = re.search(extract_answer_pattern, original_predict_str, re.DOTALL)
            if match:
                original_predict_str = match.group(1)
            else:
                reward = 0.0
                return reward
        else:
            raise ValueError("Such value is not implemented for extra_info['extract_answer_tags']: {}".format(extra_info['extract_answer_tags']))
    question = prompt
    prompt = QUERY_PROMPT.format(question=question, ground_truth=original_answer, prediction=original_predict_str)
    response = query(prompt=prompt, system_prompt=SYSTEM_PROMPT)
    
    
    if len(response) == 0:
        reward = {"is_filter": True, "info": "error with gpt4o"}
    else:
        reward = 1.0 if '1' in response else 0.0
    return reward

def acc_reward(prompt: str, predict_str_list: list, solution: str, extra_info: dict = None) -> float:
    gpt_extract_answer = extra_info.get("gpt_extract_answer", False)
    reward = inner_acc_reward(prompt, predict_str_list, solution, use_gpt=True, gpt_extract_answer=gpt_extract_answer, extra_info=extra_info)
    return reward

def compute_score(prompt: str, predict_str_list: list, ground_truth: list, extra_info: dict = None) -> float:
    acc_reward_weight = extra_info.get('acc_reward_weight', 1.0) if extra_info else 1.0
    format_reward_weight = extra_info.get('format_reward_weight', 1.0) if extra_info else 1.0
    tool_call_penalty = 0.1
    if extra_info is not None and 'tool_call_penalty' in extra_info:
        tool_call_penalty = extra_info.get('tool_call_penalty', 0.1)
    acc = acc_reward(prompt, predict_str_list, ground_truth, extra_info)
    if isinstance(acc, dict):
        return acc
    format_score, tool_call_count = grounding_format_reward(predict_str_list, extra_info)

    acc_score = acc_reward_weight * acc
    format_score = format_reward_weight * format_score
    
    tool_penalty_factor = (1 - tool_call_penalty) if tool_call_count > 0 else 1.0
    tool_reward = extra_info.get('use_tool_reward_weight', 0.0) if tool_call_count > 0 else 0.0
    score = tool_penalty_factor * acc_score + format_score + tool_reward

    return score, acc_score, format_score

if __name__ == '__main__':
    question = "Elena Ferrante"
    predict_str = ["""<think>To determine the name of the store with a blue sign, I\'ll need to look closely at the sign. The image shows a building with several signs, and one of them is blue and located to the right of the gray baffle. I\'ll zoom in on that area to read the sign clearly.</think> <grounding>{"bbox_2d": [2761, 715, 3160, 896]}</grounding>""", """<think>To determine the name of the store with a blue sign</think> <answer>The name of the store with a blue sign is "J&optica." </answer>"""]
    ground_truth = "Jptica"
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
