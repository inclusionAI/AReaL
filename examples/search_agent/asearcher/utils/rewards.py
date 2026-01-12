# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import string
import random

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def bool_mapping(s):
    if s == "True":
        return "yes"
    elif s == "False":
        return "no"
    else:
        return s

def contains_chinese(text):
    """
    Check if the given text contains Chinese characters.
    Returns True if any Chinese character is found, False otherwise.
    """
    for char in text:
        # Check for common Chinese characters (CJK Unified Ideographs)
        if '\u4e00' <= char <= '\u9fff':
            return True
        # Check for rare characters (CJK Unified Ideographs Extension A)
        if '\u3400' <= char <= '\u4dbf':
            return True
        # Check for compatibility characters
        if '\uf900' <= char <= '\ufaff':
            return True
        # Check for extensions B-F (requires surrogate pairs in Python)
        # Note: This part handles supplementary characters (needed for Python < 3.3)
        if len(char) > 1:  # Surrogate pair
            code = ord(char[0]) << 16 | ord(char[1])
            if (0x20000 <= code <= 0x2a6df or    # Extension B
                0x2a700 <= code <= 0x2b73f or   # Extension C
                0x2b740 <= code <= 0x2b81f or   # Extension D
                0x2b820 <= code <= 0x2ceaf or   # Extension E
                0x2ceb0 <= code <= 0x2ebef):   # Extension F
                return True
    return False

def em_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(bool_mapping(prediction))
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(bool_mapping(golden_answer))
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


def subem_check(prediction, golden_answers):
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(bool_mapping(prediction))
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(bool_mapping(golden_answer))
        if golden_answer in normalized_prediction:
            score = 1
            break
    return score


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    # if "Assistant:" in solution_str:
    #     solution_str = solution_str.split("Assistant:", 1)[1]
    # elif "<|im_start|>assistant" in solution_str:
    #     solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    # else:
    #     return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    
    # If there are 0 or exactly 1 matches, return None
    if len(matches) <= 0: #1:
        return None
    
    # If there are 2 or more matches, return the last one
    return matches[-1].group(1).strip()


def compute_score_em(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    
    if isinstance(ground_truth, list):
        answer = extract_solution(solution_str=solution_str)
        return answer, max([compute_score_em(solution_str, g)[1] for g in ground_truth])

    answer = extract_solution(solution_str=solution_str)

    if answer is None:
        return None, 0
    else:
        if em_check(answer, ground_truth):
            return answer, score
        else:
            return answer, format_score


def compute_score_subem(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for substring exact match (EM).

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        method: the method to extract the solution, choices are 'strict' and 'flexible'
        format_score: the score for the format
        score: the score for the correct answer
    """
    answer = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Golden answers: {ground_truth['target']}")
        print(f"Extracted answer: {answer}")
        print(f"Solution string: {solution_str}")
    
    if answer is None:
        return 0
    else:
        if subem_check(answer, ground_truth['target']):
            return score
        else:
            return format_score

def normalize_text(text: str) -> str:
    """预处理文本，用于NQ数据集的评分
    
    处理步骤:
    1. 转换为小写
    2. 移除标点符号 (.,!?;:'"()[]{}...)
    3. 去除多余空格
    """
    # 将标点符号替换为空格
    for punct in string.punctuation:
        text = text.replace(punct, ' ')
    
    # 替换多个空格为单个空格
    text = re.sub(r'\s+', ' ', text)
    
    # 去除首尾空格
    text = text.strip().lower()
    return text

def f1_score(answer_content, gt):
    answer_content = normalize_text(bool_mapping(answer_content))
    gt = normalize_text(bool_mapping(gt))

    # 将答案和参考答案分词
    if contains_chinese(gt):
        def parse_chinese_str(s):
            # parse consecutive numbers
            numbers = []
            for i, c in enumerate(s):
                if c.isdigit():
                    if i > 0 and s[i-1].isdigit():
                        numbers[-1] = numbers[-1] + c
                    else:
                        numbers.append(c)
            for c in "0123456789，。 ,.-":
                s = s.replace(c, "")
            s = set(list(s) + numbers)
            return s
        pred_tokens = parse_chinese_str(answer_content)
        gt_tokens = parse_chinese_str(gt)
    else:
        pred_tokens = set(answer_content.split())
        gt_tokens = set(gt.split())
    
    if not gt_tokens:  # 避免除零错误
        return 0
    if not pred_tokens:
        return 0
    
    # 计算共同的词数
    common_tokens = pred_tokens & gt_tokens
    
    # 计算精确率和召回率
    precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
    recall = len(common_tokens) / len(gt_tokens) if gt_tokens else 0
    
    # 计算F1分数
    f1 = 0
    if precision + recall > 0:  # 避免除零错误
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1


def compute_score_f1(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    if isinstance(ground_truth, list):
        answer = extract_solution(solution_str=solution_str)
        return answer, max([compute_score_f1(solution_str, g)[1] for g in ground_truth])
        
    answer = extract_solution(solution_str=solution_str)

    if answer is None:
        return None, 0
    else:
        ret_score = f1_score(answer, ground_truth)
        return answer, ret_score

def cover_exact_match_score_1(solution_str, ground_truth):
    if isinstance(ground_truth, list):
        answer = extract_solution(solution_str=solution_str)
        return answer, max([cover_exact_match_score_1(solution_str, g)[1] for g in ground_truth])

    answer = extract_solution(solution_str=solution_str)

    if answer is None:
        return None, 0

    pre_list = normalize_answer(bool_mapping(answer)).split(" ")
    ground_list = normalize_answer(bool_mapping(ground_truth)).split(" ")
    # print("prediction: ",prediction)
    # print("ground_truth: ",ground_truth)
    # print("pre_list: ",pre_list)
    # print("ground_list: ",ground_list)
    # 不考虑顺序和连续
    return answer, float(all(ground in pre_list for ground in ground_list))

def correct_format_fn(idx, s):
    correct = all(
        [
            s.count("<search>") == s.count("</search>"),
            s.count("<access>") == s.count("</access>"),
            s.count("<answer>") == s.count("</answer>"),
            s.count("<search>") + s.count("<access>") + s.count("<answer>") <= 1,
            # s.count("<information>") == s.count("</information>") == s.count("<|begin_of_documents|>") == s.count("<|end_of_documents|>") == 0,
            s.count("Assistant") == s.count("assistant") == 0,
            s.count("</think>") <= 1,
           #  (s.strip().endswith("</search>") or s.strip().endswith("</answer>") or s.strip().endswith("</access>") or s.strip().endswith("</think>")),
        ]
    )
    return correct