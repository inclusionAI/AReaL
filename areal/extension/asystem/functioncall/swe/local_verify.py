import asyncio
import difflib
import json
import os
import re
import time
from typing import Any, Dict, List, Tuple, TypedDict

from areal.extension.asystem.functioncall.base.utils import construct_swe_uid, logger

SEARCH_REPLACE_REGEX = r"```.*?\n### (.*)\n<<<<<<< SEARCH\n([\s\S]*?)=======\n([\s\S]*?)>>>>>>> REPLACE\n```"


def parse_search_replace(text: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Parse the search/replace blocks from the text.

    Returns:
        A Dictionary where the key is the file path and the value is a List of search/replace pairs.
    """
    path_search_replaces: List[Tuple[str, str, str]] = re.findall(
        SEARCH_REPLACE_REGEX, text
    )
    path_search_replace_Dict = dict[str, List[Tuple[str, str]]]()
    for path, search, replace in path_search_replaces:
        if search.endswith("\n"):
            search = search[:-1]
        if replace.endswith("\n"):
            replace = replace[:-1]
        path_search_replace_Dict.setdefault(path, []).append((search, replace))
    return path_search_replace_Dict


def normalize_sr(patch: str) -> Dict:
    """
    标准化patch片段：
    1. 先去除<think></think>标记里的内容
    2. 然后提取最后的search replace里的内容

    Args:
        patch (str): 原始推理输出

    Returns:
        str: 标准化后的search replace
    """
    if not patch:
        return {}

    # 去除前后空白
    patch = patch.strip()

    # 1. 去除<think></think>标记里的内容
    # 使用非贪婪匹配，匹配所有<think></think>对
    think_pattern = r"<think>.*?</think>"
    patch = re.sub(think_pattern, "", patch, flags=re.DOTALL)

    # 2. 提取最后的search replace里的内容
    pred_search_replaces = parse_search_replace(patch)

    return pred_search_replaces


def apply_code_change(
    code_context: Dict[str, str],
    search_replace_dict: Dict[str, List[Tuple[str, str]]],
    silent: bool = True,
) -> Dict[str, str]:
    """
    Apply the search/replace edits to the code context.

    Args:
        code_context: A dictionary containing the file path and the content of the code.
        search_replace_dict: A dictionary mapping the file path to the search/replace edits.
        silent: Whether to suppress the error messages.

    Returns:
        A dictionary containing the file path and the new content of the code.
    """
    new_content_dict = dict[str, str]()
    for path, search_replaces in search_replace_dict.items():
        new_content = "\n" + code_context.get(path, "")
        for search, replace in search_replaces:
            if search == "":
                new_content = "\n" + replace
            else:
                # Ensure search block can be matched
                # "\n" + search to ensure the indentations are correct
                if not silent and len(search) == len(replace) and search == replace:
                    logger.info("SWE Reward: Search and replace blocks are identical")
                search = "\n" + search
                replace = "\n" + replace
                if not silent and search not in new_content:
                    logger.info(
                        f"SWE Reward: Search block not found in the code: {search}"
                    )
                new_content = new_content.replace(search, replace)
        # Remove the leading "\n"
        final_content = new_content[1:]
        if len(final_content) > 1000000:  # 1M限制
            final_content = code_context.get(path, "")
            print(
                f"SWE Reward: [apply_code_change] [WARNING] 最终文件过长，使用原始内容: {len(final_content)} 字符"
            )
        new_content_dict[path] = final_content
    return new_content_dict


class ChangeSimilarity(TypedDict):
    path: str
    pred_change: str
    oracle_change: str
    similarity: float


def compute_change_similarities(
    pred_patch: Dict[str, str],
    oracle_patch: Dict[str, str],
) -> List[ChangeSimilarity]:
    all_file_paths = set(oracle_patch.keys()).union(set(pred_patch.keys()))
    similarities = list[ChangeSimilarity]()
    for path in all_file_paths:
        pred_change = pred_patch.get(path, "")
        oracle_change = oracle_patch.get(path, "")
        if oracle_change == "" or pred_change == "":
            # Both are empty changes, meaning search = replace. We should penalize this to avoid
            # the model preDicting empty changes to hack the reward.
            # NOTE: this should not happen due to (1) the search == replace check in `apply_code_change`
            # and (2) the `if patch` check in `get_normalized_patch`.
            change_similarity = 0.0
        else:
            change_similarity = difflib.SequenceMatcher(
                None,
                pred_change.strip(),
                oracle_change,
                autojunk=False,
            ).ratio()
        similarities.append(
            ChangeSimilarity(
                path=path,
                pred_change=pred_change,
                oracle_change=oracle_change,
                similarity=change_similarity,
            )
        )
    return similarities


def calculate_reward(
    code_context: Dict[str, str],
    pred_new_content: Dict[str, str],
    gold_patches: Dict[str, str],
) -> Tuple[float, Dict]:
    # Obtain a unified diff for each file, for both the preDicted and the oracle patch
    pred_patch = get_normalized_patch(code_context, pred_new_content)
    # Calculate the reward based on the similarity between the preDicted and the oracle patch
    similarities = compute_change_similarities(pred_patch, gold_patches)
    # assert len(similarities) > 0
    # This means oracle_patch and pred_patch are both empty, then they are identical and we reward 1.0
    if len(similarities) == 0:
        assert len(gold_patches) == 0 and len(pred_patch) == 0
        return 1.0, dict(similarities=[])
    reward = sum(map(lambda x: x["similarity"], similarities)) / len(similarities)
    return reward, dict(similarities=similarities)


def get_normalized_patch(
    code_context: Dict[str, str],
    new_content_Dict: Dict[str, str],
) -> Dict[str, str]:
    """
    According to the code context and new content, generate the normalized patch for each file.

    Args:
        code_context: A Dictionary containing the file path and the content of the code.
        new_content_Dict: A Dictionary mapping the file path to the new content of the file.

    Returns:
        A Dictionary containing the file path and the normalized patch.
    """
    patch_Dict = dict[str, str]()
    for path, new_content in new_content_Dict.items():
        old_content = code_context.get(path, "")
        old_name = path if old_content else "/dev/null"
        new_name = path
        patch = generate_unified_diff(old_content, old_name, new_content, new_name)
        # Only add the patch if it's not empty
        # NOTE: this should not happen due to the search == replace check in `apply_code_change`
        # but it can occur in general-purpose usages
        if patch:
            patch_Dict[path] = patch
    return patch_Dict


def generate_unified_diff_old(
    old_code: str,
    old_name: str,
    new_code: str,
    new_name: str,
    n_context: int = 3,
) -> str:
    """Generate a unified diff between two code.

    Args:
        old_code: The original code.
        new_code: The modified code.
        n_context: The number of context lines to show.

    Returns:
        A string representing the unified diff."""

    original_lines = old_code.splitlines()
    modified_lines = new_code.splitlines()

    diff = difflib.unified_diff(
        original_lines,
        modified_lines,
        fromfile="a/" + old_name if old_name != "/dev/null" else old_name,
        tofile="/dev/null" if new_code == "" else "b/" + new_name,
        lineterm="",
        n=n_context,
    )
    try:
        diff_code = "\n".join(diff)
        return diff_code
    except StopIteration:
        return ""


def generate_unified_diff(
    old_code: str,
    old_name: str,
    new_code: str,
    new_name: str,
    n_context: int = 3,
) -> str:
    """Generate a unified diff between two code."""

    if old_code and not old_code.endswith("\n"):
        old_code += "\n"
    if new_code and not new_code.endswith("\n"):
        new_code += "\n"

    original_lines = old_code.splitlines(keepends=True)
    modified_lines = new_code.splitlines(keepends=True)
    diff = difflib.unified_diff(
        original_lines,
        modified_lines,
        fromfile="a/" + old_name if old_name != "/dev/null" else old_name,
        tofile="/dev/null" if new_code == "" else "b/" + new_name,
        lineterm="",
        n=n_context,
    )

    diff_lines = list(diff)

    cleaned_lines = []
    for line in diff_lines:
        if line.startswith(("+", "-", " ", "@@", "---", "+++")):
            cleaned_lines.append(line.rstrip() + "\n")
        else:
            cleaned_lines.append(line)

    return "".join(cleaned_lines)


def calculate_search_replace_reward(
    code_context: Dict[str, str],
    output: str,
    gold_patches: Dict[str, str],
) -> float:
    """
    The search/replace version of the reward calculation. It expects the output to contain
    the thought and solution in the following format:
    <think>
    ...
    </think>
    ...

    Args:
        code_context: path -> original content of the file.
        oracle_new_content: path -> oracle new content of the file after change.
        output: The output from the model containing the thought and solution.

    Returns:
        A float value representing the reward, and a Dictionary containing some metadata.
    """
    reward = -1.0
    # Parse the search/replace edits from the solution
    pred_search_replaces = normalize_sr(output)
    if len(pred_search_replaces) == 0:
        logger.info("SWE Reward: No valid search blocks found")
    # Get the new content of each file after applying the search/replace edits
    pred_new_content = apply_code_change(code_context, pred_search_replaces)
    reward, metadata = calculate_reward(code_context, pred_new_content, gold_patches)
    return reward


def compute_score(uid: str, output: str, extra_info: Dict) -> float:
    """
    使用difflib.SequenceMatcher计算patch相似度

    Args:
        output (str): 模型预测
        ground_truth_patch (str): 标准答案patch片段
        extra_info (Dict): code_context, patches

    Returns:
        float: 相似度分数 (0.0-1.0)
    """

    # 如果preDicted_patch 为空，返回0.0
    if not output:
        return 0.0
    reward = 0.0

    try:
        reward = calculate_search_replace_reward(
            (
                json.loads(extra_info["code_context"])
                if type(extra_info["code_context"]) is str
                else extra_info["code_context"]
            ),
            output,
            (
                json.loads(extra_info["gold_patches"])
                if type(extra_info["gold_patches"]) is str
                else extra_info["gold_patches"]
            ),
        )
        return reward
    except Exception as e:
        logger.error(f"error when computing swe score: {e}, uid: {uid}")
        import traceback

        traceback.print_exc()
        return 0.0


async def swe_verify(
    id2info: Dict[str, Any],
    generateds: List[str],
    query_ids: List[str],
) -> List[int]:
    if not generateds:
        return []

    assert len(generateds) == len(
        query_ids
    ), f"Mismatch in input lengths: {len(generateds)} generations vs {len(query_ids)} query IDs."

    logger.info(
        f"Starting local swe verification for {len(generateds)} items, query_ids: {query_ids}"
    )
    start_time = time.time()
    tasks = []
    for i, (query_id, generated) in enumerate(zip(query_ids, generateds)):
        base_query_id = query_id.split("@idx:")[0]
        info = id2info[base_query_id]
        uid = construct_swe_uid(query_id)
        task = asyncio.to_thread(
            compute_score,
            uid,
            generated,  # solution_str
            info["extra_info"],  # puzzle_type
        )
        tasks.append(task)

    results = await asyncio.gather(*tasks, return_exceptions=True)

    final_results = []
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Exception during local swe verification: {result}")
            final_results.append(0.0)
        else:
            final_results.append(result)

    labels = [int(score > 0) for score in final_results]

    logger.info(
        f"Local swe_verify finished, request count: {len(query_ids)}, result count: {len(labels)}, time: {time.time() - start_time:.4f} seconds, swe reward: {labels}"
    )
    return labels


def test_sr_reward():
    """测试search&replace reward"""

    test_cases = [
        # 测试用例1: 包含think标记和patch标记
        {
            "input": """
<think>
我需要分析这个问题，首先检查代码结构...
</think>

```
### src/tmp.py
<<<<<<< SEARCH
3
5
=======
3
4
5
>>>>>>> REPLACE
```

""",
            "expected": {
                "src/tmp.py": """--- a/src/tmp.py
+++ b/src/tmp.py
@@ -2,4 +2,5 @@
 1
 2
 3
+4
 5"""
            },
            "code_context": {
                "src/tmp.py": """test
1
2
3
5
"""
            },
        },
        {
            "input": """
<think>
我需要分析这个问题，首先检查代码结构...
</think>

```
### src/tmp.py
<<<<<<< SEARCH
1
2
3
5
=======
>>>>>>> REPLACE
```

""",
            "expected": {
                "src/tmp.py": """--- a/src/tmp.py
+++ /dev/null
@@ -1,4 +0,0 @@
-1
-2
-3
-5"""
            },
            "code_context": {
                "src/tmp.py": """1
2
3
5"""
            },
        },
        {
            "input": """
<think>
我需要分析这个问题，首先检查代码结构...
</think>

```
### src/tmp.py
<<<<<<< SEARCH
=======
1
2
3
5
>>>>>>> REPLACE
```

""",
            "expected": {
                "src/tmp.py": """--- /dev/null
+++ b/src/tmp.py
@@ -0,0 +1,4 @@
+1
+2
+3
+5"""
            },
            "code_context": {},
        },
    ]

    id2info = {}
    generateds = []
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}:")
        print(f"输入: {repr(test_case['input'])}")
        print(f"code_context: {repr(test_case['code_context'])}")

        id2info[f"{i}"] = {
            "extra_info": {
                "code_context": json.dumps(test_case["code_context"]),
                "gold_patches": json.dumps(test_case["expected"]),
            }
        }
        generateds.append(test_case["input"])
        expected = test_case["expected"]
        print(f"期望: {repr(expected)}")

    rewards = asyncio.run(swe_verify(id2info, generateds, list(id2info.keys())))
    print("\n" + "=" * 50)
    print(f"rewards = {rewards}")


if __name__ == "__main__":
    test_sr_reward()
