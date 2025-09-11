import difflib
import re
from typing import Any, Dict, List, Tuple, TypedDict

from arealite.extension.asystem.functioncall.base.utils import logger

SEARCH_REPLACE_REGEX = r"```.*?\n### (.*)\n<<<<<<< SEARCH\n([\s\S]*?)=======\n([\s\S]*?)>>>>>>> REPLACE\n```"


def parse_search_replace(text: str) -> dict[str, list[tuple[str, str]]]:
    """
    Parse the search/replace blocks from the text.

    Returns:
        A dictionary where the key is the file path and the value is a list of search/replace pairs.
    """
    path_search_replaces: list[tuple[str, str, str]] = re.findall(
        SEARCH_REPLACE_REGEX, text
    )
    path_search_replace_dict = dict[str, list[tuple[str, str]]]()
    for path, search, replace in path_search_replaces:
        if search.endswith("\n"):
            search = search[:-1]
        if replace.endswith("\n"):
            replace = replace[:-1]
        path_search_replace_dict.setdefault(path, []).append((search, replace))
    return path_search_replace_dict


def normalize_sr(patch: str) -> dict:
    """
    标准化patch片段：
    1. 先去除<think></think>标记里的内容
    2. 然后提取最后的search replace里的内容

    Args:
        patch (str): 原始patch片段

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
    search_replace_dict: Dict[str, list[tuple[str, str]]],
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
                    logger.info(f"SWE Reward: Search block not found in the code: {search}")
                new_content = new_content.replace(search, replace)
        # Remove the leading "\n"
        final_content = new_content[1:]
        if len(final_content) > 1000000:  # 1M限制
            final_content = code_context.get(path, '')
            print(
                f"SWE Reward: [apply_code_change] [WARNING] 最终文件过长，使用原始内容: {len(final_content)} 字符"
            )
        new_content_dict[path] = final_content
    return new_content_dict


def get_normalized_patch(
    code_context: dict[str, str],
    new_content_dict: dict[str, str],
) -> dict[str, str]:
    """
    According to the code context and new content, generate the normalized patch for each file.

    Args:
        code_context: A dictionary containing the file path and the content of the code.
        new_content_dict: A dictionary mapping the file path to the new content of the file.

    Returns:
        A dictionary containing the file path and the normalized patch.
    """
    patch_dict = dict[str, str]()
    for path, new_content in new_content_dict.items():
        old_content = code_context.get(path, "")
        old_name = path if old_content else "/dev/null"
        new_name = path
        patch = generate_unified_diff(old_content, old_name, new_content, new_name)
        # Only add the patch if it's not empty
        # NOTE: this should not happen due to the search == replace check in `apply_code_change`
        # but it can occur in general-purpose usages
        if patch:
            patch_dict[path] = patch
    return patch_dict


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
        fromfile=f"a/{old_name}",
        tofile=f"b/{new_name}",
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


def convert_swe_output_to_patch(uid, code_context: dict[str, str], output: str) -> str:
    """
    Convert swe output from search&replace format to patch format. It expects the output to contain
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
        A string value representing the git patch.
    """
    try:
        # Parse the search/replace edits from the solution
        pred_search_replaces = normalize_sr(output)
        if len(pred_search_replaces) == 0:
            logger.info(
                f"SWE Reward: No valid search blocks found, uid: {uid}, output: {output}"
            )
        pred_new_content = apply_code_change(code_context, pred_search_replaces)
        # Obtain a unified diff for each file, for both the predicted and the oracle patch
        pred_patch = get_normalized_patch(code_context, pred_new_content)
        full_patch = "\n".join(pred_patch.values())
        return full_patch
    except Exception as e:
        logger.error(f"SWE Reward Error, uid: {uid}, error: {str(e)}")
        return ""


if __name__ == "__main__":
    # test_sr_reward()

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

    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\n测试用例 {i}:")
        logger.info(f"输入: {repr(test_case['input'])}")
        logger.info(f"code_context: {repr(test_case['code_context'])}")

        patch = convert_swe_output_to_patch(
            test_case["code_context"], test_case["input"]
        )

        logger.info(f"patch: {patch}")
