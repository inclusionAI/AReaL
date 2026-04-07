#!/usr/bin/env python
"""Test cases for the firejail sandbox environment.

Test coverage:
1. Execution Test - Basic command execution in sandbox environment
2. Return Result Test - STDOUT, STDERR
2. Timeout Test - Handling of long-running process termination
3. Modules Test - Verification of essential math package availability, e.g. numpy, pandas, sympy, scipy, etc.
4. Multiprocess Press Test - Stability under concurrent process execution
"""
import json
import requests
import fire
import logging
import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_firejail_python(
    url: str = None,
    trajectory_id: str = "test-firejail-python-001",
):
    """Test Firejail Python code execution with various test cases"""
    
    print("--- Test 1: Taco test cases ---") # should pass
    action = "```python\nimport math\n\ndef race(v1, v2, g):\n\tif v2 < v1:\n\t\treturn None\n\tseconds = 0.1\n\twhile v1 / 3600 * seconds + g >= v2 / 3600 * seconds:\n\t\tseconds += 0.05\n\thours = seconds / 3600\n\thoursRest = seconds % 3600\n\tminutes = hoursRest / 60\n\tseconds = hoursRest % 60\n\treturn [math.floor(hours), math.floor(minutes), math.floor(seconds)]\n\n```"
    print(_send_test_request(url, trajectory_id, action, "Hello World", extra_field={"public_tests": ['{"fn_name": "race", "inputs": [[720, 850, 70], [80, 91, 37], [80, 100, 40], [720, 850, 37], [720, 850, 370], [120, 850, 37], [820, 850, 550], [820, 81, 550]], "outputs": [[[0, 32, 18]], [[3, 21, 49]], [[2, 0, 0]], [[0, 17, 4]], [[2, 50, 46]], [[0, 3, 2]], [[18, 20, 0]], [null]]}']}))

    print("--- Test 2: Taco test cases without fn_name---") # should pass
    action = "```python\ndef sub(maxs, mins):\n\tfor i in range(len(maxs)):\n\t\tif maxs[i] != mins[i]:\n\t\t\tif i == len(maxs) - 1:\n\t\t\t\treturn int(maxs[i]) - int(mins[i])\n\t\t\tif i == len(maxs) - 2:\n\t\t\t\treturn int(maxs[i:i + 2]) - int(mins[i:i + 2])\n\t\t\treturn 10\n\treturn 0\n\ndef checkEqual(S):\n\tans = 8\n\tfor k in range(1, len(S)):\n\t\tif len(S) % k != 0:\n\t\t\tcontinue\n\t\tmins = maxs = S[0:k]\n\t\tfor s in range(0, len(S), k):\n\t\t\tmaxs = max(maxs, S[s:s + k])\n\t\t\tmins = min(mins, S[s:s + k])\n\t\tans = min(ans, sub(maxs, mins))\n\treturn ans\n\ndef check12(S):\n\tmaxv = 0\n\tminv = 10\n\tp = 0\n\twhile p < len(S):\n\t\tv = int(S[p])\n\t\tif S[p] == '1' and p + 1 < len(S):\n\t\t\tv = 10 + int(S[p + 1])\n\t\t\tp += 1\n\t\tmaxv = max(maxv, v)\n\t\tminv = min(minv, v)\n\t\tp += 1\n\treturn maxv - minv\nS = input()\nprint(min(checkEqual(S), check12(S)))\n```"
    print(_send_test_request(url, trajectory_id, action, "Hello World", extra_field={"public_tests": ['{"inputs": ["9714431", "16612328", "23422731", "754526", "955577", "75547", "2112", "799", "88", "32523857", "4787", "1859551", "135661", "3675", "156692", "167918384", "83994", "4837847", "14513597", "15282598", "12659326", "1468417", "6280", "115464", "52376853", "2315", "3641224", "97187", "836", "195884", "36250", "2427817", "17598762", "5744554", "9295", "129848", "3863342", "3743", "133862", "1237", "1625", "1179729", "12651", "3776912", "4829", "73", "2228", "2546", "3136", "138", "3380", "4828", "3652", "5667", "7275", "774", "9329", "279", "15119", "200", "2461", "19", "2258", "31", "1250", "1216", "1595", "271", "236", "187", "166", "123", "231272", "12342923", "16587352", "32887158", "42478456", "353843", "1884868", "148239", "54241537", "213811", "3614", "1003", "177127860", "54250", "1720310", "6415742", "12117", "1293", "5541389", "44936", "550", "43448", "664", "39426", "5003285", "73925", "4379155", "2270", "123125129", "119138", "11121314"], "outputs": ["8\\n", "7\\n", "6\\n", "5\\n", "4\\n", "3\\n", "1\\n", "2\\n", "0\\n", "6\\n", "4\\n", "8\\n", "5\\n", "4\\n", "8\\n", "8\\n", "6\\n", "5\\n", "8\\n", "8\\n", "8\\n", "7\\n", "8\\n", "5\\n", "6\\n", "4\\n", "5\\n", "8\\n", "5\\n", "8\\n", "6\\n", "7\\n", "8\\n", "3\\n", "3\\n", "8\\n", "6\\n", "4\\n", "7\\n", "6\\n", "5\\n", "8\\n", "5\\n", "8\\n", "7\\n", "4\\n", "6\\n", "4\\n", "5\\n", "5\\n", "8\\n", "6\\n", "4\\n", "2\\n", "3\\n", "3\\n", "7\\n", "7\\n", "6\\n", "2\\n", "5\\n", "8\\n", "6\\n", "2\\n", "5\\n", "4\\n", "8\\n", "6\\n", "4\\n", "7\\n", "5\\n", "2\\n", "6\\n", "8\\n", "7\\n", "7\\n", "6\\n", "5\\n", "7\\n", "8\\n", "6\\n", "7\\n", "5\\n", "3\\n", "8\\n", "5\\n", "7\\n", "6\\n", "5\\n", "8\\n", "8\\n", "6\\n", "5\\n", "5\\n", "2\\n", "7\\n", "8\\n", "7\\n", "8\\n", "7\\n", "6", "5", "3"]}']}))

    print("--- Test 3: Taco test cases without fn_name one wrong test cases---") # should not pass, I changed the first outputs from 8 to 7 in the expected return
    action = "```python\ndef sub(maxs, mins):\n\tfor i in range(len(maxs)):\n\t\tif maxs[i] != mins[i]:\n\t\t\tif i == len(maxs) - 1:\n\t\t\t\treturn int(maxs[i]) - int(mins[i])\n\t\t\tif i == len(maxs) - 2:\n\t\t\t\treturn int(maxs[i:i + 2]) - int(mins[i:i + 2])\n\t\t\treturn 10\n\treturn 0\n\ndef checkEqual(S):\n\tans = 8\n\tfor k in range(1, len(S)):\n\t\tif len(S) % k != 0:\n\t\t\tcontinue\n\t\tmins = maxs = S[0:k]\n\t\tfor s in range(0, len(S), k):\n\t\t\tmaxs = max(maxs, S[s:s + k])\n\t\t\tmins = min(mins, S[s:s + k])\n\t\tans = min(ans, sub(maxs, mins))\n\treturn ans\n\ndef check12(S):\n\tmaxv = 0\n\tminv = 10\n\tp = 0\n\twhile p < len(S):\n\t\tv = int(S[p])\n\t\tif S[p] == '1' and p + 1 < len(S):\n\t\t\tv = 10 + int(S[p + 1])\n\t\t\tp += 1\n\t\tmaxv = max(maxv, v)\n\t\tminv = min(minv, v)\n\t\tp += 1\n\treturn maxv - minv\nS = input()\nprint(min(checkEqual(S), check12(S)))\n```"
    print(_send_test_request(url, trajectory_id, action, "Hello World", extra_field={"public_tests": ['{"inputs": ["9714431", "16612328", "23422731", "754526", "955577", "75547", "2112", "799", "88", "32523857", "4787", "1859551", "135661", "3675", "156692", "167918384", "83994", "4837847", "14513597", "15282598", "12659326", "1468417", "6280", "115464", "52376853", "2315", "3641224", "97187", "836", "195884", "36250", "2427817", "17598762", "5744554", "9295", "129848", "3863342", "3743", "133862", "1237", "1625", "1179729", "12651", "3776912", "4829", "73", "2228", "2546", "3136", "138", "3380", "4828", "3652", "5667", "7275", "774", "9329", "279", "15119", "200", "2461", "19", "2258", "31", "1250", "1216", "1595", "271", "236", "187", "166", "123", "231272", "12342923", "16587352", "32887158", "42478456", "353843", "1884868", "148239", "54241537", "213811", "3614", "1003", "177127860", "54250", "1720310", "6415742", "12117", "1293", "5541389", "44936", "550", "43448", "664", "39426", "5003285", "73925", "4379155", "2270", "123125129", "119138", "11121314"], "outputs": ["7\\n", "7\\n", "6\\n", "5\\n", "4\\n", "3\\n", "1\\n", "2\\n", "0\\n", "6\\n", "4\\n", "8\\n", "5\\n", "4\\n", "8\\n", "8\\n", "6\\n", "5\\n", "8\\n", "8\\n", "8\\n", "7\\n", "8\\n", "5\\n", "6\\n", "4\\n", "5\\n", "8\\n", "5\\n", "8\\n", "6\\n", "7\\n", "8\\n", "3\\n", "3\\n", "8\\n", "6\\n", "4\\n", "7\\n", "6\\n", "5\\n", "8\\n", "5\\n", "8\\n", "7\\n", "4\\n", "6\\n", "4\\n", "5\\n", "5\\n", "8\\n", "6\\n", "4\\n", "2\\n", "3\\n", "3\\n", "7\\n", "7\\n", "6\\n", "2\\n", "5\\n", "8\\n", "6\\n", "2\\n", "5\\n", "4\\n", "8\\n", "6\\n", "4\\n", "7\\n", "5\\n", "2\\n", "6\\n", "8\\n", "7\\n", "7\\n", "6\\n", "5\\n", "7\\n", "8\\n", "6\\n", "7\\n", "5\\n", "3\\n", "8\\n", "5\\n", "7\\n", "6\\n", "5\\n", "8\\n", "8\\n", "6\\n", "5\\n", "5\\n", "2\\n", "7\\n", "8\\n", "7\\n", "8\\n", "7\\n", "6", "5", "3"]}']}))

    print("--- Test 4: Taco test cases without fn_name one wrong test cases---") # should pass
    action = "```python\nt = int(input())\nfor z in range(t):\n\tn = int(input())\n\tarr = list(map(int, input().split()))\n\tif len(set(arr)) == 1:\n\t\tprint('NO ')\n\telse:\n\t\tprint('YES ')\n\t\trep = []\n\t\tfor i in range(1, n):\n\t\t\tif arr[0] == arr[i]:\n\t\t\t\trep.append(i)\n\t\t\telse:\n\t\t\t\tprint('1', i + 1)\n\t\t\t\tk = i\n\t\tfor num in rep:\n\t\t\tprint(k + 1, num + 1)\n\n```"
    print(_send_test_request(url, trajectory_id, action, "Hello World", extra_field={"public_tests": ['{"inputs": ["4\\n5\\n1 2 2 1 3\\n3\\n1 1 1\\n4\\n1 1000 101 1000\\n4\\n1 2 3 4\\n", "1\\n5\\n6756657 32231 86 234 23442\\n", "1\\n2\\n7 7\\n"], "outputs": ["YES\\n1 2\\n1 3\\n1 5\\n5 4\\nNO\\nYES\\n1 2\\n1 3\\n1 4\\nYES\\n1 2\\n1 3\\n1 4\\n", "YES\\n1 2\\n1 3\\n1 4\\n1 5\\n", "NO\\n", "NO\\n", "YES\\n1 2\\n1 3\\n1 4\\n1 5\\n"]}']}))

    print("--- Test 5: Taco test cases without fn_name one wrong test cases---") # should pass
    action = "```python\nn = int(input())\na = list(map(int, input().split()))\ncnt = {}\nfor i in range(n):\n\tmn = a[i]\n\tfor j in range(i, n):\n\t\tmn = min(mn, a[j])\n\t\tif mn in cnt:\n\t\t\tcnt[mn] += 1\n\t\telse:\n\t\t\tcnt[mn] = 1\nq = int(input())\nfor i in range(q):\n\tk = int(input())\n\tif k in cnt:\n\t\tprint(cnt[k])\n\telse:\n\t\tprint(0)\n```"
    print(_send_test_request(url, trajectory_id, action, "Hello World", extra_field={"public_tests": ['{"inputs": [["5", "4 1 2 3 4", "4", "3", "4", "6", "1", "", ""], "5\\n4 0 2 3 4\\n4\\n3\\n4\\n6\\n1", "5\\n4 0 2 3 4\\n4\\n5\\n4\\n6\\n1"], "outputs": [["2", "2", "0", "8"], "2\\n2\\n0\\n0\\n", "0\\n2\\n0\\n0\\n"]}']}))

    print("--- Test 6: Taco test cases without fn_name one wrong test cases---") # should pass
    action = "```python\nfrom collections import deque\n\nclass Node:\n    def __init__(self, key, left=None, right=None, parent=None):\n        self.key = key\n        self.left = left\n        self.right = right\n        self.parent = parent\n\nclass BinaryTree:\n    def __init__(self):\n        self.root = None\n\n    def insert(self, key):\n        z = Node(key)\n        y = None\n        x = self.root\n        while x != None:\n            y = x\n            if z.key < x.key:\n                x = x.left\n            else:\n                x = x.right\n        z.parent = y\n        if y == None:\n            self.root = z\n        elif z.key < y.key:\n            y.left = z\n        else:\n            y.right = z\n\n    def inorder_traversal(self, node):\n        if node is None:\n            return []\n        inorder = []\n        inorder.extend(self.inorder_traversal(node.left))\n        inorder.append(node.key)\n        inorder.extend(self.inorder_traversal(node.right))\n        return inorder\n\n    def preorder_traversal(self, node):\n        if node is None:\n            return []\n        preorder = [node.key]\n        preorder.extend(self.preorder_traversal(node.left))\n        preorder.extend(self.preorder_traversal(node.right))\n        return preorder\n\n    def print_keys(self):\n        inorder_keys = self.inorder_traversal(self.root)\n        preorder_keys = self.preorder_traversal(self.root)\n        print(' '.join(map(str, inorder_keys)))\n        print(' '.join(map(str, preorder_keys)))\n\nbinary_tree = BinaryTree()\noperation_count, output = 0, []\nfor _ in range(int(input())):\n    op = input().split()\n    if op[0] == \"insert\":\n        binary_tree.insert(int(op[1]))\n    else:\n        binary_tree.print_keys()\n    operation_count += 1\n```"
    print(_send_test_request(url, trajectory_id, action, "Hello World", extra_field={"public_tests": ["{\"type\": \"stdin_stdout\", \"inputs\": [\"8\\ninsert 30\\ninsert 88\\ninsert 18\\ninsert 1\\ninsert 20\\ninsert 17\\ninsert 25\\nprint\", \"8\\ninsert 30\\ninsert 113\\ninsert 18\\ninsert 1\\ninsert 20\\ninsert 17\\ninsert 25\\nprint\", \"8\\ninsert 30\\ninsert 88\\ninsert 18\\ninsert 1\\ninsert 20\\ninsert 21\\ninsert 25\\nprint\"], \"outputs\": [\" 1 17 18 20 25 30 88\\n 30 18 1 17 20 25 88\\n\", \" 1 17 18 20 25 30 113\\n 30 18 1 17 20 25 113\\n\", \" 1 18 20 21 25 30 88\\n 30 18 1 20 21 25 88\\n\"]}"]}))
    
def _send_test_request(url, trajectory_id, action, test_name, extra_field=None):
    """Helper function to send test requests and process responses"""
    logger.info(f"Testing {test_name} code execution...")
    
    if extra_field is None:
        extra_field = {}
    
    # Use server API
    payload = {
        "trajectory_ids": [trajectory_id],
        "actions": [action],
        **extra_field,
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise exception for error status codes
        
        result = response.json()
        logger.info(f"Response received for {test_name} test")
        
        # Print observation
        if "observations" in result and len(result["observations"]) > 0:
            observation = result["observations"][0]
            logger.info(f"\n--- {test_name} Result ---\n{observation}\n")
        else:
            logger.error(f"No observation found in response for {test_name}")
        
        return result
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error: {str(e)}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {"error": str(e)}


def main():
    """Main entry point for the test script
    Run with:
        python -m verl_tool.servers.tests.test_python_oj_tool --url=http://localhost:5000/get_observation
    """
    fire.Fire(test_firejail_python)

if __name__ == "__main__":
    main()