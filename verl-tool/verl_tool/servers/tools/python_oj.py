"""
add-apt-repository ppa:deki/firejail
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get -y install firejail firejail-profiles
"""
from .base import BaseTool, register_tool
import regex as re
import subprocess
import os
import json
import uuid
import numpy as np
from typing import Tuple, Dict, Any, Optional, Union, List
from .python_code import execute_python, TIMEOUT, wrap_code_blocks, PythonCodeTool

def stripped_string_compare(s1, s2):
    s1 = s1.strip()
    s2 = s2.strip()
    return s1 == s2

def only_int_check(val):
    return isinstance(val, int)

def string_int_check(val):
    return isinstance(val, str) and val.isdigit()

def combined_int_check(val):
    return only_int_check(val) or string_int_check(val)

def custom_compare(output:str, expected:str):
    expected = str(expected)
    output_lines = output.splitlines()
    if isinstance(output_lines, list):
        output_1 = "\n".join(output_lines)
        if stripped_string_compare(output_1, expected):
            return True

        # try remove extra space for each line in the output
        output_2 = [o.strip() for o in output_lines]
        output_2 = "\n".join(output_2)
        if stripped_string_compare(output_2, expected):
            return True

        # try remove extra space for each line in the expected
        expected_lines = expected.splitlines()
        expected_2 = [e.strip() for e in expected_lines]
        expected_2 = "\n".join(expected_2)
        if stripped_string_compare(output_2, expected_2):
            return True

        # check the ints and floats
        output_lines = [o for o in output_lines if o.strip() != ""]
        expected_lines = [e for e in expected.splitlines() if e.strip() != ""]
        output_lines = [o.strip() for o in output_lines]
        expected_lines = [e.strip() for e in expected_lines]
        all_ints = all(
            combined_int_check(e1) and combined_int_check(e2)
            for e1, e2 in zip(output_lines, expected_lines) if e1 and e2
        )
        try:
            if not all_ints:
                # check float
                output_float = [float(e) for e in output]
                gt_float = [float(e) for e in expected_lines]
                tmp_result = (
                    (len(output_float) == len(gt_float)) and np.allclose(output_float, gt_float)
                )
                if tmp_result:
                    return True
        except:
            pass
    return False

@register_tool
class PythonOJTool(PythonCodeTool):
    tool_type = "python_oj"
    timeout = TIMEOUT
    stop_tokens = ["```output", "<output>", "<tool_call>"]
    enable_history_code_execution = False
    force_run_test_cases = True
    done_without_error = True # passive
    python_path = None
    pre_import_lib = False
    
    def get_usage_inst(self):
        return "You are able to write and execute Python code securely inside a Firejail sandbox."

    def conduct_action(self, trajectory_id, action, extra_field):
        """
        Execute the parsed action in a Firejail sandbox.
        
        Args:
            trajectory_id: ID for tracking the action
            action: Raw action string
            extra_field: Additional parameters
            
        Returns:
            Tuple containing observation, done flag, and validity flag
        """
        parsed_action, is_valid = self.parse_action(action)
        env = self.load_env(trajectory_id)
        
        if not is_valid:
            # observation = "No valid Python code found. Please provide code in either <python>...</python> tags or ```python...``` code blocks."
            observation = ""
            execution_result = ""
            done = False
            valid = False
        else:
            code_has_error = False
            # Extract stdin if provided in extra_field
            stdin = extra_field.get("stdin", "") if extra_field else None
            
            test_input = re.findall(r"```input\n(.*?)\n```", action, re.DOTALL)
            if len(test_input) > 0:
                stdin = test_input[0].strip()

            new_code = parsed_action # 
            if self.enable_history_code_execution:
                previous_parsed_code = [obs["action"] for obs in env["previous_obs"]]
                code_to_execute = wrap_code_blocks(previous_parsed_code + [parsed_action])

            else:
                code_to_execute = parsed_action
            # execution_result, has_error = execute_python_in_firejail(code_to_execute, self.timeout, stdin, self.python_path, self.pre_import_lib)
            execution_result = ""
        
            # if not has_error and self.force_run_test_cases:
            observation = ""
            test_cases = extra_field.get("public_tests", None) if extra_field else None
            if self.force_run_test_cases and test_cases is not None:
                # print(test_cases)
                if isinstance(test_cases, str):
                    test_cases = json.loads(test_cases) # [:10] # debug
                # execute the public test cases
                if isinstance(test_cases, list):
                    # acecoder data
                    # list of assert
                    for test_case_i in test_cases:
                        test_codes = code_to_execute + "\n" + test_case_i # plus an assert test
                        test_stdout, test_stderr, has_error = execute_python(test_codes, self.timeout, stdin, self.python_path, self.pre_import_lib)
                        if has_error:
                            test_cases_passed = False
                            break
                    if test_cases_passed:
                        test_result = "\nAll public test cases passed!"
                    else:
                        test_result = f"The above code is incorrect. \nFailed test case: {test_case_i}\nError:{test_stdout}\n{test_stderr}"
                        code_has_error = True
                elif isinstance(test_cases, dict):
                    # deepcoder data
                    assert "inputs" in test_cases and "outputs" in test_cases, f"Invalid test cases format: {test_cases.keys()}"
                    test_result = ""
                    test_cases_passed = True
                    for i in range(len(test_cases["inputs"])):
                        input_case = test_cases["inputs"][i]
                        output_case = test_cases["outputs"][i]
                        
                        if "fn_name" in test_cases:
                            if isinstance(input_case, str):
                                try:
                                    input_arg = json.loads(input_case)
                                except json.JSONDecodeError:
                                    input_arg = input_case
                                if isinstance(output_case, str):
                                    try:
                                        expected_return = json.loads(output_case)
                                    except json.JSONDecodeError:
                                        expected_return = output_case
                                elif isinstance(output_case, list):
                                    expected_return = ", ".join([str(x) for x in output_case])
                                    if len(output_case) > 1:
                                        expected_return = f"({expected_return})"
                                else:
                                    raise ValueError(f"Invalid output case format: {output_case}")
                            elif isinstance(input_case, list):
                                input_arg = ", ".join([str(x) for x in input_case])
                                if isinstance(output_case, str):
                                    expected_return = output_case
                                elif isinstance(output_case, list):
                                    expected_return = ", ".join([str(x) for x in output_case])
                                    if len(output_case) > 1:
                                        expected_return = f"({expected_return})" # men_still_standing([]) == [11,11]
                                else:
                                    raise ValueError(f"Invalid output case format: {output_case}")
                            else:
                                raise ValueError(f"Invalid input case format: {input_case}")
                              
                            test_codes = code_to_execute + f"\nassert {test_cases['fn_name']}({input_arg}) == {expected_return}\n"
                            test_codes = code_to_execute + f"\nprint({test_cases['fn_name']}({input_arg}))\n"
                            
                            test_stdin = stdin
                            test_stdout, test_stderr, has_error = execute_python(test_codes, self.timeout, test_stdin, self.python_path, self.pre_import_lib)
                            # debug
                            test_case_output_match = custom_compare(test_stdout, expected_return)
                            if not test_case_output_match:
                                test_cases_passed = False
                                # print(f"The above code is incorrect and got a wrong answer.\nInput: {input_case}\nGenerated Output: {test_stdout}\nExpected: {expected_return}")
                        else:
                            # preprocess input case and output case
                            if isinstance(input_case, list):
                                input_case = "\n".join([str(x) for x in input_case if str(x).strip() != ""])
                            if isinstance(output_case, list):
                                output_case = "\n".join([str(x) for x in output_case if str(x).strip() != ""])

                            test_codes = code_to_execute
                            test_stdin = (stdin + input_case)
                            test_stdout, test_stderr, has_error = execute_python(test_codes, self.timeout, test_stdin, self.python_path, self.pre_import_lib)
                            test_case_output_match = custom_compare(test_stdout, output_case)

                            # print(f"\n\nDEBUG: Running test case {i+1} with input={input_case}, output={output_case}\n\n")
                            # print(f"Test stdin: {test_stdin}")
                            # print("Test stdout:", json.dumps(test_stdout))
                            # print("Test stderr:", test_stderr)
                            # print("Has error:", has_error)
                            # print("Expected output:", json.dumps(output_case))
                            # print(f"Test case output match: {test_case_output_match}")
                            
                            if not test_case_output_match or has_error:
                                test_cases_passed = False
                                # print(f"The above code is incorrect and got a wrong answer.\nInput: {input_case}\nGenerated Output: {test_stdout}\nExpected: {output_case}")
                        if not test_cases_passed:
                            break
                        
                    message = ""
                    
                    # match non-passed generations
                    if not test_cases_passed:
                        metadata = {
                            "error": test_stderr,
                            "inputs": input_case,
                            "expected": output_case,
                            "output": test_stdout,
                        }
                        
                        # not runtime err or time-limit exceeded
                        if not has_error:
                            # case: wrong answer
                            message = f"The above code is incorrect and got a wrong answer.\nInput: {metadata['inputs']}\nGenerated Output: {metadata['output']}\nExpected: {metadata['expected']}"
                        else:
                            # time limit exceeded
                            if "execution timed out" in observation.lower():
                                message = f"The above code is incorrect and got time limit exceeded.\n{metadata['error']}\nInput: {metadata['inputs']}\nExpected: {metadata['expected']}"
                            elif "syntaxerror" in observation.lower():
                                message = f"The above code is incorrect and got a syntax error.\nInput: {metadata['inputs']}\nExpected: {metadata['expected']}\n{metadata['error']}"
                            else:
                                message = f"The above code is incorrect and got a runtime error.\nInput: {metadata['inputs']}\nExpected: {metadata['expected']}\n{metadata['error']}"
                        test_result = message
                        code_has_error = True
                    else:
                        test_result = "All public test cases passed!\n"
                else:
                    raise ValueError(f"Invalid test cases format: {test_cases}")
                observation = test_result
                    
            if self.done_without_error:
                if code_has_error:
                    done = False
                else:
                    done = True
            else: 
                done = False
            valid = True
        
        self.update_env(trajectory_id, env, parsed_action, is_valid, extra_field, execution_result)
        self.save_env(trajectory_id, env)
        
        return observation, done, valid
        