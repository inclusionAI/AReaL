from .base import BaseTool, register_tool
import regex as re
import subprocess
import os
import signal
import sys
import json
import uuid
import hashlib
from typing import Tuple, Dict, Any, Optional
from .utils.sql_executor import sql_observation

# Timeout for code execution in seconds
TIMEOUT = 5

import concurrent.futures

def run_with_timeout(func, args=(), kwargs=None, timeout=None):
    if kwargs is None:
        kwargs = {}
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise Exception(f"Function timed out after {timeout} seconds")


SQL_START, SQL_END = "<sql>", "</sql>"
SOLUTION_START, SOLUTION_END = "<solution>", "</solution>"
OBS_START, OBS_END = "<observation>", "</observation>"

@register_tool
class SqlTool(BaseTool):
    tool_type = "sql"
    timeout = TIMEOUT
    stop_tokens = ["```output", "<output>", "<tool_call>", OBS_START, SQL_END]
    enable_history_code_execution = False
    enable_mannual_reflection = False
    force_run_test_cases = False
    done_without_error = False
    
    def get_usage_inst(self):
        return "You can execute SQL queries using <sql>...</sql> tags for intermediate verification or <solution>...</solution> tags for final answers."
    
    def parse_action(self, action: str, tag_type: str = "sql") -> Tuple[str, bool]:
        """
        Parse the raw action string to extract SQL code from either <sql></sql> or <solution></solution> tags.
        
        Args:
            action: Raw action string containing SQL code
            tag_type: Type of tag to extract ("sql" or "solution")
            
        Returns:
            Tuple containing the extracted code and a validity flag
        """
        tag_start_map = {
            "sql": SQL_START,
            "solution": SOLUTION_START
        }
        tag_end_map = {
            "sql": SQL_END,
            "solution": SOLUTION_END
        }

        # Find the last occurrence of the start tag
        start_tag = tag_start_map[tag_type]
        end_tag = tag_end_map[tag_type]
        
        sql_code_start_idx = action.rfind(start_tag)
        if sql_code_start_idx == -1:
            return "", False
        
        # Find the corresponding end tag after the start tag
        sql_code_end_idx = action.find(end_tag, sql_code_start_idx + len(start_tag))
        if sql_code_end_idx == -1:
            return "", False
        
        # Extract the content between the tags
        sql_code = action[sql_code_start_idx + len(start_tag):sql_code_end_idx].strip()
        return sql_code, True

    
    def conduct_action(self, trajectory_id, action, extra_field):
        """
        Execute the parsed SQL code and return observation.
        
        Args:
            trajectory_id: ID for tracking the action
            action: Raw action string
            extra_field: Additional parameters (db_id, db_path, gt_sql, current_step, max_turns, turns_left)
            
        Returns:
            Tuple containing observation, done flag, and validity flag
        """
        
        # first try to parse the code as if from <sql></sql> tags (intermediate interaction)
        parsed_action, is_valid = self.parse_action(action, "sql")
        env = self.load_env(trajectory_id)
        
        # Extract turn information from extra_field
        turns_left = extra_field.get("turns_left", 0) if extra_field else 0
        current_step = extra_field.get("current_step", 0) if extra_field else 0
        max_turns = extra_field.get("max_turns", 0) if extra_field else 0
        
        # print("==>")
        # print(f"===> turns_left", turns_left)
        # print(f"===> current_step", current_step)
        # print(f"===> max_turns", max_turns)
        # print(f"\n\n===> action", action)
        # print(f"\n\n===> parsed_action", parsed_action)
        # print("="*100)
        
        if not is_valid:
            # if not valid, try to parse the code as if from <solution></solution> tags (final answer)
            parsed_action, is_valid = self.parse_action(action, "solution")
            
            # case: it IS the final answer, mark the trajectory as done and leave it to to the reward manager
            if is_valid:
                observation = ""
                execution_result = ""
                done = True
                valid = False
            # neither tags are found, mark the trajectory as invalid
            else:
                observation = "Your previous action is invalid. Follow the format of outputting thinking process and sql tool, and try again."
                execution_result = ""
                done = False
                valid = False
        else:
            # call the sql tool to execute the sql code
            try:
                # Extract database information from extra_field
                db_id = extra_field.get("db_id", None) if extra_field else None
                db_path = extra_field.get("db_path", None) if extra_field else None
                gold_sql = extra_field.get("gt_sql", None) if extra_field else None

                # assemble the meta information to call the sql executor (score function)
                meta = {
                    "db_id": db_id,
                    "gold_sql": gold_sql,
                    "cmp_method": "bird",
                    "db_path": db_path
                }   
                
                # correctness, execution_result, error_message = score(parsed_action, meta)
                observation = sql_observation(parsed_action, meta, timeout=5)
                
                # if error_message and not correctness:
                # if error_message != "":
                #     if execution_result:
                #         observation = f"\n{error_message}\n\nExecution Result:\n{execution_result}"
                #     else:
                #         observation = f"\n{error_message}"
                # else:
                #     observation = f"Execution Result:\n{execution_result}"
                
                
                
                # Only mark as done if this is a final solution submission and it's correct
                done = False    # we use <sql></sql> here so this must be intermediate
                valid = True        
            except Exception as e:
                error_message = str(e)
                observation = f"Execution Error:\n{error_message}"
                done = False
                valid = False  # Code was extracted validly, just failed to execute
        
        # Create reminder text with turns left information
        reminder_text = f"<reminder>You have {turns_left} turns left to complete the task.</reminder>"
        
        # if turns_left > 0:
        #     reminder_text = f"<reminder>You have {turns_left} turns left to complete the task.</reminder>"
        # else:
        #     reminder_text = f"<reminder>This is your final turn. Please provide your final answer using the <solution></solution> tags.</reminder>"
        
        obs = f"\n\n<observation>{observation}\n{reminder_text}</observation>\n\n"
        
        self.update_env(trajectory_id, env, parsed_action, is_valid, extra_field, observation)
        self.save_env(trajectory_id, env)
        
        obs = {
            "obs": obs,
            "parsed_sql": parsed_action,
        }
        
        return obs, done, valid
        