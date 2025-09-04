import json
from typing import List, Dict, Any
import re
import ast
from tau2_train.data_model.tasks import Task

def parse_arguments_advanced(args_str: str) -> Dict[str, Any]:

    arguments = {}
    current_key = None
    current_value = []
    depth = 0  
    
    parts = []
    start = 0
    for i, char in enumerate(args_str):
        if char in '[{':
            depth += 1
        elif char in ']}':
            depth -= 1
        elif char == ',' and depth == 0:
            parts.append(args_str[start:i].strip())
            start = i + 1
    parts.append(args_str[start:].strip())
    
    for part in parts:
        if '=' in part:
            key, value_str = part.split('=', 1)
            key = key.strip()
            
            try:
                if (value_str.startswith('"') and value_str.endswith('"')) or \
                   (value_str.startswith("'") and value_str.endswith("'")):
                    value = value_str[1:-1]
                elif value_str.startswith('[') and value_str.endswith(']'):
                    value = ast.literal_eval(value_str)
                elif value_str.startswith('{') and value_str.endswith('}'):
                    value = ast.literal_eval(value_str)
                elif value_str in ('True', 'False', 'None'):
                    value = ast.literal_eval(value_str)
                elif '.' in value_str and value_str.replace('.', '').isdigit():
                    value = float(value_str)
                elif value_str.isdigit():
                    value = int(value_str)
                else:
                    value = value_str
            except (ValueError, SyntaxError):
                value = value_str
            
            arguments[key] = value
    
    return arguments

def parse_function_calls_advanced(function_calls_str: str, data_id: str) -> List[Dict[str, Any]]:

    function_calls = ast.literal_eval(function_calls_str)
    result = []
    
    for i, func_call in enumerate(function_calls):
        match = re.match(r'(\w+)\((.*)\)', func_call)
        if not match:
            continue
            
        func_name = match.group(1)
        args_str = match.group(2)
        
        arguments = parse_arguments_advanced(args_str)

        if func_name == "transfer_to_human_agents":
            arguments = {
                "summary": arguments["summary"]
            }
        
        action = {
            "action_id": f"{data_id}_{i}",
            "name": func_name,
            "arguments": arguments,
            "info": None
        }
        
        result.append(action)
    
    return result

def parse_nl_assertions(rubrics_str):
    if isinstance(rubrics_str, list):
        # breakpoint()
        return rubrics_str

    pattern = r'\d+\)\s*(.*?)(?=\s*\d+\)|$)'
    matches = re.findall(pattern, rubrics_str, re.DOTALL)
    
    result = []
    for match in matches:
        cleaned_text = match.strip()
        if not cleaned_text.endswith('.'):
            cleaned_text += '.'
        result.append(f"The agent should {cleaned_text}")
    
    return result


# fname = "tau_airline_mt_dialogs_0902_new"
fname = "tau_airline_mt_dialogs_0903_all"

with open(f"/storage/openpsi/users/xushusheng.xss/data/agent_training/tau_bench/{fname}.jsonl") as f, \
    open(f"/storage/openpsi/users/xushusheng.xss/data/agent_training/tau_bench/{fname}_format_fixs.jsonl", 'w') as fout:

    for idx, line in enumerate(f):
        data = json.loads(line)

        try:

            if isinstance(data['user_simulator']['known_info'], str):
                known_info = data['user_simulator']['known_info']
            elif isinstance(data['user_simulator']['known_info'], dict):
                known_info = json.dumps(data['user_simulator']['known_info'])
            else:
                raise RuntimeError
            cur_data = dict(
                        id="train" + "_" + str(idx),
                        description=dict(
                            purpose=data['metadata'].get("purpose"),
                        ),
                        user_scenario=dict(
                            instructions=dict(
                                task_instructions=data['user_simulator']['task_instructions'],
                                domain="airline",
                                reason_for_call=data['user_simulator']["reason_for_call"],
                                known_info=known_info
                            )
                        ),
                        evaluation_criteria=dict(
                            actions=parse_function_calls_advanced(data['eval']['functions'], "train" + "_" + str(idx)),
                            communicate_info=[],
                            nl_assertions=parse_nl_assertions(data['eval']['rubrics'])      
                        )

                    )
            task = Task.model_validate(cur_data)
            
        except Exception as e:
            print(idx, e)
            # breakpoint()
            continue
        cur_data["evaluation_criteria"] = json.dumps(cur_data["evaluation_criteria"])
        print(json.dumps(cur_data), file=fout)
    