import json

from loguru import logger
from tau2_train.data_model.simulation import DBCheck, EnvAssertionCheck
from tau2_train.data_model.tasks import Task
from tau2_train.domains.airline.environment import get_environment

fname = "/storage/openpsi/users/xushusheng.xss/data/agent_training/tau_bench/tau_airline_mt_dialogs_0902_new_format_fixs.jsonl"
# fname = "tau2_train/data/tau2/domains/airline/tasks.json"
# fname = "/storage/openpsi/users/xushusheng.xss/data/agent_training/tau_bench/tau_airline_mt_dialogs_0903_all_format_fixs.jsonl"
# fname = "tau2_train/data/tau2/domains/airline/no_op_task.jsonl"
# fname = "tau2_train/data/tau2/domains/airline/op_task.jsonl"
# fname = "/storage/openpsi/users/xushusheng.xss/data/agent_training/tau_bench/tau_airline_mt_dialogs_v0_format_fix.jsonl"


def load_data(fname):
    if fname.endswith(".jsonl"):
        with open(fname) as f:
            datas = [json.loads(x) for x in f]
            for data in datas:
                data["evaluation_criteria"] = json.loads(data["evaluation_criteria"])
    else:
        with open(fname) as f:
            datas = json.load(f)
    return datas


is_match = []
error = 0
taskid2_match = {}

for data in load_data(fname):

    task = Task.model_validate(data)

    # if task.id == "train_39":
    #     breakpoint()
    # else:
    #     continue

    initialization_data = None
    if (
        task.initial_state is not None
        and task.initial_state.initialization_data is not None
    ):
        initialization_data = task.initial_state.initialization_data

    initialization_actions = None
    if (
        task.initial_state is not None
        and task.initial_state.initialization_actions is not None
    ):
        initialization_actions = task.initial_state.initialization_actions

    message_history = []
    if (
        task.initial_state is not None
        and task.initial_state.message_history is not None
    ):
        message_history = task.initial_state.message_history

    gold_environment = get_environment()
    gold_environment.set_state(
        initialization_data=initialization_data,
        initialization_actions=initialization_actions,
        message_history=message_history,
    )

    golden_actions = task.evaluation_criteria.actions or []
    is_error = False
    for action in golden_actions:
        try:
            gold_environment.make_tool_call(
                tool_name=action.name,
                requestor=action.requestor,
                **action.arguments,
            )
            logger.info(f"{action} completed")
        except Exception as e:
            logger.warning(
                f"Error in golden actions {action.name}({action.arguments}): {e}"
            )

            is_error = True
            break
            # breakpoint()
    if is_error:
        taskid2_match[task.id] = "error"
        error += 1
        continue

    ref_environment = get_environment()
    ref_environment.set_state(
        initialization_data=initialization_data,
        initialization_actions=initialization_actions,
        message_history=message_history,
    )

    agent_db_hash = gold_environment.get_db_hash()
    user_db_hash = gold_environment.get_user_db_hash()
    ref_agent_db_hash = ref_environment.get_db_hash()
    ref_user_db_hash = ref_environment.get_user_db_hash()
    agent_db_match = agent_db_hash == ref_agent_db_hash
    user_db_match = user_db_hash == ref_user_db_hash

    if agent_db_match and user_db_match:
        db_reward = 1.0
        db_match = True
    else:
        db_reward = 0.0
        db_match = False

    db_check = DBCheck(db_match=db_match, db_reward=db_reward)

    # Run env assertions
    env_assertions = task.evaluation_criteria.env_assertions or []
    env_assertion_checks = []
    env_assertion_reward = 1.0
    for env_assertion in env_assertions:
        success = ref_environment.run_env_assertion(
            env_assertion,
            raise_assertion_error=False,
        )
        res = EnvAssertionCheck(
            env_assertion=env_assertion,
            met=success,
            reward=1.0 if success else 0.0,
        )
        env_assertion_checks.append(res)
        env_assertion_reward *= res.reward

    reward = db_reward * env_assertion_reward
    is_match.append(reward)
    taskid2_match[task.id] = reward

    print(sum(is_match), error, len(is_match))
    # print(is_match)


with open(fname.split(".json")[0] + "_no_op_acc.json", "w") as f:
    json.dump(taskid2_match, f, indent=2)
