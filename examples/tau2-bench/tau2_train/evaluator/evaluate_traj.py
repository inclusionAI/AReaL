import json
import os

from tau2_train.data_model.simulation import RewardInfo
from tau2_train.data_model.tasks import RewardType, Task
from tau2_train.domains.airline.environment import get_environment
from tau2_train.evaluator.evaluator import EvaluationType
from tau2_train.evaluator.evaluator_action import ActionEvaluator
from tau2_train.evaluator.evaluator_communicate import CommunicateEvaluator
from tau2_train.evaluator.evaluator_env import EnvironmentEvaluator
from tau2_train.evaluator.evaluator_nl_assertions import NLAssertionsEvaluator
from tau2_train.utils.llm_utils import to_tau2_messages


def load_trajs(fname):
    with open(fname) as f:
        data = [json.loads(x) for x in f]
    return data


def load_tasks(fname):
    out_data = {}
    with open(fname) as f:
        for line in f:
            data = json.loads(line)
            data["evaluation_criteria"] = json.loads(data["evaluation_criteria"])
            task = Task.model_validate(data)
            out_data[task.id] = task
    return out_data


def evaluate_simulation(
    messages,
    task: Task,
    evaluation_type,
    solo_mode: bool = False,
):
    """
    Evaluate the simulation based on the evaluation type.
    """
    # breakpoint()
    if task.evaluation_criteria is None:
        return RewardInfo(
            reward=1.0,
            info={"note": "No evaluation criteria"},
        )
    if evaluation_type == EvaluationType.ENV:
        reward_info = EnvironmentEvaluator.calculate_reward(
            environment_constructor=get_environment,
            task=task,
            full_trajectory=messages,
            solo_mode=solo_mode,
        )
    elif evaluation_type == EvaluationType.NL_ASSERTIONS:
        reward_info = NLAssertionsEvaluator.calculate_reward(
            task=task,
            full_trajectory=messages,
            llm_name="/storage/openpsi/models/Qwen__Qwen2.5-72B-Instruct/",
            api_key="empty",
            base_url="http://33.180.164.231:30000/v1/",
        )
    elif evaluation_type == EvaluationType.COMMUNICATE:
        reward_info = CommunicateEvaluator.calculate_reward(
            task=task,
            full_trajectory=messages,
        )
    elif evaluation_type == EvaluationType.ACTION:
        reward_info = ActionEvaluator.calculate_reward(
            task=task,
            full_trajectory=messages,
        )
    elif evaluation_type == EvaluationType.ALL:
        env_reward_info = EnvironmentEvaluator.calculate_reward(
            environment_constructor=get_environment,
            task=task,
            full_trajectory=messages,
            solo_mode=solo_mode,
        )
        action_reward_info = ActionEvaluator.calculate_reward(
            task=task,
            full_trajectory=messages,
        )
        communicate_reward_info = CommunicateEvaluator.calculate_reward(
            task=task,
            full_trajectory=messages,
        )
        # breakpoint()
        nl_reward_info = NLAssertionsEvaluator.calculate_reward(
            task=task,
            full_trajectory=messages,
            llm_name="/storage/openpsi/models/Qwen__Qwen2.5-72B-Instruct/",
            api_key="empty",
            base_url="http://33.180.164.231:30000/v1/",
        )
        # breakpoint()
        ## Combine all the rewards.
        reward = 1.0
        env_bases = {RewardType.DB, RewardType.ENV_ASSERTION}
        action_bases = {RewardType.ACTION}
        nl_bases = {RewardType.NL_ASSERTION}
        comm_bases = {RewardType.COMMUNICATE}
        task_reward_basis = set(task.evaluation_criteria.reward_basis)

        reward_breakdown = {}
        if task_reward_basis & env_bases:
            if env_reward_info.reward_breakdown is not None:
                reward_breakdown.update(env_reward_info.reward_breakdown)
            reward *= env_reward_info.reward
        if task_reward_basis & action_bases:
            if action_reward_info.reward_breakdown is not None:
                reward_breakdown.update(action_reward_info.reward_breakdown)
            reward *= action_reward_info.reward
        if task_reward_basis & nl_bases:
            if nl_reward_info.reward_breakdown is not None:
                reward_breakdown.update(nl_reward_info.reward_breakdown)
            reward *= nl_reward_info.reward
        if task_reward_basis & comm_bases:
            if communicate_reward_info.reward_breakdown is not None:
                reward_breakdown.update(communicate_reward_info.reward_breakdown)
            reward *= communicate_reward_info.reward

        reward_info = RewardInfo(
            reward=reward,
            db_check=env_reward_info.db_check,
            env_assertions=env_reward_info.env_assertions,
            action_checks=action_reward_info.action_checks,
            nl_assertions=nl_reward_info.nl_assertions,
            communicate_checks=communicate_reward_info.communicate_checks,
            reward_basis=task.evaluation_criteria.reward_basis,
            reward_breakdown=reward_breakdown,
            info={
                "env": env_reward_info.info,
                "nl": nl_reward_info.info,
                "communicate": communicate_reward_info.info,
                "action": action_reward_info.info,
            },
        )
    else:
        raise ValueError(f"Unknown evaluation type: {evaluation_type}")
    breakpoint()
    return reward_info


if __name__ == "__main__":

    task_fname = "/storage/openpsi/users/xushusheng.xss/data/agent_training/tau_bench/tau_airline_mt_dialogs_v0_format_fix.jsonl"
    tasks = load_tasks(task_fname)

    fname = "/storage/openpsi/experiments/logs/admin/xss-tau2-qwq_32B-train-v4/trial-25/generated/0/train_88_758a5f23139649e0bb3d69bd66848a73.jsonl"
    trajs = load_trajs(fname)

    for traj in trajs:
        task_id = f"train_{os.path.basename(fname).split('_')[1]}"
        task = tasks[task_id]
        messages = to_tau2_messages(traj["messages"])
        reward_info = evaluate_simulation(messages, task, "all")

        print(reward_info)
