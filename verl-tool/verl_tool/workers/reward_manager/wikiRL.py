import nltk
import json
import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score

import os
import time
import asyncio
import regex as re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from verl.workers.reward_manager import register

from mini_webarena.rl_utils import format_score
from mini_webarena.evaluator import metric_heuristic
# ------------------------------------------------------------------------------
# WikiRL Reward Manager
# ------------------------------------------------------------------------------

def clean_text(text):
    # 删除控制字符 & 非打印字符
    return re.sub(r'[\x00-\x1F\x7F-\x9F\u200b-\u200f\u2028-\u202f\u2060-\u206f]', '', text)

@register("wikiRL")
class WikiRLRewardManager:
    """
    Reward Manager for the WikiRL dataset.

    This class computes a combined reward for each predicted answer by comparing it with
    the ground truth answers. The final reward is a weighted combination of a fuzzy matching
    score and a structure score.
    # """
    def __init__(self, tokenizer=None, num_examine=1, compute_score=None, **kwargs) -> None:
        """
        Initialize the WikiRLRewardManager.

        Parameters:
        - fuzzy_weight: The weight applied to the fuzzy matching score.
        - structure_weight: The weight applied to the structure score.
        """
        if tokenizer is None:
            # Simply use QWen2.5-7B tokenizer
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.fuzzy_weight = 0.7
        self.structure_weight = 0.3
        if "record_dir" in kwargs:
            self.record_dir = Path(kwargs['record_dir'])
            self.record_dir.mkdir(parents=True, exist_ok=True)

    def answer_score(self, pred, ground_truths):
        def extract_last_stop_content(input_str: str) -> str:
            matches = re.findall(r"```stop\s*\[([^\]]*)\]```", input_str)
            if matches:
                return matches[-1]
            return ""
        # First match ```stop [...]``` use regex to find the last ```stop [...]``` in the string
        pred = extract_last_stop_content(pred)
        score = metric_heuristic(ground_truths, pred)
        # print("answer score", ground_truths, pred, score)
        return score

    def format_score(self, actions):
        # Average of format_score
        scores = [format_score(action) for action in actions]
        return sum(scores) / len(scores) if scores else 0

    # def __call__(self, data:DataProto):
    #     # Retrieve the list of predicted responses.
    #     # print("")
    #     # print(data)
    #     # import pickle
    #     # with open("data_stub.pkl", "wb") as f:
    #     #     pickle.dump(data, f)
    #     pass

    def __call__(self, data: DataProto):
        """
        Compute scalar rewards for a batch and append per‑sample logs to
        ``reward_manager_history.jsonl``.

        Each JSON line now stores token‑separated strings (not raw ID lists):

        {
            "uid": <trajectory_uid>,
            "input_tokens": "▁This ▁is ▁a ...",      # whitespace‑joined tokens
            "pred_tokens": "▁Answer ▁text ...",
            "actions": [...],
            "observations": [...],
            "answer_score": <float>,
            "format_score": <float>
        }
        """
        print("")
        print(data)
        print(len(data))
        import pickle
        with open("data_stub_new_qwq.pkl", "wb") as f:
            pickle.dump(data, f)

        import json
        from pathlib import Path

        special_token_ids = set(self.tokenizer.all_special_ids)

        actions_list, observations_list, response_list = [], [], []

        # ---------- 1.  decode actions / obs / responses --------------------
        for i in range(len(data)):
            input_ids = data.batch["input_ids"][i].tolist()
            attention_mask = data.batch["attention_mask"][i].tolist()
            action_lens = data.non_tensor_batch["action_lengths"][i]
            obs_lens = data.non_tensor_batch["obs_lengths"][i]

            prompt_len = 2048
            resp_ids   = input_ids[prompt_len:]
            resp_mask  = attention_mask[prompt_len:]
            resp_tokens = [
                tid for tid, m in zip(resp_ids, resp_mask)
                if m == 1 and tid not in special_token_ids
            ]
            resp_text = self.tokenizer.decode(resp_tokens,
                                              skip_special_tokens=True).strip()
            response_list.append(resp_text)

            cursor, actions, observations = 0, [], []
            for a_len, o_len in zip(action_lens, obs_lens):
                actions.append(self.tokenizer.decode(
                    resp_tokens[cursor:cursor + a_len - 1],
                    skip_special_tokens=True).strip())
                cursor += a_len - 1
                observations.append(self.tokenizer.decode(
                    resp_tokens[cursor:cursor + o_len - 1],
                    skip_special_tokens=True).strip())
                cursor += o_len - 1
            if cursor < len(resp_tokens):
                actions.append(self.tokenizer.decode(
                    resp_tokens[cursor:],
                    skip_special_tokens=True).strip())

            actions_list.append(actions)
            observations_list.append(observations)

        # ---------- 2.  reward tensor --------------------------------------
        prompt_ids   = data.batch["prompts"]
        prompt_len   = prompt_ids.shape[-1]
        responses_id = data.batch["responses"]
        valid_resp_len = data.batch["attention_mask"][:, prompt_len:].sum(dim=-1)
        reward_tensor = torch.zeros_like(responses_id, dtype=torch.float32)

        answer_scores, format_scores = [], []

        for i in range(len(data)):
            gts = data.non_tensor_batch["reward_model"][i]["ground_truth"]
            pred = response_list[i]
            answer_reward  = self.answer_score(pred, gts)
            format_reward  = self.format_score(actions_list[i])
            final_reward   = answer_reward + 0.5 * format_reward

            reward_tensor[i, valid_resp_len[i].item() - 1] = final_reward
            answer_scores.append(answer_reward)
            format_scores.append(format_reward)

        # ---------- 3.  persistent logging ---------------------------------
        try:
            log_file = Path("reward_manager_history.jsonl")
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with log_file.open("a", encoding="utf-8") as f:
                for idx in range(len(data)):
                    # convert entire sequence and prediction to whitespace‑joined tokens
                    input_text = clean_text(self.tokenizer.decode(
                        data.batch["input_ids"][idx].tolist(),
                        skip_special_tokens=True
                    ).strip())
                    input_tokens = " ".join(self.tokenizer.tokenize(input_text))
                    pred_tokens = " ".join(self.tokenizer.tokenize(clean_text(response_list[idx])))

                    log_entry = {
                        "uid": data.non_tensor_batch.get("uid", [None]*len(data))[idx],
                        "input_tokens": input_tokens,
                        "pred_tokens":  pred_tokens,
                        "actions": actions_list[idx],
                        "observations": observations_list[idx],
                        "answer_score": answer_scores[idx],
                        "format_score": format_scores[idx],
                    }
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[WARN] could not append to reward_manager_history.jsonl: {e}")

        print(f"Computed rewards for {len(data)} samples.")
        print("Answer scores:", answer_scores)
        print("Format scores:", format_scores)
        exit(1)
        return reward_tensor


if __name__ == '__main__':
    import pickle

    # Load the saved data object from disk
    with open("data_stub_new.pkl", "rb") as f:
        dummy_data = pickle.load(f)

    # Instantiate the WikiRLRewardManager (you can pass in config if needed)
    reward_manager = WikiRLRewardManager()

    # Compute rewards for the loaded data
    rewards = reward_manager(dummy_data)
    print("Rewards:", rewards)


"""
(TaskRunner pid=2019847) ==== Call WikiRLRewardManager ====
(TaskRunner pid=2019847) DataProto(batch=TensorDict(
(TaskRunner pid=2019847)     fields={
(TaskRunner pid=2019847)         attention_mask: Tensor(shape=torch.Size([4, 8192]), device=cpu, dtype=torch.int64, is_shared=False),
(TaskRunner pid=2019847)         loss_mask: Tensor(shape=torch.Size([4, 8192]), device=cpu, dtype=torch.int64, is_shared=False),
(TaskRunner pid=2019847)         input_ids: Tensor(shape=torch.Size([4, 8192]), device=cpu, dtype=torch.int64, is_shared=False),
(TaskRunner pid=2019847)         old_log_probs: Tensor(shape=torch.Size([4, 4096]), device=cpu, dtype=torch.float32, is_shared=False),
(TaskRunner pid=2019847)         position_ids: Tensor(shape=torch.Size([4, 8192]), device=cpu, dtype=torch.int64, is_shared=False),
(TaskRunner pid=2019847)         prompts: Tensor(shape=torch.Size([4, 4096]), device=cpu, dtype=torch.int64, is_shared=False),
(TaskRunner pid=2019847)         ref_log_prob: Tensor(shape=torch.Size([4, 4096]), device=cpu, dtype=torch.float32, is_shared=False),
(TaskRunner pid=2019847)         responses: Tensor(shape=torch.Size([4, 4096]), device=cpu, dtype=torch.int64, is_shared=False),
(TaskRunner pid=2019847)         responses_with_loss_mask: Tensor(shape=torch.Size([4, 4096]), device=cpu, dtype=torch.int64, is_shared=False)},
(TaskRunner pid=2019847)     batch_size=torch.Size([4]),
(TaskRunner pid=2019847)     device=None,
(TaskRunner pid=2019847)     is_shared=False), non_tensor_batch={'data_source': array(['wiki_qa', 'wiki_qa', 'wiki_qa', 'wiki_qa'], dtype=object), 'ability': array(['wiki', 'wiki', 'wiki', 'wiki'], dtype=object), 'reward_model': array([{'ground_truth': array(['Ginnifer Goodwin'], dtype=object), 'style': 'rule'},
(TaskRunner pid=2019847)        {'ground_truth': array(['Ginnifer Goodwin'], dtype=object), 'style': 'rule'},
(TaskRunner pid=2019847)        {'ground_truth': array(['Natalia Gastiain Tena'], dtype=object), 'style': 'rule'},
(TaskRunner pid=2019847)        {'ground_truth': array(['Natalia Gastiain Tena'], dtype=object), 'style': 'rule'}],
(TaskRunner pid=2019847)       dtype=object), 'index': array([0, 0, 0, 0], dtype=object), 'uid': array(['ca6a0e8e-6821-4a00-8a0c-5049019e7da7',
(TaskRunner pid=2019847)        'ca6a0e8e-6821-4a00-8a0c-5049019e7da7',
(TaskRunner pid=2019847)        'b58d9f7c-48c6-487f-911f-10db4a2f7b2b',
(TaskRunner pid=2019847)        'b58d9f7c-48c6-487f-911f-10db4a2f7b2b'], dtype=object)}, meta_info={'turns_stats': [4, 4], 'active_mask': [True, True], 'valid_action_stats': [4, 4], 'global_token_num': [5541, 5541, 3697, 5542], 'temperature': 0.9})
"""
