import asyncio
from datetime import datetime
from functools import reduce
from typing import List, Dict, Optional

import torch
from transformers.utils.dummy_pt_objects import TrajectoryTransformerModel

from realhf.api.core.agent_api import Agent, register_agent
from realhf.api.core.data_api import SequenceSample, load_hf_tokenizer
from realhf.api.core.env_api import EnvironmentService
from realhf.api.core.model_api import BundledGenerationOutputs
from realhf.base import constants, logging

logger = logging.getLogger("WerewolfAgent")


class WerewolfAgent(Agent):
    """Multi-turn agent for the werewolf game."""

    def __init__(
        self,
        gconfig,
        tokenizer_path,
        reward_scaling: float = 1.0,
        reward_bias: float = 0.0,
        turn_level_discount: float = 1,
        num_turns: int = 40,
        role: str = "villager",
        opponent_path: Optional[str] = None,
    ):
        self.gconfig = gconfig.new(n=1)
        self.tokenizer = load_hf_tokenizer(tokenizer_path)

        self.opponent_model = None
        self.opponent_tokenizer = None
        if opponent_path:
            from transformers import AutoTokenizer, AutoModelForCausalLM

            self.opponent_tokenizer = AutoTokenizer.from_pretrained(
                opponent_path, trust_remote_code=True
            )
            self.opponent_model = AutoModelForCausalLM.from_pretrained(
                opponent_path, torch_dtype=torch.float16, trust_remote_code=True,
            )
            self.opponent_model.eval()

        self.reward_scaling = reward_scaling
        self.reward_bias = reward_bias
        self.turn_level_discount = turn_level_discount
        self.num_turns = num_turns
        self.role = role

        logger.info(f"Initializing an agent with role {role}.")

    async def collect_trajectory(
        self,
        prompt: SequenceSample,
        env: EnvironmentService,
        obs_queue: asyncio.Queue,
        act_queue: asyncio.Queue,
    ) -> List[SequenceSample]:
        # --- 0) CONTEXT‐LENGTH SETTINGS -----------------------------
        MAX_NEW_TOKENS          = getattr(self.gconfig, "max_new_tokens", 2048)

        # --- 1) RESET & INITIALIZE ---------------------------------
        init_obs, _ = await env.reset()
        assert prompt.bs == 1 and self.gconfig.n == 1

        step_idx      = 0
        traj_len      = 0
        orig_qid      = prompt.ids[0]
        prompt_ids    = prompt.data["packed_prompts"].cpu().tolist()
        prefix_len    = len(prompt_ids)
        birth_time_ms = int(datetime.now().timestamp() * 1000)

        # initial observation tokens
        current_obs_ids = self.tokenizer(init_obs, add_special_tokens=False)["input_ids"]

        # --- 2) BUFFERS FOR THE FULL TRAJECTORY -------------------
        cumulative_rewards = []
        trajectories       = []

        version_start = None
        version_end   = None

        # --- 3) INTERACTIVE LOOP -----------------------------------
        for step_idx in range(self.num_turns):
            # 3.1) submit to model with a unique qid for this step --
            tagged_qid = f"{orig_qid}##step{step_idx}"

            if (((env.agent_role != "werewolf" and self.role == "werewolf")
                or (env.agent_role == "werewolf" and self.role == "villager"))
                and self.opponent_model):
                # logger.info(f"[DEBUG] Calling OPP generation with role {env.agent_role}.")
                # Use frozen opponent model for other roles
                input_ids = torch.tensor(current_obs_ids, dtype=torch.long).unsqueeze(0)
                input_ids = input_ids.to(self.opponent_model.device)
                with torch.inference_mode():
                    gen = self.opponent_model.generate(
                        input_ids,
                        max_new_tokens=512,
                        do_sample=True,
                    )
                gen_tokens = gen[0].tolist()[len(current_obs_ids) :]
                answer = self.opponent_tokenizer.decode(gen_tokens, skip_special_tokens=True)

                # print(f"[DEBUG] OPP generation: {answer}.")

                next_obs, raw_rewards, done, _, _ = await env.step((orig_qid, [answer]))
                r_v = raw_rewards[0] * self.reward_scaling - self.reward_bias
                r_w = raw_rewards[1] * self.reward_scaling - self.reward_bias
                cumulative_rewards.append((r_v, r_w))
                logger.info(f"{orig_qid}##step{step_idx} (OPP) finished with step reward {(r_v, r_w)}.")
                current_obs_ids = self.tokenizer(next_obs, add_special_tokens=False)["input_ids"]
                if done:
                    break
                else:
                    continue

            await obs_queue.put((tagged_qid, current_obs_ids, self.gconfig))
            act: BundledGenerationOutputs = await act_queue.get()

            # 3.1.1) recover original qid & sanity‐check
            recv_qid = act.qid
            base_qid, tag = recv_qid.split("##", 1)
            assert base_qid == orig_qid and tag == f"step{step_idx}"

            # 3.2) record version stamps ----------------------------
            if version_start is None:
                version_start = act.version_start[0]
            version_end = act.version_end[0]

            # 3.3) extract & CPU‐offload ----------------------------
            gen_part    = act.seqs[0][act.prompt_len:]
            n           = len(gen_part)
            logp_part   = act.logprobs[0][-n:]
            no_eos_flag = act.no_eos[0]

            gen_tokens   = gen_part.cpu().tolist()    if isinstance(gen_part, torch.Tensor) else list(gen_part)
            gen_logprobs = logp_part.cpu().tolist()   if isinstance(logp_part, torch.Tensor) else list(logp_part)

            # decode just this turn
            answer = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)

            # 3.4) step env & record reward ------------------------
            next_obs, raw_rewards, done, _, _ = await env.step((orig_qid, [answer]))
            r_v = raw_rewards[0] * self.reward_scaling - self.reward_bias
            r_w = raw_rewards[1] * self.reward_scaling - self.reward_bias
            cumulative_rewards.append((r_v, r_w))
            # logger.info(f"{orig_qid}##step{step_idx} finished with step reward {(r_v, r_w)}. The LLM resp is: {answer}.")
            logger.info(f"{orig_qid}##step{step_idx} finished with step reward {(r_v, r_w)}.")

            # tokenize & append info to buffer 
            tokens_i      = current_obs_ids + gen_tokens
            traj_len      += len(tokens_i)
            prompt_mask_i = [1] * len(current_obs_ids) + [0] * len(gen_tokens)
            logprobs_i    = [0.] * (len(current_obs_ids) - 1) + gen_logprobs[-len(gen_tokens):]

            if len(tokens_i) > MAX_NEW_TOKENS + len(current_obs_ids):
                logger.info(f"Trajectory {tagged_qid} is longer than {MAX_NEW_TOKENS}. Truncation is applied.")
                reduce_length = len(tokens_i) - MAX_NEW_TOKENS + len(current_obs_ids)
                tokens_i      = tokens_i[:-reduce_length]
                prompt_mask_i = prompt_mask_i[:-reduce_length]
                logprobs_i    = logprobs_i[:-reduce_length]

            trajectories.append((tokens_i, prompt_mask_i, logprobs_i, no_eos_flag, (r_v, r_w)))

            # Prepare for next step
            current_obs_ids = self.tokenizer(next_obs, add_special_tokens=False)["input_ids"]

            if done:
                break

        # --- 4) DISCOUNTED RETURN ---------------------------------
        rets = [0., 0.] # Villager, werewolf
        for rewards in reversed(cumulative_rewards):
            rets[0] = rewards[0] + rets[0] * self.turn_level_discount
            rets[1] = rewards[1] + rets[1] * self.turn_level_discount
        logger.info(f"Trajectory collected with final reward {rets}, steps {step_idx}.")

        # --- 5) LOGGING STATS  ----------------------
        stats: Dict = {}
        if hasattr(env, "get_stats"):
            try:
                stats = env.get_stats()
            except Exception:
                logger.error(f"No stats are available for trajectory {orig_qid}.")

        logging = []
        logging_keys = [
            "traj_steps",
            "traj_len", 
            "vill_rewards", 
            "were_rewards",
            "vill_wins",
            "were_wins",
            "werewolf_kills",
            "villager_correct_votes",
            "villager_wrong_votes",
            "witch_heals",
            "witch_poisons",
            "hunter_shots",
        ]
        for k in logging_keys:
            v = None
            if k == "traj_steps":
                v = step_idx
            elif k == "traj_len":
                v = traj_len
            elif k == "vill_rewards":
                v = rets[0]
            elif k == "were_rewards":
                v = rets[1]
            else:
                v = stats.get(k, 0)
            logging.append(v)

        # --- 6) BUILD SequenceSample ----------------------
        seqlens            = []
        packed_input_ids   = []
        prompt_mask        = []
        packed_logprobs    = []
        seq_no_eos_mask    = []
        final_rewards      = []

        for traj in trajectories:
            tokens_j, prompt_mask_j, logprobs_j, no_eos_flag_j, (r_vj, r_wj) = traj

            seqlens.append(len(tokens_j))
            packed_input_ids.extend(tokens_j)
            prompt_mask.extend(prompt_mask_j)
            packed_logprobs.extend(logprobs_j)
            seq_no_eos_mask.append(no_eos_flag_j)
            if self.role == "both":
                final_rewards.append(r_vj + r_wj)
            else:
                final_rewards.append((r_vj if self.role == "villager" else r_wj))

        data = {
            "packed_input_ids": torch.tensor(packed_input_ids,  dtype=torch.long,   device="cpu"),
            "prompt_mask":      torch.tensor(prompt_mask,       dtype=torch.bool,  device="cpu"),
            "packed_logprobs":  torch.tensor(packed_logprobs,   dtype=torch.float32,device="cpu"),
            "seq_no_eos_mask":  torch.tensor(seq_no_eos_mask,   dtype=torch.bool,   device="cpu"),
            "packed_prompts":   torch.tensor(prompt_ids,        dtype=torch.long,   device="cpu"),
            "version_start":    torch.tensor([version_start],   dtype=torch.int,    device="cpu"),
            "version_end":      torch.tensor([version_end],     dtype=torch.int,    device="cpu"),
            "rewards":          torch.tensor(final_rewards,     dtype=torch.float32,device="cpu"),
            "birth_time":       torch.tensor([birth_time_ms],   dtype=torch.long,   device="cpu"),
            "logging":          torch.tensor(logging,           dtype=torch.float32,device="cpu"),
        }
        seqlens = {
            "packed_input_ids": [seqlens],
            "prompt_mask":      [seqlens],
            "packed_logprobs":  [[s-1 for s in seqlens]],
            "seq_no_eos_mask":  [[1] * len(seqlens)],
            "packed_prompts":   [[prefix_len]],
            "version_start":    [[1]],
            "version_end":      [[1]],
            "rewards":          [[1] * len(final_rewards)],
            "birth_time":       [[1]],
            "logging":          [[len(logging_keys)]],
        }

        sample = SequenceSample(
            keys=list(data.keys()),
            ids=[orig_qid],
            dtypes={k: data[k].dtype for k in data},
            trailing_shapes={k: () for k in data},
            seqlens=seqlens,
            data=data,
        )
        y = SequenceSample(
            keys=["task_ids"],
            ids=[orig_qid],
            dtypes=dict(task_ids=torch.long),
            trailing_shapes=dict(task_ids=()),
            seqlens=dict(task_ids=[[1]]),
            data=dict(task_ids=torch.tensor([4], dtype=torch.long)),
        )
        sample.update_(y)
        return [sample]


register_agent("werewolf_agent", WerewolfAgent)
