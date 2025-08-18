import asyncio
import os
import uuid

import aiofiles
import aiofiles.os
import colorama
import torch
import json
import re
from collections import defaultdict
from realhf.impl.environment import werewolf_env
from tensordict import TensorDict
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest
from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import RolloutWorkflow
from areal.utils.data import concat_padded_tensors
from realhf.base import logging
from realhf.impl.environment.werewolf_env import WerewolfEnv

logger = logging.getLogger("Werewolf workflow")

class WerewolfWorkflow(RolloutWorkflow):
    """
    Workflow for running the werewolf game.
    """
    def __init__(
        self, 
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        max_turns: int = 50,
        turn_discount: float = 1.0,
        dump_dir: str | None = None,
        env_kwargs: dict | None = None,
        role: str = "villager",
        opp_rollout: InferenceEngine | None = None,
        opp_tokenizer: PreTrainedTokenizerFast | None = None,
        teacher_rollout: InferenceEngine | None = None,
        teacher_tokenizer: PreTrainedTokenizerFast | None = None,
        questions: list[str] | None = None,
    ):
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.max_turns = max_turns
        self.turn_discount = turn_discount
        self.dump_dir = dump_dir
        self.env_kwargs = env_kwargs
        self.role = role
        self.opp_rollout = opp_rollout
        self.opp_tokenizer = opp_tokenizer
        self.use_teacher = True
        self.teacher_rollout = teacher_rollout
        self.teacher_tokenizer = teacher_tokenizer
        self.use_summary = True
        self.answer_questions = True
        self.questions = questions or [
            "Who do you suspect is the werewolf?",
        ]
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

    async def _run_one_episode(self, engine: InferenceEngine, data, rid):
        if self.env_kwargs:
            env = WerewolfEnv(**self.env_kwargs)
        else:
            env = WerewolfEnv()
        obs, guide, _ = await env.sreset()

        results = []
        prompt_strs = []
        completions_strs = []
        rewards = []
        seqlens = []
        vill_total = 0.0
        were_total = 0.0
        traj_len = [0, 0] # input, output
        summaries = defaultdict(list)
        qa_logs = []
        teacher_logs = []

        for turn in range(self.max_turns):
            # Store the current agent and get its summary
            current_agent = env.agent_player
            prev_summary = summaries[current_agent][-1] if summaries[current_agent] else "None yet."
            
            # Prepare all the questions for agents
            if self.use_summary:
                action_prompt = f"{obs} Your previous summary: {prev_summary} {guide}"
            else:
                action_prompt = f"{obs} {guide}"
            action_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": action_prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )

            use_opp_generation = (((env.agent_role != "werewolf" and self.role == "werewolf")
                or (env.agent_role == "werewolf" and self.role == "villager"))
                and self.opp_rollout)
            if use_opp_generation:
                req = ModelRequest(
                    rid=rid,
                    input_ids=action_ids,
                    gconfig=self.gconfig.new(n_samples=1),
                    tokenizer=self.opp_tokenizer or self.tokenizer,
                )
                action_task = self.opp_rollout.agenerate(req)

                # logger.warning(f"The OPP generation is: {self.tokenizer.decode(resp.output_tokens)}.")
            else:
                req = ModelRequest(
                    rid=rid,
                    input_ids=action_ids,
                    gconfig=self.gconfig.new(n_samples=1),
                    tokenizer=self.tokenizer,
                )
                action_task = engine.agenerate(req)

            question_tasks = []
            teacher_question_tasks = []
            if self.answer_questions:
                for qi, q in enumerate(self.questions):
                    q_prompt = f"{obs} Your previous summary: {prev_summary}. Based on this information, answer this question: {q}"
                    q_ids = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": q_prompt}],
                        tokenize=True,
                        add_generate_prompt=True,
                    )
                    q_req = ModelRequest(
                        rid=rid,
                        input_ids=q_ids,
                        gconfig=self.gconfig.new(n_samples=1),
                        tokenizer=self.tokenizer,
                    )
                    question_tasks.append(engine.agenerate(q_req))
                    if self.teacher_rollout:
                        tq_req = ModelRequest(
                            rid=rid,
                            input_ids=q_ids,
                            gconfig=self.gconfig.new(n_samples=1),
                            tokenizer=self.teacher_tokenizer or self.tokenizer,
                        )
                        teacher_question_tasks.append(
                            self.teacher_rollout.agenerate(tq_req)
                        )

                responses = await asyncio.gather(
                    action_task, *question_tasks, *teacher_question_tasks
                )
            else:
                responses = await asyncio.gather(action_task)

            resp = responses[0]
            q_resps = responses[1 : 1 + len(question_tasks)]
            tq_resps = responses[1 +len(question_tasks) : ]

            # Decode all the answers
            if use_opp_generation and self.opp_tokenizer:
                completion_str = self.opp_tokenizer.decode(resp.output_tokens)
            else:
                completion_str = self.tokenizer.decode(resp.output_tokens)
            agent_q_answers = [
                self.tokenizer.decode(r.output_tokens) for r in q_resps
            ]
            if self.teacher_tokenizer:
                teacher_q_answers = [
                    self.teacher_tokenizer.decode(r.output_tokens) for r in tq_resps
                ]
            else:
                teacher_q_answers = [
                    self.tokenizer.decode(r.output_tokens) for r in tq_resps
                ]

            seq = resp.input_tokens + resp.output_tokens
            logprobs = [0.0] * resp.input_len + resp.output_logprobs
            loss_mask = [0] * resp.input_len + [1] * resp.output_len
            versions = [-1] * resp.input_len + resp.output_versions

            prompt_str = self.tokenizer.decode(action_ids)
            
            # Get next env state
            current_agent = env.agent_player
            next_obs, next_guide, reward_list, done, _, _ = await env.step(
                (data.get("query_id", ""), [completion_str])
            )
            vill_total += float(reward_list[0])
            were_total += float(reward_list[1])

            if self.role == "both":
                reward = reward_list[0] + reward_list[1]
            elif self.role == "werewolf":
                reward = reward_list[1]
            else:
                reward = reward_list[0]

            # Record the results for this step
            res = dict(
                input_ids=torch.tensor(seq).unsqueeze(0),
                loss_mask=torch.tensor(loss_mask).unsqueeze(0),
                logprobs=torch.tensor(logprobs).unsqueeze(0),
                versions=torch.tensor(versions).unsqueeze(0),
                attention_mask=torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
                rewards=torch.tensor([float(reward)]),
            )
            results.append(TensorDict(res, batch_size=[1]))
            prompt_strs.append(prompt_str)
            completions_strs.append(completion_str)
            rewards.append(reward)
            seqlens.append(len(seq))
            traj_len[0] += resp.input_len
            traj_len[1] += resp.output_len

            # Ask for agent summarization
            if self.use_summary:
                m = re.findall(r"<answer>(.*?)</answer>", completion_str, re.DOTALL)
                action = m[-1].strip().lower() if m else ""
                summary_prompt = (
                    f"{obs} Your last actions: {action}. "
                    f"Your previous summary: {prev_summary}. Provide a brief summary of your thoughts and current game state,"
                    "to guide your future planning and next action."
                )
                summary_ids = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": summary_prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                summary_req = ModelRequest(
                    rid=f"{rid}-s-{turn}",
                    input_ids=summary_ids,
                    gconfig=self.gconfig.new(n_samples=1),
                    tokenizer=self.tokenizer,
                )
                summary_tasks = [engine.agenerate(summary_req)]
                if self.teacher_rollout:
                    t_summary_req = ModelRequest(
                        rid=f"{rid}-ts-{turn}",
                        input_ids=summary_ids,
                        gconfig=self.gconfig.new(n_samples=1),
                        tokenizer=self.teacher_tokenizer or self.tokenizer,
                    )
                    summary_tasks.append(self.teacher_rollout.agenerate(t_summary_req))
                summary_resps = await asyncio.gather(*summary_tasks)
                agent_summary = self.tokenizer.decode(summary_resps[0].output_tokens)
                summaries[current_agent].append(agent_summary)
                qa_logs.append(
                    {
                        "agent": current_agent,
                        "QAs": agent_q_answers,
                        "summary": agent_summary,
                    }
                )
                if self.teacher_rollout:
                    if self.teacher_tokenizer:
                        teacher_summary = self.teacher_tokenizer.decode(
                            summary_resps[1].output_tokens
                        )
                    else:
                        teacher_summary = self.tokenizer.decode(
                            summary_resps[1].output_tokens
                        )
                    teacher_logs.append(
                        {
                            "agent": current_agent,
                            "QAs": teacher_q_answers,
                            "summary": teacher_summary,
                        }
                    )
            else: # Do not use summary
                qa_logs.append(
                    {
                        "agent": current_agent,
                        "QAs": agent_q_answers,
                    }
                )
                teacher_logs.append(
                    {
                        "agent": current_agent,
                        "QAs": teacher_q_answers,
                    }
                )

            if done or (turn == self.max_turns - 1):
                logger.info(f"Trajectory ended with {turn + 1} turns, total reward: {sum(rewards)}.")
                break
            obs = next_obs
            guide = next_guide
        
        stats = {}
        if hasattr(env, "get_stats"):
            try:
                stats = env.get_stats()
            except Exception:
                logger.error("No stats are available for this trajectory.")

        logging_vals = [
            len(results),
            traj_len[0] + traj_len[1],
            traj_len[0],
            traj_len[1],
            vill_total,
            were_total,
            stats.get("vill_wins", 0),
            stats.get("were_wins", 0),
            stats.get("werewolf_kills", 0),
            stats.get("werewolf_correct_kills", 0),
            stats.get("villager_correct_votes", 0),
            stats.get("villager_wrong_votes", 0),
            stats.get("witch_heals", 0),
            stats.get("witch_correct_heals", 0),
            stats.get("witch_poisons", 0),
            stats.get("witch_correct_poisons", 0),
            stats.get("hunter_shots", 0),
            stats.get("hunter_correct_shots", 0),
        ]
        log_tensor = torch.tensor(logging_vals, dtype=torch.float32).unsqueeze(0)
        if results:
            results[0]["logging"] = log_tensor
            zero_log = torch.zeros_like(log_tensor)
            for i in range(1, len(results)):
                results[i]["logging"] = zero_log

        trajectory = []
        if hasattr(env, "get_trajectory"):
            try:
                trajectory = env.get_trajectory()
            except Exception:
                logger.error("Failed to get trajectory from env.")

        return results, prompt_strs, completions_strs, rewards, seqlens, trajectory, qa_logs, teacher_logs

    async def arun_episode(self, engine: InferenceEngine, data):
        rid = uuid.uuid4().hex
        tasks = [
            self._run_one_episode(engine, data, rid)
            for _ in range(self.gconfig.n_samples)
        ]
        episodes = await asyncio.gather(*tasks)

        results = []
        for res_list, *_ in episodes:
            results.extend(res_list)

        if self.dump_dir is not None and results:
            version = engine.get_version()
            dump_path = os.path.join(self.dump_dir, str(version))
            await aiofiles.os.makedirs(dump_path, exist_ok=True)
            qid = None
            for key in ["query_id", "id", "qid"]:
                qid = data.get(key, None)
                if qid is not None:
                    break
            qid = qid or uuid.uuid4().hex

            file_path = os.path.join(dump_path, f"{qid}.txt")
            async with aiofiles.open(file_path, "a") as f:
                for i, (_, p_list, c_list, r_list, sl_list, traj, qa_logs, t_logs) in enumerate(episodes):
                    for p, c, r, sl in zip(p_list, c_list, r_list, sl_list):
                        info = "\n".join(
                            [
                                f"idx: {i + 1}, seqlen: {sl}, reward is {r}.",
                                f"prompt is \n{colorama.Fore.YELLOW + colorama.Style.DIM}{p}{colorama.Style.RESET_ALL}",
                                f"sequence is: \n{colorama.Fore.YELLOW + colorama.Style.DIM}{c}{colorama.Style.RESET_ALL}",
                            ]
                        )
                        await f.write(info + "\n")
                    if traj:
                        traj_info = "\n".join(traj)
                        await f.write("Trajectory:\n\n" + traj_info + "\n")

                    # Log agent and teacher QAs and summaries
                    async with aiofiles.open(os.path.join(dump_path, f"{qid}_qalogs.json"), "a") as jsonf:
                        for item in qa_logs:
                            await jsonf.write(json.dumps(item) + "\n")
                    if self.teacher_rollout and t_logs:
                        async with aiofiles.open(os.path.join(dump_path, f"{qid}_tlogs.json"), "a") as jsonf:
                            for item in t_logs:
                                await jsonf.write(json.dumps(item) + "\n")

        return concat_padded_tensors(results)