import asyncio
import os
import uuid
import time

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


def _extract_three_questions(text: str) -> list[str]:
    """
    Try to parse three questions from the model output. Accepts formats like:
      Q1: ...
      Q2: ...
      Q3: ...
    Falls back to splitting into sentences ending with '?'. Returns up to 3.
    """
    qs = []
    # Pattern 1: Q1...Q2...Q3
    if ("Q1" in text) and ("Q2" in text) and ("Q3" in text):
        q1 = text[text.find("Q1") + 2 : text.find("Q2")]
        q2 = text[text.find("Q2") + 2 : text.find("Q3")]
        q3 = text[text.find("Q3") + 2 :]
        qs = [q1, q2, q3]
    # Pattern 2: Q1:/Q2:/Q3:
    if len(qs) < 3:
        for m in re.findall(
            r"Q\s*([123])\s*[:：]\s*(.+?)(?=(?:\nQ\s*[123]\s*[:：])|\Z)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            _, q = m
            q = q.strip()
            if q.endswith(("</s>", "</answer>", "</assistant>")):
                q = re.sub(r"</s>|</answer>|</assistant>", "", q).strip()
            qs.append(q)
    # If not found, try to split by question marks.
    if len(qs) < 3:
        cand = re.findall(r"([^?？]+[?？])", text, flags=re.DOTALL)
        cand = [c.strip() for c in cand if c.strip()]
        for c in cand:
            if len(qs) >= 3:
                break
            qs.append(c)
    return qs[:3]


class WerewolfWorkflow(RolloutWorkflow):
    """
    Workflow for running the werewolf game.
    Agent now self-generates 3 questions each turn, answers them,
    and uses the answers to guide action-making. Teacher also answers
    these questions to provide data.
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
        questions: list[str] | None = None,  # kept for backward-compat but unused now
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
        # Deprecated path: we no longer use predefined questions
        self.answer_questions = True
        self.questions = []

        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

    async def _run_one_episode(self, engine: InferenceEngine, data, rid):
        if self.env_kwargs:
            env = WerewolfEnv(**self.env_kwargs)
        else:
            env = WerewolfEnv()
        obs, guide, _ = await env.sreset()

        results = []  # Final return tensor
        prompt_strs = []  # All prompts
        completions_strs = []  # All completions
        rewards = []  # All rewards
        seqlens = []
        vill_total = 0.0 # Villager side total reward
        were_total = 0.0 # Werewolf side total reward
        traj_len = [0, 0]  # input, output
        summaries = defaultdict(list)
        qa_logs = []
        teacher_logs = []

        # ---- timing accumulators (seconds) ----
        t_qgen_total = 0.0
        t_qgen_decode_total = 0.0
        t_agent_answer_total = 0.0
        t_teacher_answer_total = 0.0
        t_action_build_total = 0.0
        t_action_gen_total = 0.0
        t_action_decode_total = 0.0
        t_env_step_total = 0.0
        t_summary_agent_total = 0.0
        t_summary_teacher_total = 0.0
        t_pack_tensors_total = 0.0
        t_tokenize_total = 0.0

        turns_done = 0

        for turn in range(self.max_turns):
            turns_done += 1
            # Store the current agent and get its summary
            current_agent = env.agent_player
            current_role = env.agent_role
            prev_summary = summaries[current_agent][-1] if summaries[current_agent] else "None yet."

            # ========== 1) Agent self-generates 3 questions ==========
            qgen_prompt = (
                f"{obs}\n"
                f"Your previous summary: {prev_summary}\n\n"
                "Formulate exactly three short, decision-focused questions with ground truth answers you should ask yourself "
                "to choose the best next action in this werewolf game turn.\n"
                "Be specific to the current roles, suspicions, allies, and public information.\n"
                "Output strictly in the following format:\n"
                "Q1: ...\\n\nQ2: ...\\n\nQ3: ...\\n"
            )
            t0 = time.perf_counter()
            qgen_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": qgen_prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
            t_tokenize_total += time.perf_counter() - t0

            qgen_req = ModelRequest(
                rid=f"{rid}-qgen-{turn}",
                input_ids=qgen_ids,
                gconfig=self.gconfig.new(n_samples=1, max_new_tokens=1024),
                tokenizer=self.tokenizer,
            )
            t0 = time.perf_counter()
            qgen_resp = await engine.agenerate(qgen_req)
            t_qgen_total += time.perf_counter() - t0

            t0 = time.perf_counter()
            qgen_text = self.tokenizer.decode(qgen_resp.output_tokens)
            t_qgen_decode_total += time.perf_counter() - t0

            self_questions = _extract_three_questions(qgen_text)
            if not self_questions:
                ask_target_role = "a werewolf" if current_role != "werewolf" else "the biggest living threat"
                # Safety fallback
                self_questions = [
                    f"Who is most likely {ask_target_role} this turn and why?",
                    "What action should I take now to maximize team success?",
                    "What information should I reveal or conceal to influence votes?",
                ]

            # ========== 2) Agent answers the 3 questions ==========
            agent_answer_tasks = []
            agent_answer_ids = []
            for qi, q in enumerate(self_questions):
                aprompt = (
                    f"{obs}\n"
                    f"Your previous summary: {prev_summary}\n"
                    f"Question: {q}\n"
                    "Answer concisely and concretely for this turn only."
                )
                t0 = time.perf_counter()
                a_ids = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": aprompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                t_tokenize_total += time.perf_counter() - t0

                a_req = ModelRequest(
                    rid=f"{rid}-ans-{turn}-{qi}",
                    input_ids=a_ids,
                    gconfig=self.gconfig.new(n_samples=1, max_new_tokens=512),
                    tokenizer=self.tokenizer,
                )
                agent_answer_ids.append(a_ids)
                agent_answer_tasks.append(engine.agenerate(a_req))

            # Teacher also answers for data (not used to guide action)
            teacher_answer_tasks = []
            if self.teacher_rollout:
                for qi, q in enumerate(self_questions):
                    taprompt = (
                        f"{obs}\n"
                        f"Previous agent summary: {prev_summary}\n"
                        f"Question: {q}\n"
                        "Answer concisely and concretely for this turn only."
                    )
                    t0 = time.perf_counter()
                    ta_ids = (self.teacher_tokenizer or self.tokenizer).apply_chat_template(
                        [{"role": "user", "content": taprompt}],
                        tokenize=True,
                        add_generation_prompt=True,
                    )
                    t_tokenize_total += time.perf_counter() - t0

                    ta_req = ModelRequest(
                        rid=f"{rid}-tans-{turn}-{qi}",
                        input_ids=ta_ids,
                        gconfig=self.gconfig.new(n_samples=1, max_new_tokens=512),
                        tokenizer=self.teacher_tokenizer or self.tokenizer,
                    )
                    teacher_answer_tasks.append(self.teacher_rollout.agenerate(ta_req))

            if teacher_answer_tasks:
                t0 = time.perf_counter()
                resps = await asyncio.gather(*agent_answer_tasks, *teacher_answer_tasks)
                t_all = time.perf_counter() - t0
                # Split timing crudely by counts
                n_a, n_t = len(agent_answer_tasks), len(teacher_answer_tasks)
                if n_a + n_t > 0:
                    t_agent_answer_total += t_all * (n_a / (n_a + n_t))
                    t_teacher_answer_total += t_all * (n_t / (n_a + n_t))
                agent_ans_resps = resps[: len(agent_answer_tasks)]
                teacher_ans_resps = resps[len(agent_answer_tasks) :]
            else:
                t0 = time.perf_counter()
                agent_ans_resps = await asyncio.gather(*agent_answer_tasks)
                t_agent_answer_total += time.perf_counter() - t0
                teacher_ans_resps = []

            t0 = time.perf_counter()
            agent_answers = [self.tokenizer.decode(r.output_tokens) for r in agent_ans_resps]
            if self.teacher_rollout:
                if self.teacher_tokenizer:
                    teacher_answers = [self.teacher_tokenizer.decode(r.output_tokens) for r in teacher_ans_resps]
                else:
                    teacher_answers = [self.tokenizer.decode(r.output_tokens) for r in teacher_ans_resps]
            else:
                teacher_answers = []
            # Count decode time into "agent answer" bucket (dominant part after gen)
            t_agent_answer_total += time.perf_counter() - t0

            # ========== 3) Use agent's Q&A to guide action generation ==========
            qa_block = "\n".join([
                f"{i+1}) {self_questions[i]}\nAnswer: {agent_answers[i].strip()}"
                for i in range(len(agent_answers))
            ])
            t0 = time.perf_counter()
            if self.use_summary:
                action_prompt = (
                    f"{obs}\nYour previous summary: {prev_summary}\n\n"
                    f"Your self-questions and answers:\n{qa_block}\n\n{guide}"
                )
            else:
                action_prompt = f"{obs}\n\nYour self-questions and answers:\n{qa_block}\n\n{guide}"
            action_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": action_prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
            t_action_build_total += time.perf_counter() - t0
            t_tokenize_total += 0.0  # (already included in build section)

            use_opp_generation = (
                ((env.agent_role != "werewolf" and self.role == "werewolf") or (env.agent_role == "werewolf" and self.role == "villager"))
                and self.opp_rollout
            )
            if use_opp_generation:
                req = ModelRequest(
                    rid=rid,
                    input_ids=action_ids,
                    gconfig=self.gconfig.new(n_samples=1, max_new_tokens=2048),
                    tokenizer=self.opp_tokenizer or self.tokenizer,
                )
                t0 = time.perf_counter()
                resp = await self.opp_rollout.agenerate(req)
                t_action_gen_total += time.perf_counter() - t0
                t0 = time.perf_counter()
                completion_str = self.opp_tokenizer.decode(resp.output_tokens) if self.opp_tokenizer else self.tokenizer.decode(resp.output_tokens)
                t_action_decode_total += time.perf_counter() - t0
            else:
                req = ModelRequest(
                    rid=rid,
                    input_ids=action_ids,
                    gconfig=self.gconfig.new(n_samples=1),
                    tokenizer=self.tokenizer,
                )
                t0 = time.perf_counter()
                resp = await engine.agenerate(req)
                t_action_gen_total += time.perf_counter() - t0
                t0 = time.perf_counter()
                completion_str = self.tokenizer.decode(resp.output_tokens)
                t_action_decode_total += time.perf_counter() - t0

            seq = resp.input_tokens + resp.output_tokens
            logprobs = [0.0] * resp.input_len + resp.output_logprobs
            loss_mask = [0] * resp.input_len + [1] * resp.output_len
            versions = [-1] * resp.input_len + resp.output_versions

            prompt_str = self.tokenizer.decode(action_ids)

            # Get next env state
            t0 = time.perf_counter()
            next_obs, next_guide, reward_list, done, _, _ = await env.step(
                (data.get("query_id", ""), [completion_str])
            )
            t_env_step_total += time.perf_counter() - t0

            vill_total += float(reward_list[0])
            were_total += float(reward_list[1])

            if self.role == "both":
                reward = reward_list[0] + reward_list[1]
            elif self.role == "werewolf":
                reward = reward_list[1]
            else:
                reward = reward_list[0]

            # Record the results for this step
            t0 = time.perf_counter()
            res = dict(
                input_ids=torch.tensor(seq).unsqueeze(0),
                loss_mask=torch.tensor(loss_mask).unsqueeze(0),
                logprobs=torch.tensor(logprobs).unsqueeze(0),
                versions=torch.tensor(versions).unsqueeze(0),
                attention_mask=torch.ones(len(seq), dtype=torch.bool).unsqueeze(0),
                rewards=torch.tensor([float(reward)]),
            )
            results.append(TensorDict(res, batch_size=[1]))
            t_pack_tensors_total += time.perf_counter() - t0

            prompt_strs.append(prompt_str)
            completions_strs.append(completion_str)
            rewards.append(reward)
            seqlens.append(len(seq))
            traj_len[0] += resp.input_len
            traj_len[1] += resp.output_len

            # ========== 4) Ask for agent summarization, then log all Q&As ==========
            if self.use_summary:
                m = re.findall(r"<answer>(.*?)</answer>", completion_str, re.DOTALL)
                action_txt = m[-1].strip().lower() if m else ""
                summary_prompt = (
                    f"{obs} Your last action: {action_txt}. "
                    f"Your previous summary: {prev_summary}. Provide a brief summary of your thoughts and current game state, "
                    "to guide your future planning and next action. Be concise and brief for this answer."
                )
                summary_ids = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": summary_prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
                summary_req = ModelRequest(
                    rid=f"{rid}-s-{turn}",
                    input_ids=summary_ids,
                    gconfig=self.gconfig.new(n_samples=1, max_new_tokens=512),
                    tokenizer=self.tokenizer,
                )
                summary_tasks = [engine.agenerate(summary_req)]
                # if self.teacher_rollout:
                #     t_summary_req = ModelRequest(
                #         rid=f"{rid}-ts-{turn}",
                #         input_ids=summary_ids,
                #         gconfig=self.gconfig.new(n_samples=1, max_new_tokens=512),
                #         tokenizer=self.teacher_tokenizer or self.tokenizer,
                #     )
                #     summary_tasks.append(self.teacher_rollout.agenerate(t_summary_req))
                t0 = time.perf_counter()
                summary_resps = await asyncio.gather(*summary_tasks)
                t_sum = time.perf_counter() - t0
                if len(summary_tasks) == 2:
                    # Rough split
                    t_summary_agent_total += t_sum * 0.5
                    t_summary_teacher_total += t_sum * 0.5
                else:
                    t_summary_agent_total += t_sum

                agent_summary = self.tokenizer.decode(summary_resps[0].output_tokens)
                summaries[current_agent].append(agent_summary)

                qa_logs.append(
                    {
                        "agent": current_agent,
                        "role": current_role,
                        "QAs": [{"question": self_questions[i], "answer": agent_answers[i]} for i in range(len(self_questions))],
                        "generated_questions": self_questions,
                        "summary_prompt": summary_prompt,
                        "summary": agent_summary,
                    }
                )
                # if self.teacher_rollout:
                #     if self.teacher_tokenizer:
                #         teacher_summary = self.teacher_tokenizer.decode(summary_resps[1].output_tokens)
                #     else:
                #         teacher_summary = self.tokenizer.decode(summary_resps[1].output_tokens)
                #     teacher_logs.append(
                #         {
                #             "agent": current_agent,
                #             "role": current_role,
                #             "QAs": [{"question": self_questions[i], "answer": teacher_answers[i] if i < len(teacher_answers) else ""} for i in range(len(self_questions))],
                #             "generated_questions": self_questions,
                #             "summary_prompt": summary_prompt,
                #             "summary": teacher_summary,
                #         }
                #     )
            else:  # Do not use summary
                qa_logs.append(
                    {
                        "agent": current_agent,
                        "role": current_role,
                        "QAs": [{"question": self_questions[i], "answer": agent_answers[i]} for i in range(len(self_questions))],
                        "generated_questions": self_questions,
                    }
                )
                if self.teacher_rollout:
                    teacher_logs.append(
                        {
                            "agent": current_agent,
                            "role": current_role,
                            "QAs": [{"question": self_questions[i], "answer": teacher_answers[i] if i < len(teacher_answers) else ""} for i in range(len(self_questions))],
                            "generated_questions": self_questions,
                        }
                    )

            if done or (turn == self.max_turns - 1):
                logger.info(f"Trajectory ended with {turn + 1} turns, total reward: {sum(rewards)}.")
                break
            obs = next_obs
            guide = next_guide

        # Stats logging
        stats = {}
        if hasattr(env, "get_stats"):
            try:
                stats = env.get_stats()
            except Exception:
                logger.error("No stats are available for this trajectory.")

        # ---- finalize timing aggregates ----
        avg_div = max(1, turns_done)
        timing_vals = [
            # t_qgen_total,                 # 18: total time generating questions
            # t_qgen_decode_total,          # 19: total time decoding questions
            # t_agent_answer_total,         # 20: total time answering (agent) incl. decodes
            # t_teacher_answer_total,       # 21: total time answering (teacher)
            # t_action_build_total,         # 22: total time building action prompt (+tokenize build)
            # t_action_gen_total,           # 23: total time generating action
            # t_action_decode_total,        # 24: total time decoding action
            # t_env_step_total,             # 25: total env.step time
            # t_summary_agent_total,        # 26: total time generating agent summary
            # t_summary_teacher_total,      # 27: total time generating teacher summary
            # t_pack_tensors_total,         # 28: total time packing tensors
            # t_tokenize_total,             # 29: total time spent in tokenizer.apply_chat_template
            # Averages per turn (useful to monitor)
            t_qgen_total / avg_div,               # 30: avg qgen
            t_agent_answer_total / avg_div,       # 31: avg agent answers
            t_teacher_answer_total / avg_div,     # 32: avg teacher answers
            t_action_gen_total / avg_div,         # 33: avg action gen
            # t_env_step_total / avg_div,           # 34: avg env step
            (t_summary_agent_total / avg_div),    # 35: avg agent summary
        ]

        logging_vals = [
            len(results),                           # 0
            traj_len[0] + traj_len[1],              # 1
            traj_len[0],                            # 2
            traj_len[1],                            # 3
            vill_total,                             # 4
            were_total,                             # 5
            stats.get("vill_wins", 0),              # 6
            stats.get("were_wins", 0),              # 7
            stats.get("werewolf_kills", 0),         # 8
            stats.get("werewolf_correct_kills", 0), # 9
            stats.get("villager_correct_votes", 0), # 10
            stats.get("villager_wrong_votes", 0),   # 11
            stats.get("witch_heals", 0),            # 12
            stats.get("witch_correct_heals", 0),    # 13
            stats.get("witch_poisons", 0),          # 14
            stats.get("witch_correct_poisons", 0),  # 15
            stats.get("hunter_shots", 0),           # 16
            stats.get("hunter_correct_shots", 0),   # 17
        ] + timing_vals                              # 18+ timing slots as documented above

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
                        await jsonf.write("[")
                        for i in range(len(qa_logs)):
                            await jsonf.write(json.dumps(qa_logs[i]) + (",\n" if i < len(qa_logs) - 1 else "\n"))
                        await jsonf.write("]")
                    if self.teacher_rollout and t_logs:
                        async with aiofiles.open(os.path.join(dump_path, f"{qid}_tlogs.json"), "a") as jsonf:
                            await jsonf.write("[")
                            for i in range(len(t_logs)):
                                await jsonf.write(json.dumps(t_logs[i]) + (",\n" if i < len(t_logs) - 1 else "\n"))
                            await jsonf.write("]")

        return concat_padded_tensors(results)
