import asyncio
import os
import uuid
import re
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import time 
import aiofiles
import aiofiles.os
import colorama
import torch
from tensordict import TensorDict
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest
from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import RolloutWorkflow
from areal.utils import logging, stats_tracker
from areal.utils.data import concat_padded_tensors
from transformers import AutoTokenizer
import math
logger = logging.getLogger("Multi-Turn workflow")


# User-override hook: post-process rewards per sample.
# Inputs are lists of equal length (one entry per generated sample):
# - thread_token_counts: List[int]
# - main_token_counts: List[int]
# - longest_paths: List[int]
# - rewards: List[float] (original rewards)
# Output should be a List[float] with the same length.
def sigmoid(x: float) -> float:
    """Numerically stable sigmoid that operates on Python floats.
    Ensures return type is float to avoid mixing Torch tensors in rewards.
    """
    x = float(x)
    # clamp to avoid overflow in exp for very large magnitude x
    
    z = math.exp(-x)
    return 1.0 / (1.0 + z)
def postprocess_rewards(
    thread_token_counts: List[int],
    main_token_counts: List[int],
    longest_paths: List[int],
    rewards: List[float],
) -> List[float]:
    type = 1
    ratio = 0.22
    parallel_ratio = [
        t / (m + t) if (m + t) > 0 else 0
        for t, m in zip(thread_token_counts, main_token_counts)
    ]
    average_parallel_ratio = float(sum(parallel_ratio) / len(parallel_ratio)) if parallel_ratio else 0.0
    std_parallel_ratio = (
        float(torch.std(torch.tensor(parallel_ratio, dtype=torch.float32)).item())
        if parallel_ratio else 0.0
    )
    # parallel_over_main = [
    #     t / m if m > 0 else 0
    #     for t, m in zip(thread_token_counts, main_token_counts)
    # ]
    # Default is identity (no change). Edit this function to customize the reward.
    if thread_token_counts:
        _tt = torch.tensor(thread_token_counts, dtype=torch.float32)
        average_thread_tokens = float(torch.mean(_tt).item())
        thread_tokens_std = float(torch.std(_tt).item())
    else:
        average_thread_tokens = 0.0
        thread_tokens_std = 0.0

    if main_token_counts:
        _mt = torch.tensor(main_token_counts, dtype=torch.float32)
        average_main_tokens = float(torch.mean(_mt).item())
        main_tokens_std = float(torch.std(_mt).item())
    else:
        average_main_tokens = 0.0
        main_tokens_std = 0.0

    if longest_paths:
        _lp = torch.tensor(longest_paths, dtype=torch.float32)
        average_longest_path = float(torch.mean(_lp).item())
        longest_path_std = float(torch.std(_lp).item())
    else:
        average_longest_path = 0.0
        longest_path_std = 0.0
    rewards_coeffcient_1 = [
        sigmoid((thread_tokens - average_thread_tokens) / (thread_tokens_std + 1e-5)) if thread_tokens_std > 0 else 1.0
        for thread_tokens in thread_token_counts
    ]
    rewards_coeffcient_2 = [
        sigmoid((parallel - average_parallel_ratio) / (std_parallel_ratio + 1e-5)) if std_parallel_ratio > 0 else 1.0
        for parallel in parallel_ratio
    ]
    rewards_coeffcient_3 = [
        sigmoid((longest_path - average_longest_path) / (longest_path_std + 1e-5)) if longest_path_std > 0 else 1.0
        for longest_path in longest_paths
    ]
    for i in range(len(rewards)):
        if type == 0:
            rewards[i] = rewards[i] * (1 + ratio * rewards_coeffcient_1[i])
        elif type == 1:
            rewards[i] = rewards[i] * (1 + ratio * rewards_coeffcient_2[i])
        elif type == 2:
            rewards[i] = rewards[i] * (1 - ratio * rewards_coeffcient_3[i])
        else:
            rewards[i] = rewards[i]
    return rewards


# Define system prompts for parallel thinking
MAIN_SYSTEM_PROMPT = """You are a helpful assistant that solve math problems. 
The input is a math problem and you need to solve it step by step. If you think you need to launch a thread, you can do so by using the '<launch_threads>' tag. Each thread will have its own task and objective, put the whole thread_launching process in the following format:
```
<launch_threads>
<thread id='0'>
<task>
[Task Name]
</task>
<objective> 
[Objective of the thread]
</objective>
</thread>
<thread id='1'>
<task>
[Task Name]
</task> 
<objective> 
[Objective of the thread]
</objective>
</thread>
</launch_threads>
```
You should complete the whole reasoning process of the original problem, rather than just a partial step in main mode. If you are in the main mode, start the reasoning process with the special tag '<think>'
"""

THREAD_SYSTEM_PROMPT = """You are a helpful assistant that solve math problems.  

The input is a math problem and reasoning process before this thread is launched. Also the task and objective of this thread will be provided in the end of the input. You should complete the task and objective of this thread, start your processing with <thread_processing id = 'i'> where i is the index of the thread. End your processing with '</thread_processing>'. After this, put the result of this step between '<thread_result id='i'>' and '</thread_result>'. DO NOT output the special tag '<think>'  DO NOT output the special tag '<think>', what you need to do is to finish the reasoning of <thread_processing id='i'> and output its result, you only need to solve this partial step not the full problem
 Stop reasoning when you reach the end of the thread processing and then output the result in the format of '<thread_result id='i'>result</thread_result>'.
 NEVER solve the whole problem, you MUST STOP after the objective of this step is reached. Also, think for a while before you output the result, put the reasoning process in <thread_processing id='i'> ... </thread_processing> tag, where 'i' is the id of this thread. Put the result of **THIS STEP** (not the whole problem) in the <thread_result id='i'> ... </thread_result> tag"""


def format_chat_template(system_prompt: str, user_content: str, assistant_start: str = "") -> str:
    """Format content using ChatML template"""
    formatted = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
{assistant_start}"""
    return formatted
    tokenizer = AutoTokenizer.from_pretrained("/storage/openpsi/users/zzy/model/parallel_1_5b/nvidia-parallel-thinking_1_5B_lr5/checkpoint-267")
    print ("[DEBUG]: Successfully load tokenizer")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_start}
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize = False)
    formatted = ''.join(formatted.rsplit('<|im_end|>', 1))
    return formatted


def extract_thread_launch_info(text: str) -> List[Dict[str, str]]:
    """Extract thread launch information from text - only unprocessed threads"""
    # Find all launch_threads blocks
    launch_pattern = r'<launch_threads>(.*?)</launch_threads>'
    launch_matches = re.findall(launch_pattern, text, re.DOTALL)
    
    # If no complete launch_threads blocks found, try to find incomplete ones
    if not launch_matches:
        incomplete_pattern = r'<launch_threads>(.*?)$'
        incomplete_matches = re.findall(incomplete_pattern, text, re.DOTALL)
        if incomplete_matches:
            launch_matches = incomplete_matches
        else:
            return []
    
    # Get all existing thread results to avoid reprocessing
    result_pattern = r'<thread_result id=\'(\d+)\'>'
    existing_results = re.findall(result_pattern, text)
    existing_thread_ids = set(existing_results)
    
    thread_info = []
    
    # Process all launch_threads blocks to find unprocessed threads
    for launch_content in launch_matches:
        # Updated regex pattern to handle multiline task and objective content
        thread_pattern = r"<thread id='(\d+)'>\s*<task>\s*(.*?)\s*</task>\s*<objective>\s*(.*?)\s*</objective>\s*</thread>"
        threads = re.findall(thread_pattern, launch_content, re.DOTALL)
        
        for thread_id, task, objective in threads:
            thread_id = thread_id.strip()
            task = task.strip()
            objective = objective.strip()
            
            # Only add threads that don't have results yet
            if thread_id not in existing_thread_ids:
                thread_info.append({
                    'id': thread_id,
                    'task': task,
                    'objective': objective
                })
    
    return thread_info


def extract_thread_result(thread_response: str, thread_id: str) -> str:
    """Extract thread result from <thread_result> tag to end of output"""
    # Look for thread_result opening tag
    result_start_pattern = rf'<thread_result id=\'{thread_id}\'>'
    result_start_match = re.search(result_start_pattern, thread_response)
    
    if result_start_match:
        start_pos = result_start_match.start()
        return thread_response[start_pos:]
    else:
        # If no thread_result tag found, wrap the whole response
        return f"<thread_result id='{thread_id}'>{thread_response}</thread_result>"
        return f"<thread_result id='{thread_id}'>\n\n</thread_result>"

class ParallelThinkingWorkflow(RolloutWorkflow):
    # Class-level executor to prevent premature shutdown
    _shared_async_reward_fn = None
    _shared_reward_fn = None
    
    def __init__(
        self,
        reward_fn,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast,
        max_turns: int,
        turn_discount: float,
        rollout_stat_scope: str = "rollout",
        dump_dir: str = "/storage/openpsi/users/langzhang.zzy/workspace/AReaL_new/AReaL/result",
        max_context_length: int = 24576,  # Maximum context length before stopping
    ):
        self.reward_fn = reward_fn
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.max_turns = max_turns
        self.turn_discount = turn_discount
        self.rollout_stat_scope = rollout_stat_scope
        
        # Use shared async reward function to prevent executor shutdown issues
        if (ParallelThinkingWorkflow._shared_async_reward_fn is None or 
            ParallelThinkingWorkflow._shared_reward_fn != reward_fn):
            ParallelThinkingWorkflow._shared_reward_fn = reward_fn
            ParallelThinkingWorkflow._shared_async_reward_fn = AsyncRewardWrapper(reward_fn)
        
        self.async_reward_fn = ParallelThinkingWorkflow._shared_async_reward_fn
        self.dump_dir = dump_dir
        self.max_context_length = max_context_length
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)

        # Create tokens that should be amended if the answer is incorrect.
        # This method eliminates the encode-decode inconsistency issue and cancels system prompts.
        messages = [{"role": "assistant", "content": "some random message."}]
        s1 = self.tokenizer.apply_chat_template(messages, tokenize=True)
        messages += [
            {
                "role": "user",
                "content": "Your answer is either wrong or not parsable to the reward function. You may misunderstand the original question. "
                "Please carefully read the original question, check the preivous errors, and try to answer it again.",
            }
        ]
        s2 = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        self.multi_turn_prompt_ids = s2[len(s1) :]

    @classmethod
    def cleanup_shared_resources(cls):
        """Cleanup shared resources when no longer needed"""
        if cls._shared_async_reward_fn is not None:
            try:
                del cls._shared_async_reward_fn
                cls._shared_async_reward_fn = None
                cls._shared_reward_fn = None
            except Exception:
                pass  # Ignore cleanup errors

    def _check_context_length(self, input_ids_or_text) -> bool:

        """Check if context length exceeds maximum allowed length"""
        length = len(input_ids_or_text)
      # (f"INPUT LENGTH: {length}")
        if isinstance(input_ids_or_text, str):
            # If it's text, encode it to get token count
            token_count = len(self.tokenizer.encode(input_ids_or_text))
        else:
            # If it's already token IDs, get the length
            token_count = len(input_ids_or_text)
        
        if token_count > self.max_context_length:
            print(f"⚠️ Context length {token_count} exceeds maximum {self.max_context_length}, stopping generation")
            return False
        else:
            # print(f"✅ Context length {token_count} is within limit {self.max_context_length}")
            return True

    async def _run_one_episode(self, engine: InferenceEngine, data, rid):
        # Enforces `n_samples=1`
        # Placeholders for the results
        seq, logprobs, loss_mask, versions = [], [], [], []
        messages = data["messages"]
        
        # Convert the prompt into input_ids
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        
        # Extract the problem text from messages
        problem_text = ""
        for msg in messages:
            if msg["role"] == "user":
                problem_text = msg["content"]
                problem_text = problem_text.replace("Please put your final answer within \\boxed{}.", "").strip()
                break
        
        # Run multi-turn rollout until correct
        t = reward = 0
        discount = 1
        
        # Initialize context for parallel thinking
        reasoning_context = ""
        first_token = "<think>Okay"
        
        # Initialize variables that will be used in return statement
        prompt_str = self.tokenizer.decode(input_ids)
        full_response_str = ""
        
        # Store all generations for dumping and reward computation
        all_generations = []
        all_generated_content = []  # Collect all generated content for reward function
        all_thread_seq_data = []  # Collect sequence data for all threads
        
        while not ("</think>" in full_response_str and "boxed{" in full_response_str )  and t < self.max_turns:
            #print("[DEBUG]", t, self.max_turns, flush=True)
            # print(f"\n🔄 MAIN TURN {t + 1}: Starting main thread processing...")
            if not reasoning_context.endswith("</launch_threads>\n"):
            # Step 1: Generate main reasoning until thread launch or completion
                if t == 0:
                    # First turn: generate from beginning with first token
                    formatted_prompt = format_chat_template(MAIN_SYSTEM_PROMPT, problem_text, first_token)
                    main_input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").squeeze()
                    prompt_str = formatted_prompt
                else:
                    # Subsequent turns: continue from existing context
                    formatted_prompt = format_chat_template(MAIN_SYSTEM_PROMPT, problem_text, reasoning_context)
                    main_input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").squeeze()
                
                # Check context length before generation
                if not self._check_context_length(main_input_ids):
                    print(f"🛑 Stopping generation due to context length limit")
                    reward = 0  # Set reward to 0 and break
                    break
                
                # Generate main reasoning
                req = ModelRequest(
                    rid=f"{rid}_main_{t}",
                    input_ids=main_input_ids.tolist(),
                    gconfig=self.gconfig.new(n_samples=1, max_new_tokens=5000, stop=["</launch_threads>", "<|im_end|>", "<|endoftext|>", "</s>"]),
                    tokenizer=self.tokenizer,
                )
                main_resp = await engine.agenerate(req)
                
                # Update sequences for main generation
                main_response = self.tokenizer.decode(main_resp.output_tokens)
                
                # Store main generation for dumping
                all_generations.append({
                    'type': 'main',
                    'turn': t,
                    'prompt': formatted_prompt,
                    'response': main_response,
                    'rid': f"{rid}_main_{t}",
                    'output_token_count': getattr(main_resp, 'output_len', len(getattr(main_resp, 'output_tokens', [])))
                })
                full_response_str += all_generations[-1]['response'] + "\n"
                # print(f"\n📝 MAIN TURN {t + 1} OUTPUT:")
                # print(full_response_str)
                # print("Sleep 5 s")
                # time.sleep(0)
                # Add main response to all generated content for reward computation
                all_generated_content.append(main_response)
                
                # Add main response to reasoning context
                if t == 0:
                    reasoning_context = first_token + main_response
                else:
                    reasoning_context = reasoning_context + main_response
                
                # Check if we need to add the closing tag for incomplete launch_threads
                if not main_response.endswith("</launch_threads>\n") and "<launch_threads>" in main_response:
                    main_response += "</launch_threads>\n"
                    reasoning_context += "</launch_threads>\n"
                
                # Update tracking for main generation
                if t == 0:
                    # For the first turn, add the full input and output tokens
                    seq += main_resp.input_tokens + main_resp.output_tokens
                    logprobs += [0.0] * len(main_resp.input_tokens) + main_resp.output_logprobs
                    loss_mask += [0] * len(main_resp.input_tokens) + [1] * main_resp.output_len
                    versions += [-1] * len(main_resp.input_tokens) + main_resp.output_versions
                else:
                    # For subsequent turns, only add the new output
                    new_tokens = main_resp.output_tokens
                    if new_tokens:
                        seq += new_tokens
                        logprobs += main_resp.output_logprobs
                        loss_mask += [1] * len(new_tokens)
                        versions += main_resp.output_versions
            
            # Step 2: Process threads in parallel if any are launched
            thread_turn_count = 1
            while True:
                # print(f"\n🔄 MAIN TURN {t + 1}, THREAD TURN {thread_turn_count}: Looking for threads to process...")
                
                # Extract thread information from reasoning context
                thread_infos = extract_thread_launch_info(reasoning_context)
                
                if not thread_infos:
                    # print(f"❌ No threads found in main turn {t + 1}, thread turn {thread_turn_count}")
                    break  # Break inner loop, continue main thread
                
                # print(f"\n🔍 Found {len(thread_infos)} threads to process:")
                # for thread_info in thread_infos:
                #     # print(f"  • Thread {thread_info['id']}: {thread_info['task']}")
                
                # Process threads in parallel using asyncio
                pattern = r'<launch_threads>.*?</launch_threads>'
                matches = list(re.finditer(pattern, reasoning_context, re.DOTALL))
                
                if matches:
                    # Get the last match
                    last_match = matches[-1]
                    # Remove the last launch_threads block
                    reasoning_context_temp = reasoning_context[:last_match.start()] + reasoning_context[last_match.end():]


                thread_results, thread_seq_data_list = await self._process_threads_parallel(engine, problem_text, reasoning_context_temp, thread_infos, rid, t, thread_turn_count, all_generations, all_generated_content)

                # Store thread sequence data
                all_thread_seq_data.extend(thread_seq_data_list)
                
                # Combine thread results
                # print(f"\n📊 Combining thread results...")
                combined_results = ""
                for thread_id in sorted(thread_results.keys()):
                    combined_results += thread_results[thread_id] + "\n"
                
                # Update reasoning context with thread results
                reasoning_context = reasoning_context + "<step_resolution>\n" + combined_results + "</step_resolution>\n"
                
                # Update sequences for thread results
                thread_tokens = self.tokenizer.encode("<step_resolution>\n" + combined_results + "</step_resolution>\n", add_special_tokens=False)
                seq += thread_tokens
                logprobs += [0.0] * len(thread_tokens)  # Thread results don't contribute to loss
                loss_mask += [0] * len(thread_tokens)
                versions += [-1] * len(thread_tokens)
                
                thread_turn_count += 1
                if thread_turn_count > 5:  # Safety limit for thread turns
                    # print(f"⚠️ Reached maximum thread turn limit (5)")
                    break
            
            # Step 3: Continue main generation after thread processing
            if "<launch_threads>" in main_response:
                # print(f"\n📝 MAIN TURN {t + 1}: Continuing main generation after thread processing...")
                
                # Continue main generation from the updated context
                formatted_prompt = format_chat_template(MAIN_SYSTEM_PROMPT, problem_text, reasoning_context)
                continue_input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").squeeze()
                
                # Check context length before continuation generation
                if not self._check_context_length(continue_input_ids):
                    print(f"🛑 Stopping continuation generation due to context length limit")
                    break  # Break main loop, end generation
                
                req = ModelRequest(
                    rid=f"{rid}_continue_{t}",
                    input_ids=continue_input_ids.tolist(),
                    gconfig=self.gconfig.new(n_samples=1, max_new_tokens=5000, stop=["</launch_threads>", "<|im_end|>", "<|endoftext|>", "</s>"]),
                    tokenizer=self.tokenizer,
                )
                continue_resp = await engine.agenerate(req)
                
                # Update sequences for continued generation
                continue_response = self.tokenizer.decode(continue_resp.output_tokens)
                reasoning_context += continue_response
                
                # Store continuation generation for dumping
                all_generations.append({
                    'type': 'main_continue',
                    'turn': t,
                    'prompt': formatted_prompt,
                    'response': continue_response,
                    'rid': f"{rid}_continue_{t}",
                    'output_token_count': getattr(continue_resp, 'output_len', len(getattr(continue_resp, 'output_tokens', [])))
                })
                full_response_str += all_generations[-1]['response'] + "\n"
                # print("[CONTINUED CONTENT]")
                # print(f"\n📝 MAIN TURN {t + 1} CONTINUATION OUTPUT:")
                # print(full_response_str)
                # print("Sleep 5s")
                # time.sleep(0)
                # Add continuation response to all generated content for reward computation
                all_generated_content.append(continue_response)
                
                # Add continued response to sequences
                if continue_resp.output_tokens:
                    seq += continue_resp.output_tokens
                    logprobs += continue_resp.output_logprobs
                    loss_mask += [1] * len(continue_resp.output_tokens)
                    versions += continue_resp.output_versions
                
                # Check if the new continuation contains more thread launches
                if "<launch_threads>" not in continue_response:
                    # print(f"✅ No more thread launches found after main turn {t + 1}, ending generation")
                    break  # Break main loop, end turn-based generation
            
            # Step 4: Compute reward for the complete response (main + wrapped threads)
            # Extract just the output tokens for reward computation
            all_output_tokens = seq[len(input_ids):]
            
            # Create comprehensive response string from all generated content
            
            
            try:
                
                reward = await self.async_reward_fn(
                    prompt_str,
                    full_response_str,
                    input_ids,
                    all_output_tokens,
                    **data,
                )
                # print("[REWARD]: Context sent to reward function:")
                # print("[VALUE]: ", reward)
                # print(full_response_str)
                # print("Sleep 5 s")
                # print("===============================================================================================================")
                # time.sleep(0)
            except RuntimeError as e:
                if "cannot schedule new futures after shutdown" in str(e):
                    logger.warning(f"Executor shutdown detected, recreating shared async reward function")
                    # Recreate the shared async reward function if executor was shut down
                    ParallelThinkingWorkflow._shared_async_reward_fn = AsyncRewardWrapper(self.reward_fn)
                    self.async_reward_fn = ParallelThinkingWorkflow._shared_async_reward_fn
                    reward = await self.async_reward_fn(
                        prompt_str,
                        full_response_str,
                        input_ids,
                        all_output_tokens,
                        **data,
                    )
                else:
                    raise e
            except Exception as e:
                logger.error(f"Error computing reward: {e}")
                raise e
            
            # Increase counter
            t += 1
            
            # If answer is incorrect and we haven't reached max turns, add correction prompt
            # if reward == 0 and t < self.max_turns:
            #     # Add the multi-turn correction prompt to the context
            #     correction_tokens = self.multi_turn_prompt_ids
            #     reasoning_context += self.tokenizer.decode(correction_tokens)
                
            #     seq += correction_tokens
            #     logprobs += [0.0] * len(correction_tokens)
            #     loss_mask += [0] * len(correction_tokens)
            #     versions += [-1] * len(correction_tokens)
                
            #     discount *= self.turn_discount
        all_output_tokens = seq[len(input_ids):]
        correctness_reward = await self.async_reward_fn(
            prompt_str,
            full_response_str,
            input_ids,
            all_output_tokens,
            **data,
        )
        
        # Log reward.
        num_launch = full_response_str.count("<launch_threads>")
        #reward = correctness_reward * (0.8 + 0.2*(num_launch > 0))
        reward = correctness_reward
        # Compute token usage by part (main vs threads) using output_token_count recorded in all_generations
        main_token_count = sum(
            gen.get('output_token_count', 0)
            for gen in all_generations
            if gen.get('type') in ('main', 'main_continue')
        )
        thread_token_count = sum(
            gen.get('output_token_count', 0)
            for gen in all_generations
            if gen.get('type') == 'thread'
        )
        # Longest path metric: for each launch round (turn, thread_turn), add the longest thread output tokens, then add main tokens
        longest_thread_sum = 0
        threads_by_round: Dict[tuple, list] = {}
        for gen in all_generations:
            if gen.get('type') == 'thread':
                key = (gen.get('turn'), gen.get('thread_turn'))
                cnt = gen.get('output_token_count', 0)
                threads_by_round.setdefault(key, []).append(cnt)
        for counts in threads_by_round.values():
            if counts:
                longest_thread_sum += max(counts)
        longest_path_tokens = int(main_token_count + longest_thread_sum)
        # logger.info(f"Token usage — main: {main_token_count}, thread: {thread_token_count}, longest_path: {longest_path_tokens}")
        parallel_ratio = thread_token_count / (main_token_count + thread_token_count)
        parallel_over_main = thread_token_count / (main_token_count )  
        stats_tracker.get(self.rollout_stat_scope).scalar(
            reward=reward,
            num_turns=t,
            num_launch=num_launch,
            correctness_reward=correctness_reward,
            main_tokens=main_token_count,
            longest_path_tokens=longest_path_tokens,
            parallel_over_main=parallel_over_main,
            parallel_ratio=parallel_ratio,
            thread_tokens=thread_token_count,
        )

        # Create main conversation result
        main_res = dict(
            input_ids=torch.tensor(seq),
            logprobs=torch.tensor(logprobs),
            loss_mask=torch.tensor(loss_mask),
            versions=torch.tensor(versions),
            rewards=torch.tensor(float(reward )),
            attention_mask=torch.ones(len(seq), dtype=torch.bool),
        )
        main_res = {k: v.unsqueeze(0) for k, v in main_res.items()}
        
        # Create thread conversation results
        thread_results = []
        for thread_seq_data in all_thread_seq_data:
            thread_res = dict(
                input_ids=torch.tensor(thread_seq_data['input_ids']),
                logprobs=torch.tensor(thread_seq_data['logprobs']),
                loss_mask=torch.tensor(thread_seq_data['loss_mask']),
                versions=torch.tensor(thread_seq_data['versions']),
                rewards=torch.tensor(float(reward)),
                attention_mask=torch.ones(len(thread_seq_data['input_ids']), dtype=torch.bool),
            )
            thread_res = {k: v.unsqueeze(0) for k, v in thread_res.items()}
            thread_results.append(TensorDict(thread_res, batch_size=[1]))
        
        # Combine main result with thread results
        all_results = [TensorDict(main_res, batch_size=[1])] + thread_results
        # print(f"\n🎯 Final reward: {reward} after {t} turns")
        # print (f"-----------------------------------")
        # print(f"FINAL OUTPUT: {full_response_str}")
        # print (f"-----------------------------------")
        # print(f"Sleep 5 s")
        # time.sleep(0)
        if "NO_VALID_RESULT" in full_response_str:
            reward = 0
        return (
            all_results, # _ in dump
            prompt_str, # p in dump
            full_response_str, # c in dump
            reward, # r in dump
            len(seq), # sl in dump
            all_generations,  # all_gens in dump
            thread_token_count,  # thread tokens for this sample
            main_token_count,    # main tokens for this sample
            longest_path_tokens, # longest path tokens for this sample
        )
    
    async def _process_threads_parallel(self, engine: InferenceEngine, problem_text: str, context: str, thread_infos: List[Dict[str, str]], rid: str, main_turn: int, thread_turn: int, all_generations: List, all_generated_content: List) -> tuple:
        """Process multiple threads in parallel using asyncio"""
        results = {}
        thread_seq_data_list = []
        
        # print(f"\n🔄 STARTING PARALLEL PROCESSING OF {len(thread_infos)} THREADS")
        
        # Create tasks for parallel thread processing
        tasks = []
        for thread_info in thread_infos:
            task = self._process_single_thread(engine, problem_text, context, thread_info, rid, main_turn, thread_turn, all_generations, all_generated_content)
            tasks.append(task)
        
        # Wait for all threads to complete
        thread_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results and sequence data
        for i, result in enumerate(thread_results):
            thread_info = thread_infos[i]
            thread_id = thread_info['id']
            
            if isinstance(result, Exception):
                # print(f"❌ Thread {thread_id} generated an exception: {result}")
                results[thread_id] = f"<thread_result id='{thread_id}'>Error: {str(result)}</thread_result>"
            else:
                thread_id_result, result_content, thread_seq_data = result
                results[thread_id_result] = result_content
                if thread_seq_data is not None:
                    thread_seq_data_list.append(thread_seq_data)
                # print(f"✅ Collected result from thread {thread_id_result}")
        
        # print(f"\n🎉 ALL THREADS COMPLETED!")
        return results, thread_seq_data_list
    
    async def _process_single_thread(self, engine: InferenceEngine, problem_text: str, context: str, thread_info: Dict[str, str], rid: str, main_turn: int, thread_turn: int, all_generations: List, all_generated_content: List) -> tuple:
        """Process a single thread and return its result and sequence data"""
        try:
            thread_id = thread_info['id']
            task = thread_info['task']
            objective = thread_info['objective']
            
            # print(f"\n🧵 PROCESSING THREAD {thread_id}: {task}")
            
            # Create the thread input
            user_content = f"""Problem: {problem_text}
{context}
<thread id='{thread_id}'>
<task>
{task}
</task>
<objective>
{objective}
</objective>
</thread>"""
            
            assistant_start = f"<thread_processing id='{thread_id}'>Okay, so"
            formatted_prompt = format_chat_template(THREAD_SYSTEM_PROMPT, user_content, assistant_start)
            
            thread_input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").squeeze()
            
            # Check context length before thread generation
            if not self._check_context_length(thread_input_ids):
                print(f"🛑 Stopping thread {thread_id} generation due to context length limit")
                error_result = f"<thread_result id='{thread_id}'>Error: Context length exceeded</thread_result>"
                return thread_id, error_result, None
            
            # Generate thread response
            req = ModelRequest(
                rid=f"{rid}_thread_{thread_id}_{main_turn}_{thread_turn}",
                input_ids=thread_input_ids.tolist(),
                gconfig=self.gconfig.new(n_samples=1, max_new_tokens=5000, stop=["</thread_result>", "<|im_end|>", "<|endoftext|>", "</s>"]),
                tokenizer=self.tokenizer,
            )
            resp = await engine.agenerate(req)
            
            thread_response = self.tokenizer.decode(resp.output_tokens)
            
            # Store thread generation for dumping
            all_generations.append({
                'type': 'thread',
                'turn': main_turn,
                'thread_turn': thread_turn,
                'thread_id': thread_id,
                'task': task,
                'objective': objective,
                'prompt': formatted_prompt,
                'response': thread_response,
                'rid': f"{rid}_thread_{thread_id}_{main_turn}_{thread_turn}",
                'output_token_count': getattr(resp, 'output_len', len(getattr(resp, 'output_tokens', [])))
            })
            
            # Add thread response to all generated content for reward computation
            all_generated_content.append(thread_response)
            
            # Create sequence data for this thread conversation
            thread_seq_data = {
                'input_ids': resp.input_tokens + resp.output_tokens,
                'logprobs': [0.0] * len(resp.input_tokens) + resp.output_logprobs,
                'loss_mask': [0] * len(resp.input_tokens) + [1] * resp.output_len,
                'versions': [-1] * len(resp.input_tokens) + resp.output_versions
            }
            
            # print(f"\n📝 THREAD {thread_id} RAW OUTPUT:")
            # print(f"{'-'*50}")
            # print(thread_response[:500] + "..." if len(thread_response) > 500 else thread_response)
            # print(f"{'-'*50}")
            
            thread_result = extract_thread_result(thread_response, thread_id)
            
            # print(f"\n✅ THREAD {thread_id} EXTRACTED RESULT:")
            # print(f"{'-'*50}")
            # print(thread_result[:300] + "..." if len(thread_result) > 300 else thread_result)
            # print(f"{'-'*50}")
            
            # print(f"✅ Thread {thread_id} completed successfully!")
            return thread_id, thread_result, thread_seq_data
            
        except Exception as e:
            # print(f"❌ Error processing thread {thread_info['id']}: {e}")
            error_result = f"<thread_result id='{thread_info['id']}'>Error: {str(e)}</thread_result>"
            return thread_info['id'], error_result, None

    async def arun_episode(self, engine: InferenceEngine, data):
        # print("[DEBUG] enter arun_episode", flush=True)
        rid = uuid.uuid4().hex
        tasks = [
            self._run_one_episode(engine, data, rid)
            for _ in range(self.gconfig.n_samples)
        ]
        results = await asyncio.gather(*tasks)

        if self.dump_dir is not None:
            version = engine.get_version()
            dump_path = os.path.join(self.dump_dir, str(version))
            await aiofiles.os.makedirs(dump_path, exist_ok=True)
            # Get the unique identifier for this prompt
            qid = None
            for key in ["query_id", "id", "qid"]:
                qid = data.get(key, None)
                if qid is not None:
                    break
            qid = qid or uuid.uuid4().hex

            # Dump rollout to file
            file_path = os.path.join(dump_path, f"{qid}.txt")
            async with aiofiles.open(file_path, "a") as f:
                n_samples = self.gconfig.n_samples
                for i, (result_list, p, c, r, sl, all_gens, *_) in enumerate(results):
                    info = "\n".join(
                        [
                            f"idx: {i + 1} / {n_samples}, seqlen: {sl}, reward is {r}.",
                            f"prompt is \n{colorama.Fore.YELLOW + colorama.Style.DIM}{p}{colorama.Style.RESET_ALL}",
                            f"sequence is: \n{colorama.Fore.YELLOW + colorama.Style.DIM}{c}{colorama.Style.RESET_ALL}",
                        ]
                    )
                    await f.write(info + "\n")

                    # Write detailed generation information
                    await f.write("\n" + "=" * 80 + "\n")
                    await f.write("DETAILED GENERATION BREAKDOWN:\n")
                    await f.write("=" * 80 + "\n")

                    for gen_idx, gen in enumerate(all_gens):
                        await f.write(f"\n--- Generation {gen_idx + 1}: {gen['type'].upper()} ---\n")
                        await f.write(f"Turn: {gen['turn']}")
                        if 'thread_turn' in gen:
                            await f.write(f", Thread Turn: {gen['thread_turn']}")
                        if 'thread_id' in gen:
                            await f.write(f", Thread ID: {gen['thread_id']}")
                        await f.write(f"\nRID: {gen['rid']}\n")

                        if gen['type'] == 'thread':
                            await f.write(f"Task: {gen['task']}\n")
                            await f.write(f"Objective: {gen['objective']}\n")

                        await f.write(f"\nPROMPT:\n{colorama.Fore.CYAN + colorama.Style.DIM}")
                        await f.write(gen['prompt'])
                        await f.write(f"{colorama.Style.RESET_ALL}\n")

                        await f.write(f"\nRESPONSE:\n{colorama.Fore.GREEN + colorama.Style.DIM}")
                        await f.write(gen['response'])
                        await f.write(f"{colorama.Style.RESET_ALL}\n")
                        await f.write("-" * 60 + "\n")

                    await f.write("\n" + "=" * 80 + "\n\n")

        # Extract results, gather metrics, postprocess rewards, and aggregate
        all_tensor_dicts: List[TensorDict] = []
        thread_counts: List[int] = []
        main_counts: List[int] = []
        longest_paths: List[int] = []
        orig_rewards: List[float] = []
        per_sample_indices: List[tuple] = []  # (start_idx, end_idx) for each sample in all_tensor_dicts

        offset = 0
        for result in results:
            result_list, _, _, reward_value, _, _, thread_tok, main_tok, longest_tok = result
            count_in_sample = len(result_list)
            all_tensor_dicts.extend(result_list)
            # One set of metrics per sample
            thread_counts.append(int(thread_tok))
            main_counts.append(int(main_tok))
            longest_paths.append(int(longest_tok))
            orig_rewards.append(float(reward_value))
            per_sample_indices.append((offset, offset + count_in_sample))
            offset += count_in_sample

        # Allow user to override rewards based on metrics
        new_rewards = postprocess_rewards(thread_counts, main_counts, longest_paths, orig_rewards)

        # Apply new rewards back into tensor dicts per sample
        for (start, end), new_r in zip(per_sample_indices, new_rewards):
            for i in range(start, end):
                td = all_tensor_dicts[i]
                # td fields are batched [1, ...]; update rewards tensor
                if 'rewards' in td:
                    td['rewards'] = td['rewards'].clone()
                    td['rewards'][0] = td['rewards'][0].new_tensor(float(new_r))

        # Clip each tensor dict to 24576 tokens if it exceeds this limit
        clipped_tensor_dicts = []
        max_length = 24576
        for tensor_dict in all_tensor_dicts:
            # Get the sequence length from input_ids
            seq_length = tensor_dict['input_ids'].shape[1]  # Assuming shape is [batch_size, seq_length]
            if seq_length > max_length:
                # Clip all sequence-related tensors to max_length
                clipped_dict = {}
                for key, tensor in tensor_dict.items():
                    if key in ['input_ids', 'logprobs', 'loss_mask', 'versions', 'attention_mask']:
                        clipped_dict[key] = tensor[:, :max_length]
                    else:
                        clipped_dict[key] = tensor
                clipped_tensor_dict = TensorDict(clipped_dict, batch_size=tensor_dict.batch_size)
                clipped_tensor_dicts.append(clipped_tensor_dict)
            else:
                clipped_tensor_dicts.append(tensor_dict)

        return concat_padded_tensors(clipped_tensor_dicts)