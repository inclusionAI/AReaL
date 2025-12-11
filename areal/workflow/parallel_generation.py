"""
Parallel Generation Workflow for Multiverse Structured Generation.

This workflow implements structured parallel generation where the model generates:
1. A Goal with multiple Outlines
2. Multiple Paths (one for each outline) in parallel
3. A Conclusion that synthesizes the paths

The generation structure follows:
<Goal>
  <Outline>[Goal 1]</Outline>
  <Outline>[Goal 2]</Outline>
  ...
</Goal>
<Path>[Path 1]</Path>
<Path>[Path 2]</Path>
...
<Conclusion>[Conclusion]</Conclusion>

All paths see the same context and are generated independently without attention masking.
"""

import asyncio
import os
import re
import uuid
from collections.abc import Callable
from typing import Any

import aiofiles
import aiofiles.os
import colorama
import torch
from transformers import PreTrainedTokenizerFast

from areal.api.cli_args import GenerationHyperparameters
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest, ModelResponse
from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import RolloutWorkflow
from areal.utils import logging, stats_tracker
from areal.utils.data import concat_padded_tensors
from areal.utils.dynamic_import import import_from_string
from areal.utils.perf_tracer import (
    atrace_session_phase,
    session_context,
    trace_session,
)

logger = logging.getLogger("ParallelGeneration workflow")


def default_get_input_ids_fn(
    data: Any,
    tokenizer: PreTrainedTokenizerFast,
    enable_thinking: bool,
) -> list[int]:
    """Default function to convert data to input_ids."""
    input_ids = tokenizer.apply_chat_template(
        data,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return list(input_ids)


def default_data_extract_prompt_fn(data: dict[str, Any]) -> Any:
    """Default function to extract prompt from data."""
    return data["messages"]


def extract_outline_prefixes(goal_text: str) -> list[str]:
    """
    Extract outline prefixes from the goal text.
    
    The goal text contains numbered items in various formats:
    - Flat: "1:", "2:", "3:"
    - Hierarchical: "1.1:", "1.2:", "2.1.1:", etc.
    
    Args:
        goal_text: The generated goal text
    
    Returns:
        List of outline prefixes (e.g., ["1", "2", "3"] or ["1.1", "1.2", "2.1"])
    """
    # Pattern to match hierarchical numbered items like "1:", "1.1:", "2.1.1:", etc.
    pattern = r'^\s*([\d.]+)\s*.'
    
    # Find all matches across all lines
    matches = re.findall(pattern, goal_text, re.MULTILINE)
    
    # Clean up the prefixes (remove trailing dots if any)
    prefixes = [m.rstrip('.') for m in matches]
    logger.debug(f"Extracted outline prefixes: {prefixes}")
    return prefixes


class ParallelGenerationWorkflow(RolloutWorkflow):
    """
    Workflow for structured parallel generation with Goals, Paths, and Conclusions.
    
    This workflow:
    1. Generates until </Goal> tag
    2. Extracts outline prefixes from the goal
    3. Generates multiple paths in parallel (one per outline)
    4. Generates a conclusion that synthesizes all paths
    5. Computes rewards and packages trajectory for training
    """
    
    def __init__(
        self,
        reward_fn: Callable[..., Any] | str,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        enable_thinking: bool = False,
        max_goal_tokens: int = 2048,
        max_path_tokens: int = 4096,
        max_conclusion_tokens: int = 2048,
        rollout_stat_scope: str = "rollout",
        dump_dir: str | None = None,
        get_input_ids_fn: Callable[
            [Any, PreTrainedTokenizerFast, bool], list[int]
        ] = default_get_input_ids_fn,
        data_extract_prompt_fn: Callable[
            [dict[str, Any]], Any
        ] = default_data_extract_prompt_fn,
    ):
        """
        Initialize the parallel generation workflow.
        
        Args:
            reward_fn: Function or import string for computing rewards
            gconfig: Generation hyperparameters
            tokenizer: Tokenizer or path to tokenizer
            enable_thinking: Whether to enable thinking tokens
            max_goal_tokens: Max tokens for goal generation
            max_path_tokens: Max tokens per path
            max_conclusion_tokens: Max tokens for conclusion
            rollout_stat_scope: Scope for statistics tracking
            dump_dir: Optional directory to dump generation results
            get_input_ids_fn: Function to convert data to input_ids
            data_extract_prompt_fn: Function to extract prompt from data
        """
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        if isinstance(self.tokenizer, str):
            from areal.utils.hf_utils import load_hf_tokenizer
            tokenizer = load_hf_tokenizer(self.tokenizer)
            self.tokenizer = tokenizer
        
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(self.tokenizer)
        self.enable_thinking = enable_thinking
        self.max_goal_tokens = max_goal_tokens
        self.max_path_tokens = max_path_tokens
        self.max_conclusion_tokens = max_conclusion_tokens
        self.dump_dir = dump_dir
        self.rollout_stat_scope = rollout_stat_scope
        
        if not isinstance(reward_fn, str):
            self.async_reward_fn = AsyncRewardWrapper(reward_fn)
        
        self.get_input_ids_fn = get_input_ids_fn
        self.data_extract_prompt_fn = data_extract_prompt_fn
        
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)
    
    async def _generate_until_goal(
        self,
        engine: InferenceEngine,
        input_ids: list[int],
    ) -> ModelResponse:
        """Generate until </Goal> tag is encountered."""
        # Add stop words for goal generation
        goal_stop_words = ["</Goal>", "<|im_end|>", "<|endoftext|>", "</s>"]
        gconfig = self.gconfig.new(
            n_samples=1,
            max_new_tokens=self.max_goal_tokens,
            stop=goal_stop_words,
        )
        
        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            gconfig=gconfig,
            tokenizer=self.tokenizer,
        )
        
        resp = await engine.agenerate(req)
        return resp
    
    async def _generate_path(
        self,
        engine: InferenceEngine,
        context_ids: list[int],
        path_prefix: str,
    ) -> ModelResponse:
        """Generate a single path given context and prefix."""
        # Tokenize path prefix and add to context
        path_start = f"\n<Path>\n{path_prefix}"
        path_start_ids = self.tokenizer.encode(path_start, add_special_tokens=False)
        full_input_ids = context_ids + path_start_ids
        
        # Add stop words for path generation
        path_stop_words = ["</Path>", "<|im_end|>", "<|endoftext|>", "</s>"]
        gconfig = self.gconfig.new(
            n_samples=1,
            max_new_tokens=self.max_path_tokens,
            stop=path_stop_words,
        )
        
        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=full_input_ids,
            gconfig=gconfig,
            tokenizer=self.tokenizer,
        )
        
        resp = await engine.agenerate(req)
        return resp
    
    async def _generate_conclusion(
        self,
        engine: InferenceEngine,
        context_ids: list[int],
    ) -> ModelResponse:
        """Generate conclusion given full context with all paths."""
        # Add conclusion start tag
        conclusion_start = "\n<Conclusion>"
        conclusion_start_ids = self.tokenizer.encode(conclusion_start, add_special_tokens=False)
        full_input_ids = context_ids + conclusion_start_ids
        
        # Add stop words for conclusion generation
        conclusion_stop_words = ["</Conclusion>", "<|im_end|>", "<|endoftext|>", "</s>"]
        gconfig = self.gconfig.new(
            n_samples=1,
            max_new_tokens=self.max_conclusion_tokens,
            stop=conclusion_stop_words,
        )
        
        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=full_input_ids,
            gconfig=gconfig,
            tokenizer=self.tokenizer,
        )
        
        resp = await engine.agenerate(req)
        return resp
    
    @trace_session("reward")
    async def _compute_reward(
        self,
        prompt_str: str,
        completion_str: str,
        input_tokens: list[int],
        output_tokens: list[int],
        task_data: dict[str, Any],
    ) -> float:
        """Compute reward for the generated sequence."""
        reward = await self.async_reward_fn(
            prompt_str,
            completion_str,
            input_tokens,
            output_tokens,
            **task_data,
        )
        return reward
    
    @session_context()
    async def _process_parallel_stage(
        self,
        engine: InferenceEngine,
        initial_input_ids: list[int],
        goal_resp: ModelResponse,
        task_data: dict[str, Any],
    ) -> tuple[list[int], list[float], list[int], str]:
        """
        Process a parallel stage by generating paths and conclusion.
        
        Returns:
            Tuple of (all_tokens, all_logprobs, all_versions, full_completion_str)
        """
        logger.info("Entering parallel stage")
        
        # Build context up to goal
        goal_tokens = goal_resp.output_tokens
        goal_str = self.tokenizer.decode(goal_tokens)
        
        # Add closing </Goal> tag if not present
        if not goal_str.strip().endswith("</Goal>"):
            goal_close_ids = self.tokenizer.encode("</Goal>", add_special_tokens=False)
            goal_tokens = goal_tokens + goal_close_ids
            goal_str = self.tokenizer.decode(goal_tokens)
        
        context_ids = initial_input_ids + goal_tokens
        
        # Extract outline prefixes from goal
        outline_prefixes = extract_outline_prefixes(goal_str)
        if not outline_prefixes:
            logger.warning("No outlines found in goal. Using default prefix '1'")
            outline_prefixes = ["1"]
        
        num_paths = len(outline_prefixes)
        logger.info(f"Generating {num_paths} paths")
        
        # Generate all paths in parallel
        async with atrace_session_phase("generate_paths"):
            path_tasks = [
                self._generate_path(engine, context_ids, prefix)
                for prefix in outline_prefixes
            ]
            path_resps = await asyncio.gather(*path_tasks)
        
        # Build complete context with all paths
        all_path_tokens = []
        for prefix, path_resp in zip(outline_prefixes, path_resps):
            # Add path markers
            path_open = f"\n<Path>\n{prefix}"
            path_open_ids = self.tokenizer.encode(path_open, add_special_tokens=False)
            path_content_ids = path_resp.output_tokens
            
            # Add closing </Path> tag
            path_str = self.tokenizer.decode(path_content_ids)
            if not path_str.strip().endswith("</Path>"):
                path_close_ids = self.tokenizer.encode("</Path>", add_special_tokens=False)
                path_content_ids = path_content_ids + path_close_ids
            
            all_path_tokens.extend(path_open_ids + path_content_ids)
        
        context_with_paths_ids = context_ids + all_path_tokens
        
        # Generate conclusion
        async with atrace_session_phase("generate_conclusion"):
            conclusion_resp = await self._generate_conclusion(engine, context_with_paths_ids)
        
        # Add conclusion markers
        conclusion_open = "\n<Conclusion>"
        conclusion_open_ids = self.tokenizer.encode(conclusion_open, add_special_tokens=False)
        conclusion_content_ids = conclusion_resp.output_tokens
        
        # Add closing </Conclusion> tag
        conclusion_str = self.tokenizer.decode(conclusion_content_ids)
        if not conclusion_str.strip().endswith("</Conclusion>"):
            conclusion_close_ids = self.tokenizer.encode("</Conclusion>", add_special_tokens=False)
            conclusion_content_ids = conclusion_content_ids + conclusion_close_ids
        
        # Assemble full sequence
        all_tokens = goal_tokens + all_path_tokens + conclusion_open_ids + conclusion_content_ids
        
        # Build logprobs and versions arrays
        # Goal tokens
        all_logprobs = goal_resp.output_logprobs + [0.0] * (len(goal_tokens) - len(goal_resp.output_logprobs))
        all_versions = goal_resp.output_versions + [-1] * (len(goal_tokens) - len(goal_resp.output_versions))
        
        # Path tokens (we only have logprobs for actual generated content, not markers)
        path_token_idx = 0
        for prefix, path_resp in zip(outline_prefixes, path_resps):
            path_open = f"\n<Path>\n{prefix}"
            path_open_ids = self.tokenizer.encode(path_open, add_special_tokens=False)
            path_content_ids = path_resp.output_tokens
            path_str = self.tokenizer.decode(path_content_ids)
            if not path_str.strip().endswith("</Path>"):
                path_close_ids = self.tokenizer.encode("</Path>", add_special_tokens=False)
            else:
                path_close_ids = []
            
            # Markers have 0 logprob and -1 version
            all_logprobs.extend([0.0] * len(path_open_ids))
            all_versions.extend([-1] * len(path_open_ids))
            
            # Actual generated content
            all_logprobs.extend(path_resp.output_logprobs)
            all_versions.extend(path_resp.output_versions)
            
            # Closing tag
            all_logprobs.extend([0.0] * len(path_close_ids))
            all_versions.extend([-1] * len(path_close_ids))
        
        # Conclusion tokens
        all_logprobs.extend([0.0] * len(conclusion_open_ids))
        all_versions.extend([-1] * len(conclusion_open_ids))
        all_logprobs.extend(conclusion_resp.output_logprobs)
        all_versions.extend(conclusion_resp.output_versions)
        
        # Add closing tag logprobs/versions if needed
        if not conclusion_str.strip().endswith("</Conclusion>"):
            conclusion_close_ids = self.tokenizer.encode("</Conclusion>", add_special_tokens=False)
            all_logprobs.extend([0.0] * len(conclusion_close_ids))
            all_versions.extend([-1] * len(conclusion_close_ids))
        
        full_completion_str = self.tokenizer.decode(all_tokens)
        logger.info(f"Parallel stage complete: {num_paths} paths, {len(all_tokens)} tokens")
        
        return all_tokens, all_logprobs, all_versions, full_completion_str
    
    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        """
        Run a single episode of parallel generation.
        
        This generates:
        1. Goal with outlines
        2. Parallel paths (one per outline)
        3. Conclusion
        4. Computes reward and returns trajectory
        
        Returns trajectory dict with keys:
            - input_ids: Full sequence (prompt + generation)
            - loss_mask: Mask for training (1 for generated, 0 for prompt)
            - logprobs: Log probabilities for each token
            - versions: Model versions for each token
            - attention_mask: All ones
            - rewards: Scalar reward
        """
        # Load reward function dynamically if given as string
        if isinstance(self.reward_fn, str):
            self.reward_fn = import_from_string(self.reward_fn)
            self.async_reward_fn = AsyncRewardWrapper(self.reward_fn)
        
        # Get input_ids
        input_ids = self.get_input_ids_fn(
            self.data_extract_prompt_fn(data),
            self.tokenizer,
            self.enable_thinking,
        )
        
        version = engine.get_version()
        prompt_str = self.tokenizer.decode(input_ids)
        
        # Step 1: Generate until </Goal>
        async with atrace_session_phase("generate_goal"):
            goal_resp = await self._generate_until_goal(engine, input_ids)
        
        # Step 2: Process parallel stage (paths + conclusion)
        output_tokens, output_logprobs, output_versions, completion_str = await self._process_parallel_stage(
            engine, input_ids, goal_resp, data
        )
        
        # Step 3: Compute reward
        full_sequence = input_ids + output_tokens
        reward = await self._compute_reward(
            prompt_str,
            completion_str,
            input_ids,
            output_tokens,
            data,
        )
        
        stats_tracker.get(self.rollout_stat_scope).scalar(reward=reward)
        
        # Step 4: Build trajectory
        logprobs = [0.0] * len(input_ids) + output_logprobs
        loss_mask = [0] * len(input_ids) + [1] * len(output_tokens)
        versions = [-1] * len(input_ids) + output_versions
        
        result = {
            "input_ids": torch.tensor(full_sequence, dtype=torch.int32).unsqueeze(0),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.int32).unsqueeze(0),
            "logprobs": torch.tensor(logprobs, dtype=torch.float32).unsqueeze(0),
            "versions": torch.tensor(versions, dtype=torch.int32).unsqueeze(0),
            "attention_mask": torch.ones(len(full_sequence), dtype=torch.bool).unsqueeze(0),
            "rewards": torch.tensor(reward, dtype=torch.float32).unsqueeze(0),
        }
        
        # Step 5: Dump to file if requested
        if self.dump_dir is not None:
            dump_path = os.path.join(self.dump_dir, str(version))
            await aiofiles.os.makedirs(dump_path, exist_ok=True)
            
            # Get unique identifier for this prompt
            qid = None
            for key in ["query_id", "id", "qid"]:
                qid = data.get(key, None)
                if qid is not None:
                    break
            qid = qid or uuid.uuid4().hex
            
            # Dump rollout to file
            file_path = os.path.join(dump_path, f"{qid}.txt")
            async with aiofiles.open(file_path, "a") as f:
                info = "\n".join([
                    f"seqlen: {len(full_sequence)}, reward: {reward}",
                    f"prompt:\n{colorama.Fore.YELLOW + colorama.Style.DIM}{prompt_str}{colorama.Style.RESET_ALL}",
                    f"completion:\n{colorama.Fore.GREEN + colorama.Style.DIM}{completion_str}{colorama.Style.RESET_ALL}",
                    "=" * 80,
                ])
                await f.write(info + "\n")
        
        return result
