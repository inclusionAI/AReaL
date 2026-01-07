"""
Parallel Generation Workflow for Multiverse Structured Generation with Multi-Turn Support.

This workflow implements structured parallel generation where the model generates:
1. A Goal with multiple Outlines
2. Multiple Paths (one for each outline) in parallel
3. A <Conclusion> marker (conclusion content is NOT generated, allowing continuation)
4. (Optional) Continue with additional goal-paths-conclusion cycles

The generation structure follows (for multi-turn with max_turns=2):
<Goal>
  <Outline>[Goal 1]</Outline>
  <Outline>[Goal 2]</Outline>
  ...
</Goal>
<Path>[Path 1]</Path>
<Path>[Path 2]</Path>
...
<Conclusion>
[Model continues from here in next turn...]
<Goal>
  <Outline>[Goal 1]</Outline>
  ...
</Goal>
<Path>[Path 1]</Path>
...
<Conclusion>
...

Key Features:
- String-based context: Rebuilds the full prompt each time (system + user + assistant content)
- All paths see the same context (prompt + goal), NOT other paths
- No conclusion generation: Just adds <Conclusion>\\n marker for continuation
- Logprobs tracking: Maintains logprobs for RL training despite string-based approach
- Each subsequent turn uses the accumulated assistant content as context
"""
MAX_POS_ENCODING = 30000
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

#logger = logging.getLogger("ParallelGeneration workflow")


def default_format_prompt_fn(user_content: str, enable_thinking: bool) -> str:
    """
    Default function to format the user's input into a string before tokenization.
    
    This function receives the user's content and returns a formatted string.
    You can customize this function to create your own prompt format instead of
    using the chat template.
    
    Args:
        user_content: The user's input content (e.g., the math problem)
        enable_thinking: Whether thinking mode is enabled
    
    Returns:
        Formatted string ready for tokenization
    
    Example:
        Input: "Solve: 2+2=?"
        Output: "<|im_start|>user\\nSolve: 2+2=?<|im_end|>\\n<|im_start|>assistant\\n"
    """
    # Default implementation: simple format
    # TODO: Customize this function to match your desired format
    formatted = f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
    return formatted


def format_prompt_with_context(user_content: str, assistant_content: str = "") -> str:
    """
    Format the full prompt with optional assistant content (for string-based generation).
    
    This function rebuilds the complete prompt each time, including any previously
    generated assistant content. This approach is similar to generation_new.py's
    format_chat_template function.
    
    Args:
        user_content: The user's input content (e.g., the math problem)
        assistant_content: Previously generated assistant content to include
    
    Returns:
        Formatted string ready for tokenization
    
    Example:
        Input: user_content="Solve: 2+2=?", assistant_content="<Goal>...</Goal>"
        Output: "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\nSolve: 2+2=?<|im_end|>\n<|im_start|>assistant\n<Goal>...</Goal>"
    """
    formatted = f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n{assistant_content}"
    return formatted


def default_get_input_ids_fn(
    data: Any,
    tokenizer: PreTrainedTokenizerFast,
    enable_thinking: bool,
    format_prompt_fn: Callable[[str, bool], str] | None = None,
) -> list[int]:
    """
    Default function to convert data to input_ids.
    
    Args:
        data: The user's input (string or messages list)
        tokenizer: Tokenizer to use
        enable_thinking: Whether thinking mode is enabled
        format_prompt_fn: Optional custom formatting function
    
    Returns:
        List of token IDs
    """
    # If format_prompt_fn is provided, use it
    if format_prompt_fn is not None:
        # print("[DEBUG]: Using custom format_prompt_fn for input IDs conversion.")
        # Extract user content from data
        if isinstance(data, str):
            user_content = data
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            # Extract from messages format
            user_content = data[0].get("content", "")
        else:
            user_content = str(data)
        
        # Format the prompt using the custom function
        formatted_text = format_prompt_fn(user_content, enable_thinking)
        
        # Tokenize the formatted text
        input_ids = tokenizer.encode(formatted_text, add_special_tokens=False)
        return list(input_ids)
    else:
        # Fallback: use chat template if available
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
    
    The goal text should contain outlines within <Outline>...</Outline> tags.
    Each outline starts with a prefix like "1.", "2.", etc.
    
    Args:
        goal_text: The generated goal text
    
    Returns:
        List of outline prefixes (e.g., ["1", "2", "3"])
    """
    # First, extract the last <Goal>...</Goal> block
    goal_pattern = r'<Goal>(.*?)</Goal>'
    goal_matches = re.findall(goal_pattern, goal_text, re.DOTALL)
    
    if not goal_matches:
        # No <Goal> block found, return empty list
        return []
    
    # Use the last <Goal> block
    last_goal = goal_matches[-1]
    
    # Extract all <Outline>...</Outline> content from the last goal
    outline_pattern = r'<Outline>\s*([\d.]+)'
    outline_matches = re.findall(outline_pattern, last_goal, re.DOTALL)
    
    # Clean up the prefixes (remove trailing dots if any)
    prefixes = [m.rstrip('.') for m in outline_matches]
    # logger.debug(f"Extracted outline prefixes: {prefixes}")
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
        max_goal_tokens: int = 30000,
        max_path_tokens: int = 10000,
        max_conclusion_tokens: int = 10000,
        max_turns: int = 20,
        rollout_stat_scope: str = "rollout",
        dump_dir: str | None = None,
        format_prompt_fn: Callable[[str, bool], str] | None = None,
        get_input_ids_fn: Callable[
            [Any, PreTrainedTokenizerFast, bool, Callable[[str, bool], str] | None], list[int]
        ] = default_get_input_ids_fn,
        data_extract_prompt_fn: Callable[
            [dict[str, Any]], Any
        ] = default_data_extract_prompt_fn,
        reward_shaping_mode: str | None = None,
        reward_shaping_alpha: float = 0.1,
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
            max_turns: Maximum number of goal-paths-conclusion cycles (default: 1)
            rollout_stat_scope: Scope for statistics tracking
            dump_dir: Optional directory to dump generation results
            format_prompt_fn: Custom function to format user input into string before tokenization.
                             If None, will use chat template. Signature: (user_content: str, enable_thinking: bool) -> str
            get_input_ids_fn: Function to convert data to input_ids
            data_extract_prompt_fn: Function to extract prompt from data
            reward_shaping_mode: Mode for reward shaping. Options:
                - None: No reward shaping (default)
                - "parallel_ratio": reward = original_reward * [1 + alpha * (parallel_ratio - avg) / std]
                - "latency": reward = original_reward * [1 - alpha * (latency - avg) / std]
            reward_shaping_alpha: Alpha hyperparameter for reward shaping (default: 0.1)
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
        self.max_turns = max_turns
        self.dump_dir = dump_dir
        self.rollout_stat_scope = rollout_stat_scope
        
        if not isinstance(reward_fn, str):
            self.async_reward_fn = AsyncRewardWrapper(reward_fn, timeout_seconds = 60)
        
        self.format_prompt_fn = format_prompt_fn
        self.get_input_ids_fn = get_input_ids_fn
        self.data_extract_prompt_fn = data_extract_prompt_fn
        
        # Reward shaping parameters
        self.reward_shaping_mode = reward_shaping_mode
        self.reward_shaping_alpha = reward_shaping_alpha
        
        if self.dump_dir is not None and not os.path.exists(self.dump_dir):
            os.makedirs(self.dump_dir, exist_ok=True)
    
    async def _log_model_call(
        self,
        stage: str,
        input_ids: list[int],
        response: ModelResponse | None,
        version: int,
        sample_idx: int,
        qid: str,
    ):
        """Log model input and output for debugging."""
        if self.dump_dir is None:
            return
        
        debug_dir = os.path.join(self.dump_dir, str(version), "debug")
        await aiofiles.os.makedirs(debug_dir, exist_ok=True)
        
        debug_file = os.path.join(debug_dir, f"{qid}_sample{sample_idx}.log")
        
        # Decode input with chat template kept
        input_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
        
        # Prepare log content
        log_lines = [
            f"\n{'=' * 80}",
            f"STAGE: {stage}",
            f"Timestamp: {asyncio.get_event_loop().time()}",
            f"{'=' * 80}",
            f"\n--- INPUT (length: {len(input_ids)} tokens) ---",
            input_text,
            f"\n--- INPUT TOKEN IDS ---",
            str(input_ids),
        ]
        
        if response is not None:
            output_text = self.tokenizer.decode(response.output_tokens, skip_special_tokens=False)
            log_lines.extend([
                f"\n--- OUTPUT (length: {len(response.output_tokens)} tokens) ---",
                output_text,
                f"\n--- OUTPUT TOKEN IDS ---",
                str(response.output_tokens),
                f"\n--- LOGPROBS (first 10) ---",
                str(response.output_logprobs[:10] if len(response.output_logprobs) > 10 else response.output_logprobs),
            ])
        else:
            log_lines.append("\n--- OUTPUT: None (exceeded MAX_POS_ENCODING) ---")
        
        log_lines.append(f"\n{'=' * 80}\n")
        
        async with aiofiles.open(debug_file, "a") as f:
            await f.write("\n".join(log_lines))
    
    async def _generate_until_goal(
        self,
        engine: InferenceEngine,
        user_content: str,
        assistant_context: str,
        version: int = -1,
        sample_idx: int = -1,
        qid: str = "unknown",
    ) -> ModelResponse | None:
        """Generate until </Goal> tag is encountered.
        
        Uses string-based context: rebuilds the full prompt each time from
        user_content and assistant_context.
        
        Args:
            engine: The inference engine
            user_content: The original user problem/question
            assistant_context: Previously generated assistant content
            version: Model version for tracking
            sample_idx: Sample index for logging
            qid: Query ID for logging
        
        Returns None if context already exceeds MAX_POS_ENCODING.
        """
        # Build full prompt from strings
        full_prompt = format_prompt_with_context(user_content, assistant_context)
        input_ids = self.tokenizer.encode(full_prompt, add_special_tokens=False)
        
        # Check if we have room to generate
        remaining_tokens = MAX_POS_ENCODING - len(input_ids)
        if remaining_tokens <= 0:
            # No room to generate, return None to signal early stop
            await self._log_model_call("GOAL_GENERATION", input_ids, None, version, sample_idx, qid)
            return None
        
        # Add stop words for goal generation
        goal_stop_words = ["</Goal>", "<|im_end|>", "<|endoftext|>", "</s>"]
        gconfig = self.gconfig.new(
            n_samples=1,
            max_new_tokens=min(self.max_goal_tokens, remaining_tokens),
            stop=goal_stop_words,
        )
        
        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            gconfig=gconfig,
            tokenizer=self.tokenizer,
        )
        
        resp = await engine.agenerate(req)
        await self._log_model_call("GOAL_GENERATION", input_ids, resp, version, sample_idx, qid)
        return resp
    
    async def _generate_path(
        self,
        engine: InferenceEngine,
        user_content: str,
        shared_context: str,
        path_prefix: str,
        version: int = -1,
        sample_idx: int = -1,
        qid: str = "unknown",
        path_idx: int = -1,
    ) -> ModelResponse | None:
        """Generate a single path given context and prefix.
        
        Uses string-based context: rebuilds the full prompt each time.
        Each path sees the same shared_context (prompt + goal), NOT other paths.
        
        Args:
            engine: The inference engine
            user_content: The original user problem/question
            shared_context: Context up to (and including) the Goal, shared by all paths
            path_prefix: The prefix for this path (e.g., "1", "2")
            version: Model version for tracking
            sample_idx: Sample index for logging
            qid: Query ID for logging
            path_idx: Index of this path for logging
        
        Returns None if context already exceeds MAX_POS_ENCODING.
        """
        # Build the prompt with context + path start
        # Each path sees: shared_context + "<Path>\n{prefix}"
        assistant_content = shared_context + f"<Path>\n{path_prefix}"
        full_prompt = format_prompt_with_context(user_content, assistant_content)
        full_input_ids = self.tokenizer.encode(full_prompt, add_special_tokens=False)
        
        # Check if we have room to generate
        remaining_tokens = MAX_POS_ENCODING - len(full_input_ids)
        if remaining_tokens <= 0:
            # No room to generate, return None to signal early stop
            await self._log_model_call(f"PATH_GENERATION_{path_idx}", full_input_ids, None, version, sample_idx, qid)
            return None
        
        # Add stop words for path generation
        path_stop_words = ["</Path>", "<|im_end|>", "<|endoftext|>", "</s>"]
        gconfig = self.gconfig.new(
            n_samples=1,
            max_new_tokens=min(self.max_path_tokens, remaining_tokens),
            stop=path_stop_words,
        )
        
        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=full_input_ids,
            gconfig=gconfig,
            tokenizer=self.tokenizer,
        )
        
        resp = await engine.agenerate(req)
        await self._log_model_call(f"PATH_GENERATION_{path_idx}", full_input_ids, resp, version, sample_idx, qid)
        return resp
    
    async def _generate_conclusion(
        self,
        engine: InferenceEngine,
        context_ids: list[int],
        version: int = -1,
        sample_idx: int = -1,
        qid: str = "unknown",
    ) -> ModelResponse | None:
        """Generate conclusion given full context with all paths.
        
        NOTE: This method is currently UNUSED. The workflow now skips conclusion
        generation and only adds a <Conclusion>\\n marker for continuation.
        Kept for potential future use or if conclusion generation is re-enabled.
        
        Returns None if context already exceeds MAX_POS_ENCODING.
        """
        # Add conclusion start tag
        conclusion_start = "<Conclusion>\n"
        conclusion_start_ids = self.tokenizer.encode(conclusion_start, add_special_tokens=False)
        full_input_ids = context_ids + conclusion_start_ids
        
        # Check if we have room to generate
        remaining_tokens = MAX_POS_ENCODING - len(full_input_ids)
        if remaining_tokens <= 0:
            # No room to generate, return None to signal early stop
            await self._log_model_call("CONCLUSION_GENERATION", full_input_ids, None, version, sample_idx, qid)
            return None
        
        # Add stop words for conclusion generation
        conclusion_stop_words = ["</Conclusion>", "<|im_end|>", "<|endoftext|>", "</s>"]
        gconfig = self.gconfig.new(
            n_samples=1,
            max_new_tokens=min(self.max_conclusion_tokens, remaining_tokens),
            stop=conclusion_stop_words,
        )
        
        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=full_input_ids,
            gconfig=gconfig,
            tokenizer=self.tokenizer,
        )
        
        resp = await engine.agenerate(req)
        await self._log_model_call("CONCLUSION_GENERATION", full_input_ids, resp, version, sample_idx, qid)
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
        user_content: str,
        goal_context: str,
        goal_resp: ModelResponse,
        task_data: dict[str, Any],
        version: int = -1,
        sample_idx: int = -1,
        qid: str = "unknown",
    ) -> tuple[list[int], list[float], list[int], str, int, int]:
        """
        Process a parallel stage by generating paths (no conclusion).
        
        Uses string-based context: rebuilds the full prompt for each path.
        Each path sees the same shared context (goal_context), NOT other paths.
        Conclusion is skipped - only adds <Conclusion>\\n marker for continuation.
        
        Args:
            engine: The inference engine
            user_content: The original user problem/question
            goal_context: Context up to and including </Goal> (string)
            goal_resp: The ModelResponse from goal generation (for logprobs)
            task_data: Task data (unused, kept for compatibility)
            version: Model version for tracking
            sample_idx: Sample index for logging
            qid: Query ID for logging
        
        Returns:
            Tuple of (all_tokens, all_logprobs, all_versions, full_completion_str, 
                      total_path_tokens, longest_path_tokens)
            - total_path_tokens: Total number of tokens enclosed in <Path>...</Path> tags
            - longest_path_tokens: The longest path's token count in this stage
        """
        # logger.info("Entering parallel stage")
        
        # Build goal tokens
        goal_tokens = goal_resp.output_tokens
        goal_str = self.tokenizer.decode(goal_tokens)
        
        # Add closing </Goal> tag if not present
        if not goal_str.strip().endswith("</Goal>"):
            goal_close_ids = self.tokenizer.encode("</Goal>", add_special_tokens=False)
            goal_tokens = goal_tokens + goal_close_ids
            goal_str = self.tokenizer.decode(goal_tokens)
        
        # Check token count using tokenized full context
        initial_prompt = format_prompt_with_context(user_content, "")
        initial_input_ids = self.tokenizer.encode(initial_prompt, add_special_tokens=False)
        context_ids = initial_input_ids + goal_tokens
        
        # Check if we've already exceeded MAX_POS_ENCODING after goal generation
        if len(context_ids) >= MAX_POS_ENCODING:
            # Return just the goal tokens, clipped if necessary
            all_tokens = goal_tokens
            all_logprobs = goal_resp.output_logprobs + [0.0] * (len(goal_tokens) - len(goal_resp.output_logprobs))
            all_versions = goal_resp.output_versions + [-1] * (len(goal_tokens) - len(goal_resp.output_versions))
            full_completion_str = self.tokenizer.decode(all_tokens)
            # No paths generated, so path tokens = 0, longest path = 0
            return all_tokens, all_logprobs, all_versions, full_completion_str, 0, 0
        
        # Extract outline prefixes from goal
        outline_prefixes = extract_outline_prefixes(goal_str)
        if not outline_prefixes:
            outline_prefixes = ["1"]
        
        num_paths = len(outline_prefixes)
        # logger.info(f"Generating {num_paths} paths")
        
        # Generate all paths - each path sees the same shared context (goal_context)
        # This is string-based: we pass user_content and goal_context to _generate_path
        async with atrace_session_phase("generate_paths"):
            path_tasks = [
                self._generate_path(engine, user_content, goal_context, prefix, version, sample_idx, qid, i)
                for i, prefix in enumerate(outline_prefixes)
            ]
            path_resps = await asyncio.gather(*path_tasks)
        
        # Build complete context with all paths
        # Track path token counts for metrics
        all_path_tokens = []
        valid_path_data = []  # Track (prefix, path_resp, path_total_tokens) for valid paths
        path_token_lengths = []  # Track individual path lengths for finding the longest
        total_path_tokens_in_stage = 0  # Total tokens in all paths (including markers)
        
        for prefix, path_resp in zip(outline_prefixes, path_resps):
            if path_resp is None:
                # Skip this path if generation was not possible (exceeded MAX_POS_ENCODING)
                continue
            
            # Add path markers
            path_open = f"<Path>\n{prefix}"
            path_open_ids = self.tokenizer.encode(path_open, add_special_tokens=False)
            path_content_ids = path_resp.output_tokens
            
            # Add closing </Path> tag
            path_str = self.tokenizer.decode(path_content_ids)
            if not path_str.strip().endswith("</Path>"):
                path_close_ids = self.tokenizer.encode("</Path>", add_special_tokens=False)
                path_content_ids = path_content_ids + path_close_ids
            
            # Calculate total tokens for this path (including markers)
            this_path_total_tokens = len(path_open_ids) + len(path_content_ids)
            path_token_lengths.append(this_path_total_tokens)
            total_path_tokens_in_stage += this_path_total_tokens
            
            all_path_tokens.extend(path_open_ids + path_content_ids)
            valid_path_data.append((prefix, path_resp))
        
        # Find the longest path in this stage
        longest_path_tokens = max(path_token_lengths) if path_token_lengths else 0
        
        # Skip conclusion generation - just add <Conclusion>\n marker
        # This matches generation_new.py behavior
        conclusion_open = "<Conclusion>\n"
        conclusion_open_ids = self.tokenizer.encode(conclusion_open, add_special_tokens=False)
        
        # Assemble full sequence (goal + paths + conclusion marker, NO conclusion content)
        all_tokens = goal_tokens + all_path_tokens + conclusion_open_ids
        
        # Build logprobs and versions arrays
        # Goal tokens
        all_logprobs = goal_resp.output_logprobs + [0.0] * (len(goal_tokens) - len(goal_resp.output_logprobs))
        all_versions = goal_resp.output_versions + [-1] * (len(goal_tokens) - len(goal_resp.output_versions))
        
        # Path tokens (we only have logprobs for actual generated content, not markers)
        for prefix, path_resp in valid_path_data:
            path_open = f"<Path>\n{prefix}"
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
        
        # Conclusion marker tokens (no conclusion content)
        all_logprobs.extend([0.0] * len(conclusion_open_ids))
        all_versions.extend([-1] * len(conclusion_open_ids))
        
        full_completion_str = self.tokenizer.decode(all_tokens)
        # logger.info(f"Parallel stage complete: {num_paths} paths, {len(all_tokens)} tokens (no conclusion)")
        
        return all_tokens, all_logprobs, all_versions, full_completion_str, total_path_tokens_in_stage, longest_path_tokens
    
    async def _generate_single_trajectory(
        self,
        engine: InferenceEngine,
        user_content: str,
        input_ids: list[int],
        prompt_str: str,
        data: dict[str, Any],
        version: int,
        sample_idx: int,
    ) -> dict[str, Any]:
        """
        Generate a single trajectory for one sample with multi-turn support.
        
        Uses string-based context: tracks assistant_context as a string and rebuilds
        the full prompt each time. This matches generation_new.py behavior.
        
        Args:
            engine: The inference engine
            user_content: The original user problem/question (extracted from data)
            input_ids: Tokenized initial prompt (for building final trajectory)
            prompt_str: Decoded prompt string
            data: Task data including answer
            version: Current model version
            sample_idx: Index of this sample in the group
        
        Returns:
            Dict with trajectory data for this sample
        """
        # Get unique identifier for this prompt
        qid = None
        for key in ["query_id", "id", "qid"]:
            qid = data.get(key, None)
            if qid is not None:
                break
        qid = qid or uuid.uuid4().hex
        
        # Initialize tracking for multi-turn generation
        all_output_tokens = []
        all_output_logprobs = []
        all_output_versions = []
        all_completion_strs = []
        
        # Initialize tracking for parallel metrics
        total_path_tokens = 0  # Total tokens in all <Path>...</Path> across all stages
        sum_longest_path_per_stage = 0  # Sum of longest path tokens per stage (for latency calculation)
        
        # String-based context tracking (like generation_new.py)
        # This is the accumulated assistant response content
        assistant_context = ""
        
        # Multi-turn loop
        for turn_idx in range(self.max_turns):
            # Check token count using current context
            current_prompt = format_prompt_with_context(user_content, assistant_context)
            current_context_ids = self.tokenizer.encode(current_prompt, add_special_tokens=False)
            
            # Check if we still have room to generate
            completion_so_far = "".join(all_completion_strs)
            if len(current_context_ids) >= MAX_POS_ENCODING or completion_so_far.endswith("<|im_end|}") or "</think>" in completion_so_far:
                # No more room, stop generating
                break
            
            # Step 1: Generate until </Goal>
            async with atrace_session_phase(f"generate_goal_{sample_idx}_turn{turn_idx}"):
                goal_resp = await self._generate_until_goal(engine, user_content, assistant_context, version, sample_idx, qid)
            
            # Handle case where goal generation was not possible (exceeded MAX_POS_ENCODING)
            if goal_resp is None:
                # Cannot continue, stop here
                break
            
            # Check if model generated <|im_end|>, stop here
            if goal_resp.output_tokens and goal_resp.output_tokens[-1] == 151645:
                all_output_tokens.extend(goal_resp.output_tokens)
                all_output_logprobs.extend(goal_resp.output_logprobs)
                all_output_versions.extend(goal_resp.output_versions)
                all_completion_strs.append(self.tokenizer.decode(goal_resp.output_tokens))
                break
            
            # Decode goal and update string context
            goal_str = self.tokenizer.decode(goal_resp.output_tokens)
            
            # Add closing </Goal> tag if stopped at </Outline>
            if goal_str.strip().endswith("</Outline>") and not goal_str.strip().endswith("</Goal>"):
                goal_str = goal_str + "</Goal>"
            
            # Update string context with goal (this is the shared context for paths)
            goal_context = assistant_context + goal_str
            
            # Step 2: Process parallel stage (paths only, no conclusion)
            # Pass string-based context to _process_parallel_stage
            turn_output_tokens, turn_output_logprobs, turn_output_versions, turn_completion_str, stage_path_tokens, stage_longest_path = await self._process_parallel_stage(
                engine, user_content, goal_context, goal_resp, data, version, sample_idx, qid
            )
            
            # Accumulate parallel metrics
            total_path_tokens += stage_path_tokens
            sum_longest_path_per_stage += stage_longest_path
            
            # Accumulate outputs from this turn
            all_output_tokens.extend(turn_output_tokens)
            all_output_logprobs.extend(turn_output_logprobs)
            all_output_versions.extend(turn_output_versions)
            all_completion_strs.append(turn_completion_str)
            
            # Update string context for next turn: add the full turn output
            assistant_context += turn_completion_str
            
            # Check stopping conditions
            completion_so_far = "".join(all_completion_strs)
            current_prompt = format_prompt_with_context(user_content, assistant_context)
            current_context_ids = self.tokenizer.encode(current_prompt, add_special_tokens=False)
            
            if len(current_context_ids) >= MAX_POS_ENCODING or completion_so_far.endswith("<|im_end|>") or "</think>" in completion_so_far:
                break
        
        # Combine all turns into a single completion string
        completion_str = "".join(all_completion_strs)
        
        # Step 3: Compute reward (using full completion)
        full_sequence = input_ids + all_output_tokens
        reward = await self._compute_reward(
            prompt_str,
            completion_str,
            input_ids,
            all_output_tokens,
            data,
        )
        
        # Step 4: Build trajectory
        logprobs = [0.0] * len(input_ids) + all_output_logprobs
        loss_mask = [0] * len(input_ids) + [1] * len(all_output_tokens)
        versions = [-1] * len(input_ids) + all_output_versions
        
        # Step 4.5: Clip to MAX_POS_ENCODING if sequence exceeds limit
        if len(full_sequence) > MAX_POS_ENCODING:
            full_sequence = full_sequence[:MAX_POS_ENCODING]
            logprobs = logprobs[:MAX_POS_ENCODING]
            loss_mask = loss_mask[:MAX_POS_ENCODING]
            versions = versions[:MAX_POS_ENCODING]
        
        # Calculate parallel metrics
        # Total sequence length of the generation (output tokens only, excluding prompt)
        total_gen_tokens = len(all_output_tokens)
        
        # Parallel ratio: total path tokens / total generation tokens
        parallel_ratio = total_path_tokens / total_gen_tokens if total_gen_tokens > 0 else 0.0
        
        # Total latency = total_gen_tokens - total_path_tokens + sum_longest_path_per_stage
        # This represents the effective sequential tokens (non-parallel + one path per stage)
        total_latency = total_gen_tokens - total_path_tokens + sum_longest_path_per_stage
        
        return {
            "full_sequence": full_sequence,
            "logprobs": logprobs,
            "loss_mask": loss_mask,
            "versions": versions,
            "reward": reward,
            "completion_str": completion_str,
            # Parallel metrics
            "total_gen_tokens": total_gen_tokens,
            "total_path_tokens": total_path_tokens,
            "parallel_ratio": parallel_ratio,
            "total_latency": total_latency,
        }
    
    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        """
        Run a single episode of parallel generation with GRPO support.
        
        Uses string-based context tracking. For each sample:
        1. Generates Goal with outlines
        2. Generates parallel paths (one per outline), each seeing the same context
        3. Adds <Conclusion>\\n marker (skips conclusion generation)
        4. Continues to next turn or finishes
        
        Returns trajectory dict with keys (batch size = n_samples):
            - input_ids: Full sequences (prompt + generation) for all samples
            - loss_mask: Mask for training (1 for generated, 0 for prompt)
            - logprobs: Log probabilities for each token
            - versions: Model versions for each token
            - attention_mask: All ones
            - rewards: Scalar rewards for each sample
        """
        # Load reward function dynamically if given as string
        if isinstance(self.reward_fn, str):
            self.reward_fn = import_from_string(self.reward_fn)
            self.async_reward_fn = AsyncRewardWrapper(self.reward_fn, 60)
        
        # Extract user content for string-based generation
        prompt_data = self.data_extract_prompt_fn(data)
        if isinstance(prompt_data, str):
            user_content = prompt_data
        elif isinstance(prompt_data, list) and len(prompt_data) > 0 and isinstance(prompt_data[0], dict):
            # Extract from messages format - get the user message content
            user_content = prompt_data[0].get("content", "")
        else:
            user_content = str(prompt_data)
        
        # Get input_ids
        input_ids = self.get_input_ids_fn(
            prompt_data,
            self.tokenizer,
            self.enable_thinking,
            self.format_prompt_fn,
        )
        
        version = engine.get_version()
        prompt_str = self.tokenizer.decode(input_ids)
        
        # Get number of samples for GRPO (from gconfig.n_samples)
        n_samples = self.gconfig.n_samples if hasattr(self.gconfig, 'n_samples') else 1
        
        # Generate n_samples trajectories in parallel for GRPO
        async with atrace_session_phase("generate_group"):
            trajectory_tasks = [
                self._generate_single_trajectory(
                    engine, user_content, input_ids, prompt_str, data, version, i
                )
                for i in range(n_samples)
            ]
            trajectories = await asyncio.gather(*trajectory_tasks)
        
        # Collect original rewards and parallel metrics
        original_rewards = [t["reward"] for t in trajectories]
        parallel_ratio_list = [t["parallel_ratio"] for t in trajectories]
        total_latency_list = [t["total_latency"] for t in trajectories]
        
        # Apply reward shaping if enabled
        if self.reward_shaping_mode is not None and len(trajectories) > 1:
            if self.reward_shaping_mode == "parallel_ratio":
                # Parallel ratio based: reward = original * [1 + alpha * (parallel_ratio - avg) / std]
                avg_pr = sum(parallel_ratio_list) / len(parallel_ratio_list)
                std_pr = torch.tensor(parallel_ratio_list, dtype=torch.float32).std().item()
                if std_pr > 1e-8:  # Avoid division by zero
                    for i, traj in enumerate(trajectories):
                        shaping_factor = 1.0 + self.reward_shaping_alpha * (traj["parallel_ratio"] - avg_pr) / std_pr
                        traj["reward"] = traj["reward"] * shaping_factor
            elif self.reward_shaping_mode == "latency":
                # Latency based: reward = original * [1 - alpha * (latency - avg) / std]
                avg_lat = sum(total_latency_list) / len(total_latency_list)
                std_lat = torch.tensor(total_latency_list, dtype=torch.float32).std().item()
                if std_lat > 1e-8:  # Avoid division by zero
                    for i, traj in enumerate(trajectories):
                        shaping_factor = 1.0 - self.reward_shaping_alpha * (traj["total_latency"] - avg_lat) / std_lat
                        traj["reward"] = traj["reward"] * shaping_factor
        
        # Collect rewards for stats (after shaping)
        rewards = [t["reward"] for t in trajectories]
        avg_reward = sum(rewards) / len(rewards)
        
        # Collect parallel metrics for stats
        total_gen_tokens_list = [t["total_gen_tokens"] for t in trajectories]
        total_path_tokens_list = [t["total_path_tokens"] for t in trajectories]
        
        avg_total_gen_tokens = sum(total_gen_tokens_list) / len(total_gen_tokens_list)
        avg_total_path_tokens = sum(total_path_tokens_list) / len(total_path_tokens_list)
        avg_parallel_ratio = sum(parallel_ratio_list) / len(parallel_ratio_list)
        avg_total_latency = sum(total_latency_list) / len(total_latency_list)
        
        # Calculate original reward stats (before shaping)
        avg_original_reward = sum(original_rewards) / len(original_rewards)
        
        stats_dict = {
            "reward": avg_reward,
            "reward_std": torch.tensor(rewards, dtype=torch.float32).std().item() if len(rewards) > 1 else 0.0,
            "n_samples": n_samples,
            # Parallel metrics
            "total_gen_tokens": avg_total_gen_tokens,
            "total_path_tokens": avg_total_path_tokens,
            "parallel_ratio": avg_parallel_ratio,
            "total_latency": avg_total_latency,
        }
        
        # Add original reward stats if reward shaping is enabled
        if self.reward_shaping_mode is not None:
            stats_dict["original_reward"] = avg_original_reward
            stats_dict["original_reward_std"] = torch.tensor(original_rewards, dtype=torch.float32).std().item() if len(original_rewards) > 1 else 0.0
        
        stats_tracker.get(self.rollout_stat_scope).scalar(**stats_dict)
        
        # Concatenate all trajectories into batch format
        all_results = []
        for i, traj in enumerate(trajectories):
            result = {
                "input_ids": torch.tensor(traj["full_sequence"], dtype=torch.int32).unsqueeze(0),
                "loss_mask": torch.tensor(traj["loss_mask"], dtype=torch.int32).unsqueeze(0),
                "logprobs": torch.tensor(traj["logprobs"], dtype=torch.float32).unsqueeze(0),
                "versions": torch.tensor(traj["versions"], dtype=torch.int32).unsqueeze(0),
                "attention_mask": torch.ones(len(traj["full_sequence"]), dtype=torch.bool).unsqueeze(0),
                "rewards": torch.tensor(traj["reward"], dtype=torch.float32).unsqueeze(0),
            }
            all_results.append(result)
        
        # Concatenate all trajectories using the existing utility
        final_result = concat_padded_tensors(all_results)
        
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
            
            # Dump all rollouts to file
            file_path = os.path.join(dump_path, f"{qid}.txt")
            async with aiofiles.open(file_path, "a") as f:
                for i, traj in enumerate(trajectories):
                    info = "\n".join([
                        f"[Sample {i+1}/{n_samples}] seqlen: {len(traj['full_sequence'])}, reward: {traj['reward']}",
                        f"prompt:\n{colorama.Fore.YELLOW + colorama.Style.DIM}{prompt_str}{colorama.Style.RESET_ALL}",
                        f"completion:\n{colorama.Fore.GREEN + colorama.Style.DIM}{traj['completion_str']}{colorama.Style.RESET_ALL}",
                        "-" * 40,
                    ])
                    await f.write(info + "\n")
                await f.write("=" * 80 + "\n")
        
        return final_result


# Define reward function at module level (must be picklable for multiprocessing)
def simple_reward_fn(prompt, completion, input_tokens, output_tokens, answer, **kwargs):
    """
    Simple reward function that checks if the answer appears in the completion.
    For real use, you should use a proper math answer parser.
    """
    # Check if the expected answer appears in the completion
    if answer in completion:
        return 1.0
    else:
        return 0.0


async def test_main():
    """
    Test function for ParallelGenerationWorkflow with a local SGLang backend.
    
    This function demonstrates how to:
    1. Connect to a running SGLang server (localhost:30000)
    2. Create a ParallelGenerationWorkflow instance
    3. Generate structured parallel completions for a test problem
    4. Display the results
    
    Prerequisites:
    - SGLang server running at localhost:30000
    - Example: python -m sglang.launch_server --model-path <your-model> --port 30000
    """
    import os
    
    # Import required modules
    from areal.api.cli_args import InferenceEngineConfig, GenerationHyperparameters
    from areal.engine.sglang_remote import RemoteSGLangEngine
    from areal.utils.hf_utils import load_hf_tokenizer
    
    # ========== Configuration ==========
    
    # SGLang server address (change if your server is at a different address)
    server_addr = "localhost:30000"
    
    # Set environment variable for server discovery
    os.environ["AREAL_LLM_SERVER_ADDRS"] = server_addr
    
    # Tokenizer path (should match your model)
    # TODO: Change this to your actual tokenizer path
    tokenizer_path = "/storage/openpsi/models/zzy/Qwen3-4B-s1-parallel-only-5"
    
    # Test problem
    test_problem = {
        "messages": [
            {
                "role": "user",
                "content": "Solve the following math problem step by step:\n\nJanet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
            }
        ],
        "answer": "18",  # The correct answer
        "query_id": "test_001",
    }
    
    # ========== Setup Engine ==========
    
    print("=" * 80)
    print("Testing Parallel Generation Workflow")
    print("=" * 80)
    print(f"\n[1/5] Connecting to SGLang server at {server_addr}...")
    
    # Create inference engine config
    engine_config = InferenceEngineConfig(
        experiment_name="test_parallel_gen",
        trial_name="trial_0",
        setup_timeout=120,
    )
    
    # Create and initialize the engine
    engine = RemoteSGLangEngine(engine_config)
    engine.initialize(addr=server_addr)
    
    print(f"✓ Connected to SGLang server")
    
    # ========== Setup Workflow ==========
    
    print(f"\n[2/5] Setting up ParallelGenerationWorkflow...")
    
    # Load tokenizer
    tokenizer = load_hf_tokenizer(tokenizer_path)
    
    # Generation hyperparameters
    gconfig = GenerationHyperparameters(
        n_samples=1,  # Generate 1 sample only
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=2048,
    )
    
    # Create workflow
    workflow = ParallelGenerationWorkflow(
        reward_fn=simple_reward_fn,
        gconfig=gconfig,
        tokenizer=tokenizer,
        enable_thinking=False,
        max_goal_tokens=32768,
        max_path_tokens=32768,
        max_conclusion_tokens=32768,
        max_turns=2,  # Allow up to 2 goal-paths-conclusion cycles
        dump_dir="./test_parallel_gen_output",  # Save outputs here
        format_prompt_fn=None,  # Use default chat template
        reward_shaping_mode=None,  # No reward shaping for testing
    )
    
    print(f"✓ Workflow created with {gconfig.n_samples} samples per problem")
    
    # ========== Run Generation ==========
    
    print(f"\n[3/5] Generating structured parallel completions...")
    print(f"Problem: {test_problem['messages'][0]['content'][:100]}...")
    
    # Run the workflow
    result = await workflow.arun_episode(engine, test_problem)
    
    print(f"✓ Generation complete!")
    
    # ========== Display Results ==========
    
    print(f"\n[4/5] Results Summary:")
    print("=" * 80)
    
    # Extract results
    input_ids = result["input_ids"]  # Shape: [n_samples, seq_len]
    rewards = result["rewards"]  # Shape: [n_samples]
    loss_mask = result["loss_mask"]  # Shape: [n_samples, seq_len]
    
    n_samples = input_ids.shape[0]
    
    for i in range(n_samples):
        print(f"\n--- Sample {i+1}/{n_samples} ---")
        
        # Decode full sequence
        full_seq = tokenizer.decode(input_ids[i].tolist(), skip_special_tokens=False)
        
        # Find where generation starts (where loss_mask becomes 1)
        gen_start_idx = (loss_mask[i] == 1).nonzero(as_tuple=True)[0][0].item()
        
        # Decode prompt and generation separately
        prompt = tokenizer.decode(input_ids[i][:gen_start_idx].tolist(), skip_special_tokens=True)
        generation = tokenizer.decode(input_ids[i][gen_start_idx:].tolist(), skip_special_tokens=False)
        
        print(f"Reward: {rewards[i].item():.2f}")
        print(f"Sequence length: {len(input_ids[i])} tokens")
        print(f"Generated tokens: {(loss_mask[i] == 1).sum().item()} tokens")
        print(f"\nPrompt (first 200 chars):\n{prompt[:200]}...")
        print(f"\nGeneration (first 500 chars):\n{generation[:50000]}...")
        
        # Count goal/path/conclusion blocks
        n_goals = generation.count("<Goal>")
        n_paths = generation.count("<Path>")
        n_conclusions = generation.count("<Conclusion>")
        print(f"\nStructure: {n_goals} Goal(s), {n_paths} Path(s), {n_conclusions} Conclusion(s)")
    
    # ========== Cleanup ==========
    
    print(f"\n[5/5] Cleaning up...")
    engine.destroy()
    print(f"✓ Engine destroyed")
    
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print(f"Full outputs saved to: ./test_parallel_gen_output")
    print("=" * 80)


if __name__ == "__main__":
    """
    Run this script to test the ParallelGenerationWorkflow.
    
    Usage:
        python parallel_generation.py
    
    Prerequisites:
        1. Start SGLang server:
           python -m sglang.launch_server --model-path <your-model> --port 30000
        
        2. Run this script:
           python parallel_generation.py
    """
    import asyncio
    asyncio.run(test_main())