# SPDX-License-Identifier: Apache-2.0

"""Multi-step tool-use workflow with CubeSandbox isolation.

This workflow enables RL training with sandboxed code execution. Each
episode follows a generate → detect tool call → sandbox execute → append
result loop, producing trajectories compatible with AReaL's training
pipeline.

Architecture
------------
1. Per-episode sandbox lifecycle: each ``arun_episode`` creates (or checks
   out from pool) an isolated sandbox, and releases it on completion.
2. Multi-turn tool-call loop: model generates text, tool calls are detected
   via configurable markers, code is executed in the sandbox, results are
   appended back to the conversation.
3. Reward computation: supports both final-turn and per-step reward.

Example
-------
.. code-block:: python

    from areal.workflow.sandbox_tool import SandboxToolWorkflow
    from areal.api.sandbox_api import SandboxConfig

    workflow = SandboxToolWorkflow(
        reward_fn="areal.reward.gsm8k.gsm8k_reward_fn",
        gconfig=gconfig,
        tokenizer="Qwen/Qwen3-8B",
        sandbox_config=SandboxConfig(enabled=True, backend="cube", api_url="http://..."),
    )

See Also
--------
areal.api.sandbox_api : Sandbox abstractions.
areal.infra.sandbox : Sandbox backends and pooling.
areal.workflow.rlvr : Single-turn RLVR workflow (similar pattern).
"""

from __future__ import annotations

import copy
import re
import uuid
from collections.abc import Callable
from typing import Any

import torch
from transformers import PreTrainedTokenizerFast

from areal import workflow_context
from areal.api import (
    AsyncRewardWrapper,
    InferenceEngine,
    ModelRequest,
    ModelResponse,
    RolloutWorkflow,
)
from areal.api.cli_args import GenerationHyperparameters
from areal.api.sandbox_api import ExecutionResult, SandboxConfig, SandboxExecutor
from areal.utils import logging, stats_tracker
from areal.utils.dynamic_import import import_from_string

logger = logging.getLogger("SandboxToolWorkflow")

# Default code markers
_DEFAULT_CODE_START_MARKERS = ["```python\n", "<python>"]
_DEFAULT_CODE_END_MARKERS = ["```", "</python>"]

# Regex to extract python code from fenced blocks
_CODE_BLOCK_RE = re.compile(
    r"```python\n(.*?)```|<python>(.*?)</python>",
    re.DOTALL,
)


def _extract_code(text: str) -> str:
    """Extract the last Python code block from text."""
    matches = list(_CODE_BLOCK_RE.finditer(text))
    if not matches:
        return ""
    last = matches[-1]
    return (last.group(1) or last.group(2) or "").strip()


class SandboxToolWorkflow(RolloutWorkflow):
    """Multi-step tool-use workflow with sandbox isolation.

    Each episode:
    1. Create / checkout a sandbox instance
    2. Loop: generate → detect code block → execute in sandbox → append output
    3. Compute reward on final output
    4. Return / checkin sandbox

    Parameters
    ----------
    reward_fn : Callable | str
        Reward function or import path string.
    gconfig : GenerationHyperparameters
        Generation hyperparameters.
    tokenizer : PreTrainedTokenizerFast | str
        Tokenizer instance or HuggingFace model name / path.
    sandbox_config : SandboxConfig
        Sandbox backend and connection configuration.
    enable_thinking : bool
        Whether to enable thinking tokens in chat template.
    code_start_markers : list[str] | None
        Markers that indicate start of code block.
    code_end_markers : list[str] | None
        Markers that indicate end of code block.
    system_prompt : str | None
        System prompt prepended to conversations. If None, a default
        tool-use system prompt is used.
    data_extract_prompt_fn : Callable | str | None
        Function to extract prompt from data dict. Default uses ``data["messages"]``.
    """

    def __init__(
        self,
        reward_fn: Callable[..., Any] | str,
        gconfig: GenerationHyperparameters,
        tokenizer: PreTrainedTokenizerFast | str,
        sandbox_config: SandboxConfig | None = None,
        enable_thinking: bool = False,
        code_start_markers: list[str] | None = None,
        code_end_markers: list[str] | None = None,
        system_prompt: str | None = None,
        data_extract_prompt_fn: Callable[[dict[str, Any]], Any] | str | None = None,
    ):
        super().__init__()

        # Tokenizer
        if isinstance(tokenizer, str):
            from areal.utils.hf_utils import load_hf_tokenizer

            tokenizer = load_hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizerFast = tokenizer

        # Reward
        self.reward_fn = reward_fn
        if not isinstance(reward_fn, str):
            self.async_reward_fn = AsyncRewardWrapper(reward_fn)

        # Generation config
        self.gconfig = gconfig.new_with_stop_and_pad_token_ids(self.tokenizer)
        self.enable_thinking = enable_thinking

        # Sandbox config
        if sandbox_config is None:
            sandbox_config = SandboxConfig(enabled=True, backend="local")
        self.sandbox_config = sandbox_config

        # Tool markers
        self.code_start_markers = code_start_markers or _DEFAULT_CODE_START_MARKERS
        self.code_end_markers = code_end_markers or _DEFAULT_CODE_END_MARKERS

        # System prompt
        if system_prompt is None:
            system_prompt = (
                "You are a helpful assistant. You can write and execute Python code "
                "to solve problems. Put your code in ```python\\n...``` blocks. "
                "The code will be executed in a sandboxed environment and the output "
                "will be returned to you. Use code execution to verify your reasoning."
            )
        self.system_prompt = system_prompt

        # Data extraction
        if data_extract_prompt_fn is None:
            data_extract_prompt_fn = lambda data: data["messages"]  # noqa: E731
        elif isinstance(data_extract_prompt_fn, str):
            data_extract_prompt_fn = import_from_string(data_extract_prompt_fn)
        self.data_extract_prompt_fn = data_extract_prompt_fn

        # Sandbox manager (lazily initialized)
        self._sandbox_manager_configured = False

    def _ensure_sandbox_manager(self) -> None:
        """Configure the per-thread sandbox manager if not done yet."""
        if self._sandbox_manager_configured:
            return
        from areal.infra.sandbox.manager import configure

        configure(self.sandbox_config)
        self._sandbox_manager_configured = True

    async def _get_sandbox(self) -> SandboxExecutor:
        """Get a sandbox instance (from pool or newly created)."""
        self._ensure_sandbox_manager()
        from areal.infra.sandbox.manager import checkout_sandbox

        return await checkout_sandbox()

    async def _return_sandbox(self, sandbox: SandboxExecutor) -> None:
        """Return sandbox to pool."""
        from areal.infra.sandbox.manager import checkin_sandbox

        await checkin_sandbox(sandbox)

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, torch.Tensor] | None:
        """Run a single tool-use episode with sandbox isolation.

        Parameters
        ----------
        engine : InferenceEngine
            The inference engine for text generation.
        data : dict[str, Any]
            Input data containing at least ``"messages"`` key.

        Returns
        -------
        dict[str, torch.Tensor] | None
            Trajectory tensors with keys: ``input_ids``, ``logprobs``,
            ``loss_mask``, ``versions``, ``attention_mask``, ``rewards``.
            Returns None if the trajectory should be rejected.
        """
        # Lazy-load reward function if given as string
        if isinstance(self.reward_fn, str):
            self.reward_fn = import_from_string(self.reward_fn)
            self.async_reward_fn = AsyncRewardWrapper(self.reward_fn)

        # Prepare messages
        messages = self.data_extract_prompt_fn(data)
        messages = copy.deepcopy(messages)

        # Ensure system prompt
        if messages and messages[0]["role"] == "system":
            messages[0]["content"] = self.system_prompt
        else:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        # Tokenize
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        input_ids = list(input_ids)

        # Get sandbox
        sandbox = await self._get_sandbox()
        try:
            return await self._multi_turn_with_sandbox(engine, sandbox, input_ids, data)
        finally:
            await self._return_sandbox(sandbox)

    async def _multi_turn_with_sandbox(
        self,
        engine: InferenceEngine,
        sandbox: SandboxExecutor,
        prompt_ids: list[int],
        data: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Execute multi-turn generate → execute → append loop.

        Returns
        -------
        dict[str, torch.Tensor]
            Trajectory tensor dict with batch dimension 1.
        """
        max_len = (
            self.gconfig.max_new_tokens + len(prompt_ids)
            if self.gconfig.max_new_tokens
            else 4096
        )
        max_turns = self.sandbox_config.max_tool_turns
        tool_timeout = self.sandbox_config.timeout

        prompt_str = self.tokenizer.decode(prompt_ids)
        completions_str = ""

        # Sequence-level accumulators
        context_ids = copy.deepcopy(prompt_ids)
        seq = copy.deepcopy(prompt_ids)
        logprobs: list[float] = [0.0] * len(context_ids)
        loss_mask: list[int] = [0] * len(context_ids)
        versions: list[int] = [-1] * len(context_ids)
        output_ids: list[int] = []

        tool_call_count = 0
        tool_success_count = 0
        has_tool = False
        # State: waiting for code start or code end
        waiting_for_start = True
        code_start_idx = -1
        turn = 0

        while turn <= max_turns:
            if len(context_ids) >= max_len:
                break

            # Select stop markers based on state
            if waiting_for_start:
                stop_markers = list(self.code_start_markers)
            else:
                stop_markers = list(self.code_end_markers)

            gconfig = self.gconfig.new(n_samples=1, stop=stop_markers)
            req = ModelRequest(
                rid=uuid.uuid4().hex,
                input_ids=context_ids,
                gconfig=gconfig,
                tokenizer=self.tokenizer,
            )

            resp: ModelResponse = await engine.agenerate(req)

            # Append generated tokens
            context_ids.extend(resp.output_tokens)
            seq.extend(resp.output_tokens)
            logprobs.extend(resp.output_logprobs)
            loss_mask.extend([1] * resp.output_len)
            versions.extend(resp.output_versions)
            output_ids.extend(resp.output_tokens)

            cur_text = self.tokenizer.decode(resp.output_tokens)
            completions_str += cur_text

            # Check for EOS
            if context_ids[-1] in (
                self.tokenizer.pad_token_id,
                self.tokenizer.eos_token_id,
            ):
                break

            if len(context_ids) >= max_len:
                break

            # State machine transitions
            if waiting_for_start and resp.stop_reason == "stop":
                # Check if we stopped at a code start marker
                for marker in self.code_start_markers:
                    if cur_text.rstrip().endswith(marker.rstrip()):
                        waiting_for_start = False
                        code_start_idx = len(completions_str)
                        break
                else:
                    # No code marker found, model finished naturally
                    break

            elif not waiting_for_start and resp.stop_reason == "stop":
                # Stopped at code end marker — extract and execute code
                code_text = completions_str[code_start_idx:]
                code = _extract_code(completions_str) or code_text.strip()

                if not code:
                    waiting_for_start = True
                    continue

                # Execute in sandbox
                turn += 1
                has_tool = True
                tool_call_count += 1

                exec_result: ExecutionResult = await sandbox.run_code(
                    code, language="python", timeout=tool_timeout
                )

                if exec_result.success:
                    tool_success_count += 1

                # Format tool output
                output = exec_result.text or exec_result.stdout
                if exec_result.error:
                    output = f"Error: {exec_result.error}"
                if not output:
                    output = "(no output)"

                # Truncate long outputs
                if len(output) > 2000:
                    output = output[:1000] + "\n...(truncated)...\n" + output[-500:]

                tool_response = f"\n```output\n{output}\n```\n"

                # Tokenize tool response and append (masked, not trained on)
                tool_token_ids = self.tokenizer.encode(
                    tool_response, add_special_tokens=False
                )
                context_ids.extend(tool_token_ids)
                seq.extend(tool_token_ids)
                logprobs.extend([0.0] * len(tool_token_ids))
                loss_mask.extend([0] * len(tool_token_ids))
                versions.extend([-1] * len(tool_token_ids))
                completions_str += tool_response

                # Reset state for next tool call
                waiting_for_start = True
                code_start_idx = -1

            elif resp.stop_reason == "length":
                # Hit max tokens
                break

        # Compute reward
        reward = await self.async_reward_fn(
            prompt_str,
            completions_str,
            prompt_ids,
            output_ids,
            tool_using=has_tool,
            tool_status=tool_call_count,
            **data,
        )

        # Record stats
        stats_tracker.get(workflow_context.stat_scope()).scalar(
            reward=reward,
            sandbox_tool_call_count=tool_call_count,
            sandbox_tool_success_count=tool_success_count,
        )

        # Truncate to max_len
        seq = seq[:max_len]
        logprobs = logprobs[:max_len]
        loss_mask = loss_mask[:max_len]
        versions = versions[:max_len]

        res = {
            "input_ids": torch.tensor(seq, dtype=torch.int32),
            "logprobs": torch.tensor(logprobs, dtype=torch.float32),
            "loss_mask": torch.tensor(loss_mask, dtype=torch.int32),
            "versions": torch.tensor(versions, dtype=torch.int32),
            "attention_mask": torch.ones(len(seq), dtype=torch.bool),
            "rewards": torch.tensor(float(reward), dtype=torch.float32),
        }
        return {k: v.unsqueeze(0) for k, v in res.items()}
