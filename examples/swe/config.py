"""Configuration for SWE SFT training with AReaL."""

from dataclasses import dataclass, field

from areal.api.cli_args import SFTConfig


@dataclass
class SweDataConfig:
    """SWE-specific data processing configuration."""

    filter_errors: bool = field(
        default=True,
        metadata={
            "help": "Discard pairs whose current segment contains a tool result "
            "with is_error=True. Set to false to keep all pairs."
        },
    )
    pre_split: bool = field(
        default=False,
        metadata={
            "help": "Input JSONL is already in pair format "
            '(each line: {"messages": [...]}). '
            "Skip trajectory splitting and error filtering."
        },
    )
    num_proc: int = field(
        default=4,
        metadata={"help": "Number of parallel workers for tokenization."},
    )
    strip_all_thinking: bool = field(
        default=False,
        metadata={
            "help": "Strip <think>...</think> from ALL assistant turns "
            "including the training target. By default only context "
            "turns are stripped."
        },
    )
    no_tools: bool = field(
        default=False,
        metadata={
            "help": "Do not pass tool definitions to apply_chat_template. "
            "By default, tools are auto-extracted from the data and "
            "rendered in the system prompt (e.g. Qwen3 '# Tools' block)."
        },
    )
    filter_empty_tool_calls: bool = field(
        default=False,
        metadata={
            "help": "Discard pairs whose training-target assistant turn has "
            "no text content but has tool_calls (silent tool invocations)."
        },
    )
    filter_bare_text_tool_calls: bool = field(
        default=False,
        metadata={
            "help": "Discard pairs whose training-target assistant turn has "
            "text content without <think> tags and has tool_calls."
        },
    )
    truncate_task_notifications: bool = field(
        default=False,
        metadata={
            "help": "Truncate trajectories at the first <task-notification> "
            "that follows a pure-text assistant turn. Removes noise from "
            "background task completions."
        },
    )
    cleanup_processed_dataset: bool = field(
        default=True,
        metadata={
            "help": "Remove the processed dataset cache directory after training. "
            "Set to false to keep it for faster restarts."
        },
    )


@dataclass
class SweSFTConfig(SFTConfig):
    """SFT configuration with SWE-specific data processing settings."""

    swe: SweDataConfig = field(default_factory=SweDataConfig)
