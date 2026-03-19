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


@dataclass
class SweSFTConfig(SFTConfig):
    """SFT configuration with SWE-specific data processing settings."""

    swe: SweDataConfig = field(default_factory=SweDataConfig)
