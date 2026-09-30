# SPDX-License-Identifier: Apache-2.0

"""Train GLM on one length-capped tool-use conversation per JSON file."""

import argparse
import copy
import json
import sys
from collections import Counter
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from typing import Any

ROLE_TOKENS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "tool": "<|observation|>",
}


def read_trajectory(path: Path, include_reasoning: bool = True) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig") as stream:
        record = json.load(stream)
    if not isinstance(record, dict) or not isinstance(record.get("messages"), list):
        raise ValueError(f"{path}: expected one object containing a messages list")
    if not record["messages"]:
        raise ValueError(f"{path}: empty conversation")
    record = copy.deepcopy(record)
    for index, message in enumerate(record["messages"]):
        if not isinstance(message, dict) or message.get("role") not in ROLE_TOKENS:
            raise ValueError(f"{path}: unsupported message at index {index}")
        if not isinstance(message.get("content", ""), str):
            raise ValueError(f"{path}: message {index} must have text content")
        message.setdefault("content", "")
        if not include_reasoning:
            message.pop("reasoning_content", None)
            if "<think>" in message["content"]:
                raise ValueError(
                    f"{path}: embedded reasoning must be cleaned explicitly"
                )
        for call in message.get("tool_calls", []):
            function = call.get("function", {})
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError(
                    f"{path}: message {index} tool arguments must be objects"
                )
            function["arguments"] = arguments
    if not isinstance(record.get("tools", []), list):
        raise ValueError(f"{path}: tools must be a list")
    return record


def tokenize_trajectory(
    record: dict[str, Any], tokenizer: Any, max_length: int | None = None
) -> dict[str, list[int]]:
    """Mask by GLM role tokens, with strict checks against embedded role markers."""
    if max_length is not None and max_length <= 0:
        raise ValueError("max_length must be positive or null")
    role_ids = {}
    for role, marker in ROLE_TOKENS.items():
        ids = tokenizer.encode(marker, add_special_tokens=False)
        if len(ids) != 1 or ids[0] not in tokenizer.all_special_ids:
            raise ValueError(f"Expected a GLM tokenizer with special token {marker}")
        role_ids[ids[0]] = role
    if len(role_ids) != len(ROLE_TOKENS):
        raise ValueError("GLM role token IDs must be distinct")

    # One full render avoids repeatedly tokenizing growing million-token prefixes.
    input_ids = tokenizer.apply_chat_template(
        record["messages"],
        tools=record.get("tools") or None,
        tokenize=True,
        return_dict=False,
        add_generation_prompt=False,
        clear_thinking=False,
    )
    # Transformers versions and custom tokenizers differ in their return type.
    if isinstance(input_ids, Mapping):
        input_ids = input_ids["input_ids"]
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()
    if isinstance(input_ids, (list, tuple)) and len(input_ids) == 1:
        if isinstance(input_ids[0], (list, tuple)):
            input_ids = input_ids[0]
    if not isinstance(input_ids, (list, tuple)) or not all(
        isinstance(token, Integral) and not isinstance(token, bool)
        for token in input_ids
    ):
        raise ValueError("Chat template must return one sequence of integer token IDs")
    input_ids = [int(token) for token in input_ids]

    expected = []
    for message in record["messages"]:
        role = message["role"]
        # GLM groups consecutive tool responses under one observation marker.
        if role != "tool" or not expected or expected[-1] != "tool":
            expected.append(role)
    observed = [role_ids[token] for token in input_ids if token in role_ids]
    # Some installed templates emit an observation header for every tool result.
    ungrouped = [message["role"] for message in record["messages"]]

    def matches(roles: list[str]) -> bool:
        extra = len(observed) - len(roles)
        return (
            extra >= 0
            and observed[:extra] == ["system"] * extra
            and observed[extra:] == roles
        )

    if not matches(expected) and not matches(ungrouped):
        # Report structure only, never the potentially sensitive message contents.
        embedded = []
        for index, message in enumerate(record["messages"]):
            serialized = json.dumps(message, ensure_ascii=False)
            for marker in ROLE_TOKENS.values():
                if marker in serialized:
                    embedded.append(f"messages[{index}] contains {marker}")
                    if len(embedded) >= 5:
                        break
            if len(embedded) >= 5:
                break
        tool_text = json.dumps(record.get("tools", []), ensure_ascii=False)
        embedded.extend(
            f"tools contains {marker}"
            for marker in ROLE_TOKENS.values()
            if marker in tool_text
        )
        expected_body = list(expected)
        observed_body = list(observed)
        while expected_body and expected_body[0] == "system":
            expected_body.pop(0)
        while observed_body and observed_body[0] == "system":
            observed_body.pop(0)
        mismatch = next(
            (
                i
                for i, pair in enumerate(zip(expected_body, observed_body))
                if pair[0] != pair[1]
            ),
            min(len(expected_body), len(observed_body)),
        )
        raise ValueError(
            "Rendered role boundaries differ from the conversation. The template "
            "is unsupported or content contains reserved role tokens; refusing "
            "to generate an ambiguous loss mask. "
            f"Tokenizer={getattr(tokenizer, 'name_or_path', 'unknown')}; "
            f"expected_grouped={dict(Counter(expected))}; "
            f"expected_ungrouped={dict(Counter(ungrouped))}; "
            f"observed={dict(Counter(observed))}; "
            f"first_mismatch_after_initial_systems={mismatch}; "
            f"expected_next={expected_body[mismatch : mismatch + 5]}; "
            f"observed_next={observed_body[mismatch : mismatch + 5]}; "
            f"embedded_markers={embedded}."
        )

    loss_mask = []
    supervise = False
    for token in input_ids:
        if token in role_ids:
            supervise = role_ids[token] == "assistant"
            loss_mask.append(0)
        else:
            loss_mask.append(int(supervise))
    # Validate full role boundaries before applying the same prefix to both arrays.
    if max_length is not None:
        input_ids = input_ids[:max_length]
        loss_mask = loss_mask[:max_length]
    if not any(loss_mask):
        raise ValueError(
            "Retained conversation has no assistant tokens to supervise. "
            "Increase max_length if it ends before the first assistant output."
        )
    return {"input_ids": input_ids, "loss_mask": loss_mask}


def trajectory_files(path: str, pattern: str) -> list[Path]:
    source = Path(path)
    files = [source] if source.is_file() else sorted(source.glob(pattern))
    files = [
        file for file in files if file.is_file() and file.name != "token-lengths.json"
    ]
    if not files:
        raise ValueError(
            f"No conversation files found at {path} with pattern {pattern}"
        )
    return files


def load_trajectories(config: Any, tokenizer: Any) -> Any:
    from datasets import Dataset

    from areal.utils import logging

    logger = logging.getLogger("Dataset")
    if config.type != "sft" or not config.path:
        raise ValueError("Set dataset type=sft and path to a JSON file or directory")
    options = dict(config.dataset_kwargs)
    pattern = options.pop("file_pattern", "*__primary-seg-*.json")
    include_reasoning = options.pop("include_reasoning", True)
    if not isinstance(include_reasoning, bool):
        raise ValueError("include_reasoning must be a YAML boolean")
    if options:
        raise ValueError(f"Unknown trajectory dataset options: {sorted(options)}")
    if config.max_length is not None and config.max_length <= 0:
        raise ValueError("max_length must be positive or null")

    # Include file metadata in the dataset cache key so updated files are reread.
    sources = [
        (str(path), path.stat().st_size, path.stat().st_mtime_ns)
        for path in trajectory_files(config.path, pattern)
    ]

    def rows(sources):
        for filename, _, _ in sources:
            path = Path(filename)
            try:
                record = read_trajectory(path, include_reasoning)
                row = tokenize_trajectory(record, tokenizer, config.max_length)
            except (ValueError, TypeError, KeyError) as error:
                raise ValueError(f"{path}: {error}") from error
            logger.info(
                "%s: messages=%d, context_tokens=%d, supervised_tokens=%d",
                path.name,
                len(record["messages"]),
                len(row["input_ids"]),
                sum(row["loss_mask"]),
            )
            yield row

    return Dataset.from_generator(rows, gen_kwargs={"sources": sources})


def main(args: list[str]) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--inspect-only", action="store_true")
    options, config_args = parser.parse_known_args(args)

    from areal.api.cli_args import SFTConfig, load_expr_config
    from areal.utils import logging
    from areal.utils.hf_utils import load_hf_tokenizer

    config, _ = load_expr_config(config_args, SFTConfig)
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    train_dataset = load_trajectories(config.train_dataset, tokenizer)
    valid_dataset = (
        load_trajectories(config.valid_dataset, tokenizer)
        if config.valid_dataset is not None
        else None
    )
    logger = logging.getLogger("Dataset")
    for name, dataset in (("train", train_dataset), ("validation", valid_dataset)):
        if dataset is not None:
            lengths = [len(row["input_ids"]) for row in dataset]
            logger.info(
                "%s: examples=%d, min_tokens=%d, max_tokens=%d, total_tokens=%d",
                name,
                len(lengths),
                min(lengths),
                max(lengths),
                sum(lengths),
            )
    if options.inspect_only:
        return
    if config.train_dataset.max_length is None:
        raise ValueError("Set train_dataset.max_length explicitly before training")
    if len(train_dataset) < config.train_dataset.batch_size:
        raise ValueError("Training dataset has fewer conversations than batch_size")

    from areal import SFTTrainer

    with SFTTrainer(
        config, train_dataset=train_dataset, valid_dataset=valid_dataset
    ) as trainer:
        trainer.train()


if __name__ == "__main__":
    main(sys.argv[1:])
