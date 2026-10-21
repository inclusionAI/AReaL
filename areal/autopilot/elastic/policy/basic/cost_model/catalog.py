# SPDX-License-Identifier: Apache-2.0

"""Bundled public facts and screenshot trial templates, with explicit provenance."""

import json
from dataclasses import fields
from pathlib import Path

from .config import ActorConfig, Hardware, ModelShape

DATA = Path(__file__).with_name("data")


def read_data(name: str) -> dict:
    with (DATA / name).open() as handle:
        return json.load(handle)


def qwen3_30b_a3b() -> ModelShape:
    cfg = read_data("qwen3_30b_a3b.json")["config"]
    return ModelShape(
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["moe_intermediate_size"],
        layers=cfg["num_hidden_layers"],
        attention_heads=cfg["num_attention_heads"],
        kv_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"],
        experts=cfg["num_experts"],
        top_k=cfg["num_experts_per_tok"],
        vocab_size=cfg["vocab_size"],
        tied_embeddings=cfg["tie_word_embeddings"],
    )


def hardware_profile(name: str) -> Hardware:
    data = read_data("hardware.json")["profiles"]
    if name not in data:
        raise ValueError(
            f"Unknown hardware profile {name}; choose {list(data)} or supply custom Hardware"
        )
    allowed = {f.name for f in fields(Hardware)}
    return Hardware(**{k: v for k, v in data[name].items() if k in allowed})


def trial_config(trial_id: str) -> ActorConfig:
    """Only screenshot fields; actual PPO/logprob/checkpoint options must be supplied."""
    data = read_data("trials.json")
    trial = next((row for row in data["trials"] if row["id"] == trial_id), None)
    if trial is None:
        raise ValueError(f"Unknown trial {trial_id}")
    return ActorConfig.from_areal(
        {"backend": trial["allocation"], **data["common_actor"]}
    )
