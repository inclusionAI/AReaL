# SPDX-License-Identifier: Apache-2.0

"""CPU preflight, reduced checkpoint preparation, and paired logp comparison.

Run directly from a checkout; configuration audit does not import AReaL/CUDA.
"""

import argparse
import importlib.util
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path


def config_module():
    path = (
        Path(__file__).resolve().parents[2] / "areal/models/mcore/glm5_next_config.py"
    )
    spec = importlib.util.spec_from_file_location("_glm5_preflight_config", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def audit(source: Path, check_runtime: bool = False) -> dict:
    raw = json.loads((source / "config.json").read_text())
    config = config_module().Glm5NextConfig.from_hf(raw)
    result = dict(
        layers=len(config.layer_types),
        kda_layers=[
            i for i, kind in enumerate(config.layer_types) if kind == "linear_attention"
        ],
        dsa_layers=[
            i
            for i, kind in enumerate(config.layer_types)
            if kind == "deepseek_sparse_attention"
        ],
        moe_layers=sum(config.moe_layers),
        residual_streams=config.text["hc_mult"],
        checkpoint_quantization=(raw.get("quantization_config") or {}).get(
            "quant_method"
        ),
        actor_compute="bfloat16",
        context_parallel_size=1,
        mtp=False,
    )
    if check_runtime:
        import torch
        from mbridge.core.parallel_states import ParallelStates
        from megatron.core.transformer import MLATransformerConfig
        from transformers import PretrainedConfig

        import areal.engine.megatron_engine  # noqa: F401
        from areal.models.mcore.glm5_next import check_glm5_next_capabilities
        from areal.models.mcore.glm5_next_bridge import Glm5NextBridge

        check_glm5_next_capabilities(MLATransformerConfig)
        import tilelang  # noqa: F401
        import triton  # noqa: F401
        from fla.modules import FusedRMSNormGated, ShortConvolution  # noqa: F401
        from fla.ops.kda import chunk_kda  # noqa: F401
        from fla.ops.kda.gate import fused_kda_gate  # noqa: F401

        bridge = Glm5NextBridge(
            PretrainedConfig.from_dict(raw),
            dtype=torch.bfloat16,
            parallel_states=ParallelStates(vpp_size=None),
        )
        result["runtime_bridge_layers"] = bridge.config.num_layers
        result["runtime_imports"] = "passed (GPU kernels not executed)"
    return result


def prepare_smoke(source: Path, output: Path, layers: int = 4) -> dict:
    """Copy a real leading-layer checkpoint; never point an index at extra layers."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    raw = json.loads((source / "config.json").read_text())
    config = config_module().Glm5NextConfig.from_hf(raw)
    if not 4 <= layers <= len(config.layer_types):
        raise ValueError(
            "Smoke checkpoint must include at least four layers (KDA + DSA + MoE)"
        )
    if output.exists():
        raise ValueError("Smoke output must be a new directory")
    source_index = json.loads((source / "model.safetensors.index.json").read_text())
    grouped = defaultdict(list)
    for name, shard in source_index["weight_map"].items():
        match = re.match(r"model\.language_model\.layers\.(\d+)\.", name)
        if (match and int(match[1]) < layers) or name.startswith(
            (
                "model.language_model.embed_tokens.",
                "model.language_model.norm.",
                "model.visual.",
                "lm_head.",
            )
        ):
            grouped[shard].append(name)
    if not grouped:
        raise ValueError("Source index has no GLM-5.3 weights")
    output.mkdir(parents=True)
    weight_map, total_size = {}, 0
    for i, (shard, names) in enumerate(sorted(grouped.items())):
        with safe_open(source / shard, framework="pt", device="cpu") as handle:
            tensors = {name: handle.get_tensor(name) for name in names}
        output_shard = f"model-{i + 1:05d}-of-{len(grouped):05d}.safetensors"
        save_file(tensors, output / output_shard, metadata={"format": "pt"})
        total_size += sum(t.numel() * t.element_size() for t in tensors.values())
        weight_map.update({name: output_shard for name in names})
        del tensors
    text = raw["text_config"]
    text["num_hidden_layers"] = layers
    for key in ("layer_types", "mlp_layer_types", "indexer_types"):
        if key in text:
            text[key] = text[key][:layers]
    for key in ("kda_layers", "full_attn_layers"):
        if key in text.get("linear_attn_config", {}):
            text["linear_attn_config"][key] = [
                i for i in text["linear_attn_config"][key] if i < layers
            ]
    for node in (raw, text):
        for key in ("num_nextn_predict_layers", "mtp_num_hidden_layers"):
            if key in node:
                node[key] = 0
    if raw.get("quantization_config"):
        raw["quantization_config"] = config_module().glm5_next_quantization_config(
            raw["quantization_config"]
        )
    (output / "config.json").write_text(json.dumps(raw, indent=2) + "\n")
    (output / "model.safetensors.index.json").write_text(
        json.dumps(
            dict(metadata=dict(total_size=total_size), weight_map=weight_map), indent=2
        )
        + "\n"
    )
    for pattern in (
        "tokenizer*",
        "chat_template*",
        "special_tokens_map.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "*.py",
    ):
        for asset in source.glob(pattern):
            if asset.is_file():
                shutil.copy2(asset, output / asset.name)
    return dict(
        layers=layers,
        tensors=len(weight_map),
        total_size=total_size,
        output=str(output),
    )


def compare_logps(reference: Path, candidate: Path, atol: float) -> dict:
    """Compare NPZ captures only when token IDs and scoring masks match exactly."""
    import numpy as np

    with (
        np.load(reference, allow_pickle=False) as a,
        np.load(candidate, allow_pickle=False) as b,
    ):
        for key in ("token_ids", "loss_mask"):
            if not np.array_equal(a[key], b[key]):
                raise ValueError(f"Cannot compare unpaired {key}")
        if (
            a["logprobs"].shape != b["logprobs"].shape
            or a["logprobs"].shape != a["loss_mask"].shape
        ):
            raise ValueError("logprobs and loss_mask shapes must agree")
        mask = a["loss_mask"].astype(bool)
        delta = np.abs(
            a["logprobs"][mask].astype(np.float64)
            - b["logprobs"][mask].astype(np.float64)
        )
    if delta.size == 0 or not np.isfinite(delta).all():
        raise ValueError(
            "Comparison requires finite logprobs on at least one scored token"
        )
    return dict(
        tokens=int(delta.size),
        mean_abs=float(delta.mean()),
        p99_abs=float(np.quantile(delta, 0.99)),
        max_abs=float(delta.max()),
        atol=atol,
        passed=bool(delta.max() <= atol),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("audit")
    preflight.add_argument("source", type=Path)
    preflight.add_argument("--check-runtime", action="store_true")
    smoke = commands.add_parser("prepare-smoke")
    smoke.add_argument("source", type=Path)
    smoke.add_argument("output", type=Path)
    smoke.add_argument("--layers", type=int, default=4)
    compare = commands.add_parser("compare-logps")
    compare.add_argument("reference", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--atol", type=float, required=True)
    args = parser.parse_args()
    if args.command == "audit":
        result = audit(args.source, args.check_runtime)
    elif args.command == "prepare-smoke":
        result = prepare_smoke(args.source, args.output, args.layers)
    else:
        if args.atol < 0:
            parser.error("--atol must be nonnegative")
        result = compare_logps(args.reference, args.candidate, args.atol)
    print(json.dumps(result, indent=2))
    return 0 if result.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
