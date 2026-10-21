# SPDX-License-Identifier: Apache-2.0

"""Checkpoint helpers for the text-only GLM actor and its multimodal HF shell."""

import json
import struct
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def copy_glm5_next_vision_weights(source: str | Path, destination: str | Path) -> None:
    """Retain the frozen source vision weights in an exported HF checkpoint.

    Read only vision tensors, one source shard at a time. The actor never builds
    the vision tower, but the exported conditional-generation architecture does.
    Call on the writer rank after all text shards have been saved.
    """
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("GLM-5.3 export must not overwrite its source checkpoint")
    source_index = json.loads((source / "model.safetensors.index.json").read_text())
    output_index_path = destination / "model.safetensors.index.json"
    output_index = json.loads(output_index_path.read_text())
    grouped = defaultdict(list)
    for name, shard in source_index["weight_map"].items():
        if name.startswith("model.visual."):
            grouped[shard].append(name)
    config = json.loads((destination / "config.json").read_text())
    if config.get("vision_config") and not grouped:
        raise ValueError(
            "GLM-5.3 source declares vision_config but has no vision weights"
        )
    for index, (shard, names) in enumerate(sorted(grouped.items())):
        with safe_open(source / shard, framework="pt", device="cpu") as handle:
            tensors = {name: handle.get_tensor(name) for name in names}
        output_shard = f"glm5-vision-{index:05d}.safetensors"
        save_file(tensors, destination / output_shard, metadata={"format": "pt"})
        for name in names:
            output_index["weight_map"][name] = output_shard
        del tensors
    # Recompute from the index, so repeated exports do not double-count vision.
    size = 0
    grouped.clear()
    for name, shard in output_index["weight_map"].items():
        grouped[shard].append(name)
    for shard, names in grouped.items():
        with (destination / shard).open("rb") as handle:
            header_length = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_length))
        for name in names:
            begin, end = header[name]["data_offsets"]
            size += end - begin
    output_index.setdefault("metadata", {})["total_size"] = size
    output_index_path.write_text(json.dumps(output_index, indent=2) + "\n")
    # The actor intentionally omits the next-token predictor.
    for node in (config, config.get("text_config", {})):
        for key in ("num_nextn_predict_layers", "mtp_num_hidden_layers"):
            if key in node:
                node[key] = 0
    if "text_config" in config:
        config["text_config"]["model_type"] = "glm5_next_text"
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
