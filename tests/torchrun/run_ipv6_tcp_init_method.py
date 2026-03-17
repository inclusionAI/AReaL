import argparse
import importlib.util
import os
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_network = _load_module("areal_utils_network", _ROOT / "areal" / "utils" / "network.py")
_distributed = _load_module(
    "areal_engine_core_distributed", _ROOT / "areal" / "engine" / "core" / "distributed.py"
)

format_host_for_url = _network.format_host_for_url
init_custom_process_group = _distributed.init_custom_process_group


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init-host", type=str, required=True)
    parser.add_argument("--init-port", type=int, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    init_method = f"tcp://{format_host_for_url(args.init_host)}:{args.init_port}"
    pg = init_custom_process_group(
        backend="gloo",
        init_method=init_method,
        world_size=world_size,
        rank=rank,
        group_name="ipv6_e2e",
    )

    x = torch.tensor([rank + 1], dtype=torch.int64)
    torch.distributed.all_reduce(x, group=pg)

    expected = world_size * (world_size + 1) // 2
    ok = int(x.item()) == expected

    if rank == 0:
        with open(args.output, "w") as f:
            f.write("Passed" if ok else f"Failed: got={int(x.item())} expected={expected}")


if __name__ == "__main__":
    main()
