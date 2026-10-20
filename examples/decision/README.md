# Qwen3.5 PointerHead decision SFT

Train a Qwen3.5 decision model (PointerHead readout + Gated DeltaNet LoRA) with
AReaL on Ascend NPUs.

## Single-NPU PoC

Two epochs, LoRA rank 16, lr `5e-5`, `head_dim=256`. Give the script a local
Qwen3.5 checkpoint and your decision data. If you pass a suite directory it
samples `n_per_source` records so you do not encode the full train split.

Put this checkout first on `PYTHONPATH` if the image also has an `/AReaL`
install. The launch script appends the node IP to `NO_PROXY` so local worker
health checks are not sent through `HTTP_PROXY`.

### Suite directory (sampled)

A suite is a folder with `manifest.json` plus `train.jsonl` /
`development.jsonl`.

```bash
cd /path/to/AReaL   # repo root

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES=0

MODEL_PATH=/path/to/Qwen3.5-4B-Base \
SUITE_PATH=/path/to/decision-suite \
  bash examples/decision/run_npu_poc.sh
```

### Already-sliced JSONL

```bash
MODEL_PATH=/path/to/Qwen3.5-4B-Base \
TRAIN_PATH=/path/to/train.jsonl \
VALID_PATH=/path/to/development.jsonl \
  bash examples/decision/run_npu_poc.sh
```

### Unpacked command

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export ASCEND_RT_VISIBLE_DEVICES=0

python3 examples/decision/prepare_v7_sample.py \
  --suite /path/to/decision-suite \
  --out-dir examples/decision/data \
  --n-per-source 32 --dev-per-source 4

python3 examples/decision/train_decision.py \
  --config examples/decision/qwen3_5_decision_sft_npu_poc.yaml \
  scheduler.type=local \
  cluster.n_gpus_per_node=1 \
  actor.path=/path/to/Qwen3.5-4B-Base \
  train_dataset.path=$PWD/examples/decision/data/poc-train.jsonl \
  valid_dataset.path=$PWD/examples/decision/data/poc-dev.jsonl
```

A successful run finishes two epochs, logs `sft/decision_loss` trending down,
writes LoRA adapters plus `head.pt`, and exits 0. Override `EPOCHS`, `STEPS`,
`N_PER_SOURCE`, or `FILEROOT` on the shell script.

## Pytest e2e

The two-step NPU test skips unless you point it at local assets:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export DECISION_MODEL=/path/to/Qwen3.5-4B-Base
export DECISION_SUITE=/path/to/decision-suite
# or: export DECISION_TRAIN=/path/to/train.jsonl DECISION_VALID=/path/to/development.jsonl

python3 -m pytest tests/test_qwen35_decision.py tests/test_qwen35_decision_e2e.py \
  -q --tb=short
```

`test_qwen35_decision_e2e.py::test_qwen35_decision_sft_single_npu_poc` is marked
`npu` and `slow`. It trains two steps, checks `decision_loss` is logged, and
expects a `head.pt` checkpoint.

## Multi-NPU recipe

`qwen3_5_decision_sft.yaml` is the 8-NPU template (`megatron:d4p1t2`). Point
`actor.path` and `train_dataset.path` at your model and the full suite when you
want the longer schedule.
