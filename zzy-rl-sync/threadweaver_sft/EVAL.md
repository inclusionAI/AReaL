# ThreadWeaver SFT Evaluation

This document covers how to evaluate ThreadWeaver SFT checkpoints using the scripts in this directory. For production-grade evaluation with parallel rollouts, see [threadweaver_rl/README.md](../threadweaver_rl/README.md).

## Quick Evaluation (`simple_eval.py`)

A quick evaluation script that runs parallel (branching) generation with SGLang.

```bash
TRAINED_MODEL="ckpts/Q3-8B-131072-SFT"

python src/simple_eval.py \
  --data-type data/mult-10k-par_pq/train.parquet \
  --model_name $TRAINED_MODEL \
  --launch_server \
  --verbose 2 \
  --template-type model \
  --bfloat16 \
  --branching-generate \
  -n 1 \
  --max-context-length 8192
```

Reference result:
```
With strict grading function:
Pass@1: 0.9377 (93.77)
```

## Pause-Limit Evaluation (`eval_pause_sweep.sh`)

Evaluates a checkpoint with parallel (branching) generation, pausing when the longest thread reaches a token limit of **65536**. The script:

1. Launches an SGLang server
2. Runs `src/simple_eval_pause.py` for each configured token limit
3. Aggregates the results into CSV and Markdown reports

### Usage

```bash
./eval_pause_sweep.sh <checkpoint_path> [options] [extra args for simple_eval_pause.py]
```

### Options

| Option | Description |
|---|---|
| `--data-type`, `-d` | Path to a parquet file or dataset key (**required**) |
| `-n`, `--n-samples` | Number of samples to evaluate (default: `1`) |
| `--bfloat16` | Use bfloat16 precision |
| `--verbose <level>` | Verbosity level (e.g., `0`, `1`, `2`) |
| `--help`, `-h` | Show usage and exit |

Any unrecognized arguments are forwarded directly to `simple_eval_pause.py`.

### Example

```bash
./eval_pause_sweep.sh \
  /storage/openpsi/users/zzy/checkpoints/deepscaler/Fix-ckpt-saving-6/hf_ckpt/ckpt-170 \
  --bfloat16 --verbose 2 -n 32 \
  --data-type /storage/openpsi/users/zzy/sync/AIME24_converted_copy.parquet
```

### Data path resolution

The data source is resolved in the following order:

1. `--data-type` / `-d` CLI argument (recommended)
2. `DATA_TYPE` environment variable
3. Interactive terminal prompt if neither is provided

### Environment variable overrides

| Variable | Default | Notes |
|---|---|---|
| `N_SAMPLES` | `1` | Overridden by `-n` / `--n-samples` |
| `MAX_CONTEXT_LENGTH` | `65536` | Maximum context window |
| `TEMPLATE_TYPE` | `model` | Chat template type |
| `VERBOSE_LEVEL` | _(unset)_ | Overridden by `--verbose` |

### Outputs

Results are saved under the checkpoint directory:

| File | Description |
|---|---|
| `eval_pause_outputs/<run>/*_metrics.json` | Per-run metrics (`pass@1`, `pass@n`, `avg_num_tokens_in_the_longest_thread`) |
| `eval_pause_outputs/<run>/*_report.md` | Per-run human-readable report |
| `eval_pause_outputs/pause_sweep_report.csv` | Aggregated CSV summary across all token limits |
| `eval_pause_outputs/pause_sweep_report.md` | Aggregated Markdown summary across all token limits |

After the sweep finishes, a console summary is printed:

```
Pause sweep summary:
 pause_limit      pass@1     pass@32   avg_longest
       65536    0.xxxxxx    0.xxxxxx       xxxxx.xx

Saved CSV report:  .../eval_pause_outputs/pause_sweep_report.csv
Saved Markdown report: .../eval_pause_outputs/pause_sweep_report.md
```
