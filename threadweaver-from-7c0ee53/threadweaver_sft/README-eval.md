## Evaluation (pause-limit sweep)

Use `eval_pause_sweep.sh` to evaluate one checkpoint across multiple
`num_tokens_in_the_longest_thread` pause limits and generate an aggregated report.

```bash
cd threadweaver/threadweaver_sft

./eval_pause_sweep.sh /path/to/checkpoint \
  --data-type /path/to/dataset (Point to the parquet file or directory) \
  -n 32 \
  --bfloat16 \
  --verbose 2
```

Please download the evaluation dataset from [here](https://huggingface.co/datasets/parallel-reasoner/Test/tree/main) (There is a file called aime24-test.parquet)and provide the path via `--data-type` (or `-d`).

### Required input

- `<checkpoint_path>` (first positional arg): checkpoint/model path to evaluate.
- `--data-type` / `-d`: parquet path (or dataset key). This is required and must be passed via CLI args.

### Optional controls

- `-n`, `--n-samples` (default `1`)
- `--bfloat16`
- `--verbose <level>`
- default values in the wrapper:
  - `max-context-length=40960`
  - `template-type=model`
- any extra terminal args are forwarded to `src/simple_eval_pause.py` (and can override defaults)

The sweep runs these pause limits:
`0, 2048, 4096, 8192, 16384, 24576, 32768, 40960`.

### Outputs

Outputs are written under `<checkpoint_dir>/eval_pause_outputs/`, including:

- per-run metrics JSONs: `eval_pause_outputs/<run_name>/*_metrics.json`
- per-run markdown report: `eval_pause_outputs/<run_name>/*_report.md`
- aggregated CSV: `eval_pause_outputs/pause_sweep_report.csv`
- aggregated markdown report: `eval_pause_outputs/pause_sweep_report.md`
