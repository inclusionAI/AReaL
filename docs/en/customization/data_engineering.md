# Data engineering recipes

`areal.v2.data_engineering` runs bounded foreground data jobs. It provides recipe
execution, public-task calibration, deduplication, provenance, and verified dataset
snapshots. A research controller chooses the objective and recipe; the recipe supplies
domain-specific generation, cleaning, scoring, selection, and training projection.

## Run the TIR example

From the repository root with AReaL's normal dependencies installed:

```bash
python3 -m areal.v2.data_engineering \
  --config examples/data_engineering/tir/data_generation.yaml \
  --output /tmp/tir-data-job \
  --export-hf /tmp/tir-training-data
```

Both output directories must be new. This example generates exact-answer math tasks
without a model, GPU, or inference service. AReaL's package imports still require its
normal Python dependencies. The final stdout line contains `dataset_id`, `dataset_path`,
and, when requested, `training_dataset_path`.

The TIR recipe has eight families and two structures per family, adapted from the
AReaL-RSI TIR generator at commit `97226ea54e9e009484f18c9c28e770697e9f6449`. Its
training view contains `task_id`, `messages`, and an exact boxed `answer`. It produces
task data; training rollouts, tokenization, and RL/SFT/OPD tensors are created
downstream.

## Python API and execution protocol

```python
import asyncio
from areal.v2.data_engineering import DataJobConfig, run_data_job

config = DataJobConfig(
    recipe="examples.data_engineering.tir.data_recipe:create_recipe",
    target_count=64,
    purpose="joint",
    seed=42,
    params={"candidate_count": 128},
)
snapshot = asyncio.run(run_data_job(config, "/tmp/tir-python-job"))
rows = snapshot.training_rows()
```

Inside an existing async controller, use `await run_data_job(...)`. A factory is an
importable `module:callable`, invoked as `factory(**params)`, returning a `DataRecipe`.
The public Python protocols are exported from `areal.v2.data_engineering`:

| Type or method                     | Contract                                                                                                                                   |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `PublicTask`                       | Stable `task_id`, domain `input`, and public `metadata`; never put an oracle in these public fields                                        |
| `DataSample`                       | `PublicTask` plus a private `reference`; retained samples must serialize to JSON                                                           |
| `generate(context)`                | Yield a bounded batch of samples; the context contains seed, batch index, target, purpose, and input file paths                            |
| `clean(sample)`                    | Return a cleaned sample or `None` to reject it                                                                                             |
| `score(sample, response)`          | Use the private reference to return a finite score in `[0, 1]`                                                                             |
| `select(samples, scores, context)` | Return `Selection` containing exactly `target_count` distinct existing task IDs, or raise `InsufficientDataError` to request another batch |
| `training_row(sample)`             | Project one selected task to one domain-specific JSON training row                                                                         |
| `source_files()`                   | Optional recipe/calibrator method declaring helper code and external files that affect the result                                          |

The executor runs generate → clean → deduplicate/exclude → calibrate → select. Cleaned
candidates and their scores accumulate across batches. Only `InsufficientDataError`
retries selection; other errors fail the job. Generic `max_batches` and `max_candidates`
bound the search; domain parameters stay in `params`. A recipe may read raw data through
`context.inputs` instead of synthesizing it. Inputs and exclusions are files, resolved
relative to the config file by the module entry point. Paths inside plugin `params`
remain plugin-owned; the supplied calibration adapters resolve them against the process
working directory.

`purpose` is `harness`, `model`, or `joint`. It is recorded and passed to the recipe;
recipes decide how it affects generation and selection. It does not automatically infer
whether a failure came from model capability or Harness behavior. Independent dev/test
sets and the decision to adopt a Harness/model remain with the caller.

Identical task IDs must identify identical cleaned content. Duplicate public inputs with
conflicting references fail the job. Exact JSON input duplicates are removed; semantic
or domain-specific normalization belongs in `clean()`. Exclusion JSONL files need
top-level `task_id` fields; use prior snapshots' `public.jsonl` or a previous job's
`work/batch-N-public.jsonl` to exclude previously selected or probed tasks.

## Calibration protocol

A calibration factory returns an async `Calibrator`:

```python
async def __call__(self, tasks, context):
    # tasks: Sequence[PublicTask]; context: CalibrationContext
    # Process public tasks and report implementation-specific provenance.
    return CalibrationBatch(tuple(attempts), provenance={"method": "my-calibrator-v1"})
```

An `Attempt` has `task_id`, zero-based `repeat`, JSON-serializable `response`, and
optional `failure_reason`. Every requested task/repeat pair must be returned exactly
once. The generic protocol requires no model or Harness binding. A nonempty failure
reason produces a `None` score; an incorrect completed response receives the recipe's
ordinary score. Backend mutation of public request objects does not mutate retained
candidates.

The calibration backend receives no `DataSample.reference`. The executor grades raw
responses with the recipe's private reference. This is a data contract, not a security
sandbox: recipe and calibrator code is trusted. Implementation-specific dependencies and
version checks belong in the calibrator. Optional JSON provenance is preserved per
batch; for example a rule version, source revision, or evaluated model/Harness identity.
The TIR adapter validates the latter inside its own implementation.

For TIR, use two attempts per task. Its numeric verifier requires a final `\boxed{...}`
and supports integers, decimal values, rational strings, and numeric LaTeX fractions.
Expressions such as `1+2` are not evaluated. Selection drops incomplete and
twice-correct tasks, prioritizes once-correct tasks, and then uses twice-wrong tasks. It
enforces family coverage and the configured recipe cap. Without calibration, it selects
generated tasks without claiming model difficulty calibration.

### Reuse existing TIR evaluation reports

```yaml
recipe: examples.data_engineering.tir.data_recipe:create_recipe
purpose: joint
seed: 42
target_count: 512
max_batches: 8
max_candidates: 8192
params:
  candidate_count: 1024
  minimum_one_pass: 128
  maximum_per_recipe: 48
calibration:
  implementation: examples.data_engineering.tir.data_calibration:recorded_responses
  repeats: 2
  params:
    model_ref: /path/to/the/evaluated/checkpoint
    harness_ref: actual-harness-id
    reports:
      - [/path/to/batch0-repeat0.json, /path/to/batch0-repeat1.json]
    backend_model_digest: actual-model-digest
    verifier_digest: actual-verifier-digest
```

Supply reports for every batch that may be needed. Reports must cover the generated task
IDs and match the requested model, Harness, model digest, and verifier digest. Repeats
must have distinct trial IDs and the same task set/executable identity. Stored scores
are ignored and responses are graded again. Report and source digests enter the snapshot
provenance. Reuse requires matching the original generator seed and candidate count;
this adapter does not run an evaluation or start SGLang.

## Artifacts and training integration

`job.json` records running/succeeded/failed/cancelled state. `config.json`,
`progress.json`, and per-batch files under `work/` retain execution evidence.
Publication occurs only after successful selection and validation, by renaming a staging
directory on the same filesystem. Existing output directories are rejected. Jobs run in
the foreground; interrupting the module cancels it. There is no background daemon or
resume protocol.

The dataset directory contains:

- `public.jsonl`: selected public tasks.
- `references.jsonl`: aligned private references.
- `train.jsonl`: the recipe's training view.
- `probes.jsonl`: raw calibration responses, failure reasons, and recomputed scores.
- `report.json`: generation, cleaning, exclusion, and selection counts.
- `manifest.json`: content identity, file digests, config, recipe/backend code digests,
  declared source digests, and implementation-specific calibration provenance.

Job directories and dataset files are owner-restricted. `load_snapshot(path)` and
`snapshot.training_rows()` verify file digests and row alignment. Snapshots are
immutable by convention and identified by content; permissions do not make them
write-proof. Factories should declare imported helper code and non-config input files
with `source_files()`. Declared inputs are checked again before publication.

Use the explicit materialization adapter for existing training loaders:

```python
from areal.dataset import get_custom_dataset
from areal.v2.data_engineering import export_hf_dataset, load_snapshot

snapshot = load_snapshot("/tmp/tir-data-job/dataset")
path = export_hf_dataset(snapshot, "/tmp/another-tir-training-data")
dataset = get_custom_dataset(path=str(path), type="rl")
```

The export includes `areal-source.json` with the source dataset identity. Choose a
workflow that consumes the recipe's columns. For SFT or OPD, provide the appropriate
projection/tokenization/teacher-processing adapter; TIR's task-only view is not an SFT
conversation or a tokenized RL trajectory.
