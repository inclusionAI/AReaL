<!-- Think of this file as the go-to brief for AI coding agents working on AReaL. -->

# AGENTS.md — AReaL Agent Operations Guide

## TL;DR for coding agents

- **Runtime**: Designed for distributed GPU clusters (FSDP/Megatron + NCCL). Assume
  containerized execution; no standalone local runs.
- **Environment**: Platform images and launcher specs live under `launcher/` and
  docs—reference them instead of hand-rolling virtualenvs.
- **Testing**: Integration and performance tests require multi-node hardware. Explain
  skips explicitly when you cannot access the cluster.
- **Tooling**: `.pre-commit-config.yaml` runs Ruff (lint+format), mdformat,
  clang-format, nbstripout, and CLI doc generation; install with `pre-commit install`
  before submitting patches.
- **Formatting**: Ruff + Ruff-format replace Black/isort; autoflake settings remain in
  `pyproject.toml`. Surface any formatting gaps you cannot auto-fix.
- **Docs**: Source lives under `docs/` (Jupyter Book). Coordinate doc edits with the
  docs build pipeline.
- **Legacy code**: `realhf/` is deprecated—do not modify or import from it; migrate uses
  into `areal/` equivalents instead.

When unsure, leave a `TODO(agent)` comment and note the constraint in your response.

## Repository map

| Path                      | Purpose                                                                         |
| ------------------------- | ------------------------------------------------------------------------------- |
| `areal/api/`              | Core contracts: workflows, engines, controllers, schedulers, IO structs.        |
| `areal/controller/`       | Distributed batching utilities and controller-side dataset packing.             |
| `areal/core/`             | Async orchestration primitives (task runners, remote inference, workflow exec). |
| `areal/dataset/`          | Dataset loaders & utilities that feed rollouts.                                 |
| `areal/engine/`           | Training backends (FSDP2, Megatron, PPO actors) and inference adapters.         |
| `areal/experimental/`     | Prototype engines/workflows; expect churn and breaking changes.                 |
| `areal/launcher/`         | Orchestration entrypoints (local, Ray, Slurm) plus container specs.             |
| `areal/models/`           | Model-specific adapters (Megatron-Core, Transformers wrappers).                 |
| `areal/platforms/`        | Hardware/platform abstractions (CPU/GPU/NPU backends, runtime adapters).        |
| `areal/reward/`           | Built-in reward functions plus helpers (math parsing, CLEVR counting).          |
| `areal/scheduler/`        | Scheduler implementations and allocation logic.                                 |
| `areal/tests/`            | Targeted tests; many require GPUs or mocked distributed backends.               |
| `areal/thirdparty/`       | Vendored integrations (e.g., vLLM shims).                                       |
| `areal/utils/`            | Logging (`stats_tracker`), tensor helpers, recovery, evaluation, device utils.  |
| `areal/workflow/`         | Rollout/agent implementations (`multi_turn`, `rlvr`, `vision_rlvr`).            |
| `examples/`               | Runnable entrypoints for math, multi-turn, RLHF, VLM, search agents.            |
| `evaluation/`             | Offline evaluation scripts (math/code/Elo) and utilities.                       |
| `functioncall/`           | Tool-calling utilities reused in workflows.                                     |
| `docs/`                   | Jupyter Book source published to https://inclusionai.github.io/AReaL/.          |
| `assets/`                 | Figures and other static assets.                                                |
| `benchmark/`              | Regression baselines and benchmark snapshots.                                   |
| `blog/`                   | Release and update write-ups.                                                   |
| `csrc/`                   | CUDA/C++ extensions that need `build_ext --inplace` after edits.                |
| `notebook/`               | Reference notebooks (outputs stripped by pre-commit).                           |
| `patch/`                  | Local patches for third-party deps (e.g., SGLang fixes).                        |
| `recipe/`                 | Deployment recipes and higher-level orchestration configs.                      |
| `.pre-commit-config.yaml` | Hooks: Ruff lint/format, mdformat, clang-format, nbstripout, CLI docs.          |
| `Dockerfile`              | Container recipe for the standard runtime environment.                          |
| `realhf/`                 | Legacy integrations (read-only, do **not** modify or import).                   |

### Where to find things

- **`areal/api/`** – Contracts for engines, schedulers, dataloaders, and CLI configs.
  Start here when adding new dataclasses or API surfaces.
- **`areal/controller/`** – Distributed batching utilities and controller-side dataset
  packing.
- **`areal/core/`** – Async orchestration primitives (task runners, remote inference,
  workflow execution).
- **`areal/dataset/`** – Stateful data pipeline utilities. New datasets should plug into
  these loaders for replay-safe iteration.
- **`areal/engine/`** – Training and inference engines: FSDP2, Megatron, PPO actors, and
  SGLang/vLLM adapters. Keep weight versioning logic consistent across edits.
- **`areal/experimental/`** – Prototype engines/workflows; expect churn and breaking
  changes.
- **`areal/launcher/`** – Reference launchers for local, Ray, and Slurm targets plus
  container specs; reuse these instead of ad-hoc scripts.
- **`areal/models/`** – Model-specific adapters (Megatron-Core layers, Transformers
  wrappers, custom heads).
- **`areal/platforms/`** – Hardware/platform abstractions for CPU/GPU/NPU targets and
  runtime adapters.
- **`areal/reward/`** – Reward functions and math parsers. Wrap slow logic with
  `AsyncRewardWrapper` in `areal/api/reward_api.py`.
- **`areal/scheduler/`** – Placement and allocation policies for launchers; align with
  `examples/**` configs.
- **`areal/tests/`** – Unit and integration tests colocated with code; many require GPU
  or mocked distributed backends.
- **`areal/utils/`** – Cross-cutting helpers (logging, stats, tensor containers,
  recovery, evaluation). Prefer reusing these utilities over duplicating logic.
- **`areal/workflow/`** – Concrete rollout agents (`multi_turn`, `rlvr`, `vision_rlvr`).
  Each illustrates how `RolloutWorkflow.arun_episode` should orchestrate inference and
  rewards.
- **`docs/`** – Jupyter Book source; mirrors the high-level architecture and
  customization guides published at https://inclusionai.github.io/AReaL/.
- **`evaluation/`** – Offline scoring pipelines that consume logged trajectories.
- **`examples/`** – End-to-end wiring scripts for math, multi-turn, RLHF, VLM, and
  search agents. Use them as references for config wiring and launcher usage.
- **`functioncall/`** – Tool-calling scaffolding reused by workflows.
- **`patch/`** – Maintains in-tree diffs applied to upstream dependencies; keep changes
  minimal and well-documented.
- **`realhf/`** – Legacy integrations retained for reference. Do **not** modify or
  import; port call sites into `areal/` instead.

## Distributed operations & tooling

- **Clusters & containers**: Launch configurations live under `areal/launcher/` (local,
  Ray, Slurm). Each entrypoint documents the scheduler expectations; reuse those specs
  instead of inventing ad-hoc run scripts.
- **Shared images**: Platform-specific container images and startup scripts are defined
  alongside launcher configs. Reference them or note when they are missing—do not
  attempt to rebuild CUDA/driver stacks inline.
- **Secrets & endpoints**: Credentials for remote inference (SGLang, vLLM, Redis, etc.)
  are managed outside the repo. Flag their absence rather than hardcoding replacements.
- **Testing limitations**: End-to-end tests (FSDP, Megatron, distributed RPC) require
  multi-node NCCL clusters. If you cannot execute them, state that your validation is
  limited to static analysis/doc updates.
- **Formatting & docs**: Pre-commit runs Ruff (lint+format), mdformat, clang-format,
  nbstripout, and CLI doc generation. Run `pre-commit run --all-files` (or install the
  hook) before submitting; keep doc edits aligned with the Jupyter Book structure in
  `docs/`.

## Legacy `realhf/` (read-only)

- `realhf/` remains only for archival context. The package build explicitly excludes it
  via `pyproject.toml`.
- Do **not** modify files under `realhf/`, and avoid importing them in new code. Treat
  any dependency on these modules as tech debt.
- When you encounter a `realhf` call site, prefer migrating the logic to the matching
  `areal/` module or partner with maintainers to port it.
- Flag lingering `realhf` usage in reviews/issues so we can track and eliminate it.

### Code style & patterns

- **Typing & dataclasses**: Prefer explicit type hints and reuse existing dataclasses in
  `areal/api/cli_args.py` when extending configs. When adding new configuration options,
  extend an existing dataclass if your changes are backward-compatible or the new config
  is a strict superset of an existing one. Create a new dataclass if the config is
  conceptually distinct or would introduce breaking changes. Keep new configs
  dataclass-based so Hydra/CLI integration stays consistent.
- **Imports**: Avoid wildcard imports; keep third-party vs internal groups consistent.
  Ruff enforces import ordering (isort rules) when hooks run. Place heavy optional deps
  inside functions to prevent import-time side effects.
- **Logging**: Use `areal.utils.logging.getLogger(__name__)` rather than `print`. Emit
  structured metrics through `stats_tracker`/`StatsLogger` instead of ad-hoc counters.
- **Async code**: Rollout workflows must stay non-blocking—prefer `await` with
  `aiofiles`, avoid synchronous file I/O inside `arun_episode`, and guard long-running
  CPU work with executors if needed.
- **Tensor shapes**: Follow padded batch conventions; validate with
  `check_trajectory_format` while developing. Use helpers in `areal.utils.data` for
  padding/broadcasting.
- **Config overrides**: Keep Hydra-friendly dotted names; don’t hardcode paths—expose
  options in config dataclasses and wire via YAML.
- **Testing**: New features should ship with targeted pytest coverage (mark GPU-heavy
  suites appropriately). Use `pytest.skip` with a clear reason when hardware isn’t
  guaranteed.
- **Docs & comments**: Document non-obvious behaviors inline; prefer short module-level
  docstrings summarizing workflows or engines you touch.

## Core concepts & extension points

1. **Rollout workflows (`areal/api/workflow_api.py`)** – Implement
   `RolloutWorkflow.arun_episode`. Use helpers like `concat_padded_tensors` and respect
   shape `[batch, seq_len, …]`.
1. **Inference engines (`areal/engine/sglang_remote.py`,
   `areal/engine/vllm_remote.py`)** – Handle async generation and weight updates.
   Interact with workflows via `InferenceEngine.agenerate`.
1. **Training engines (`areal/engine/ppo/actor.py`, `areal/engine/fsdp_engine.py`)** –
   Consume rollout tensors, run PPO/GRPO updates, broadcast weight versions.
1. **Rewards (`areal/api/reward_api.py`, `areal/reward/`)** – Wrap blocking reward code
   in `AsyncRewardWrapper`. Standard signature:
   `(prompt, completions, prompt_ids, completion_ids, **data)`.
1. **Configurations** – Dataclasses in `areal/api/cli_args.py`, YAML examples in
   `examples/**`. Launchers parse CLI overrides (Hydra-style dotted keys).

Reference docs:

- Agents customization guide: `docs/customization/agent.md`.
- Lite design doc: `areal/README.md`.
- Algorithm-specific docs: `docs/algorithms/*.md`.

## Common tasks for agents

### Add or adjust a rollout workflow

1. Create/modify a class in `areal/workflow/` that subclasses `RolloutWorkflow`.
1. Maintain async behavior (`async def arun_episode`); gather trajectories per prompt
   and return padded tensors (typically via `concat_padded_tensors`).
1. Expose knobs via `__init__` (tokenizer, `GenerationHyperparameters`, reward fn,
   dump_dir).
1. Update references in entry scripts or configs (e.g.,
   `examples/multi-turn-math/train.py`).

### Introduce a reward function

1. Implement reward logic in `areal/reward/<name>.py`.
1. Register via `areal/reward/__init__.py` or supply a fully qualified import path in
   configs.
1. For slow reward evaluation, wrap with `AsyncRewardWrapper` to avoid blocking rollout
   loops.

### Wire a new dataset

1. Place dataset loader under `areal/dataset/` (follow existing patterns using
   `StatefulDataLoader`).
1. Update config fields (`train_dataset.*`) to point to the new loader and schema.
1. Ensure prompts provide the keys expected by your workflow (`messages`, `answer`,
   etc.).

### Launch training / evaluation

- Follow the launcher examples in `examples/**` together with the matching scripts in
  `launcher/`. Each example README points to the expected scheduler (local, Ray, or
  Slurm) and container image.
- Always keep rollout/inference versioning in sync via `WeightUpdateMeta` (see
  `examples/multi-turn-math/train.py`). Document any skipped launcher steps if you
  cannot access the target cluster.

### Publish docs

- The docs site builds via Jupyter Book pipelines defined in `docs/`. Coordinate with
  maintainers before triggering a build and note in responses when doc rebuilds are
  deferred due to environment constraints.

## Testing & validation strategy

- **Unit tests**: Suites under `areal/tests/` frequently rely on GPUs or mocked
  distributed backends. If you cannot execute them, state which files would normally be
  run (e.g., `test_utils.py`, `test_allocation_mode.py`).
- **Workflow smoke tests**: `areal/tests/grpo` covers rollout logic but requires GPUs;
  acknowledge skipped coverage explicitly.
- **Distributed/FSDP suites**: `test_fsdp_*`, `test_sglang_engine.py`, and RPC suites
  demand multi-node NCCL clusters. Mention the dependency when deferring.
- **Static checks**: Pre-commit enforces Ruff lint/format, mdformat, clang-format,
  nbstripout, CLI doc regeneration, and autoflake. Call out when hooks cannot be run
  locally.

Always mention resource requirements in PRs and in agent responses when tests are
skipped.

## Observability & ops

- `areal/utils/stats_tracker.py` collects metrics; `StatsLogger` streams to W&B/SwanLab.
- `areal/utils/saver.py` + `areal/utils/recover.py` handle checkpoints/resume.
- Rollout workflows can persist transcripts by passing `dump_dir`; outputs are organized
  by weight version.
- `evaluation/` hosts offline evaluation scripts (math, code, Elo). They expect
  generated logs in standard formats—consult `evaluation/README.md`.

## Known constraints & best practices

- Large downloads: models/datasets fetched via Hugging Face; ensure cache directories
  point to shared storage in multi-node runs.
  - `HF_HOME` sets the root directory for all Hugging Face cache data (models, datasets,
    etc.).
  - `TRANSFORMERS_CACHE` (if set) overrides the model cache location only, and takes
    precedence over `HF_HOME` for model files.
  - Prefer setting `HF_HOME` for a unified cache; use `TRANSFORMERS_CACHE` only if you
    need to separate model files from other cache data.
- Async training relies on weight versioning—never mutate versions manually; call
  `set_version`/`update_weights` like the examples.
- Avoid blocking operations inside workflows; perform heavy I/O via `aiofiles` or
  background tasks.
- Respect the distributed launchers. For new scripts, prefer using existing launch
  utilities over bespoke `torchrun` commands.
- When editing CUDA/C++ extensions, run `pip install -e .` again or
  `python setup.py build_ext --inplace`.

## Collaboration conventions

- **Issue first**: Tie every PR to a filed issue using `Fixes #123` or `Refs #123` in
  the description. Follow the GitHub issue templates under `.github/ISSUE_TEMPLATE/`.
- **Branch naming**: Use kebab-case summaries, e.g., `feature/multi-turn-metrics` or
  `bugfix/fsdp-weight-sync`.
- **Commit messages**:
  - Prefer [Conventional Commit](https://www.conventionalcommits.org/en/v1.0.0/)
    prefixes (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`).
  - Keep the subject ≤72 characters; use imperative mood (“fix rollout metric”) and add
    a body explaining *why* when the change is non-trivial.
  - Squash noisy WIP commits before pushing; keep history clean for bisects.
- **PR titles**: Mirror the main change using the same prefix style, e.g.,
  `feat: add discounted reward stats tracker hook`.
- **PR checklist**:
  - Summarize the change, highlight risks (e.g., breaking changes, performance impacts,
    compatibility issues), list test commands run (or clearly state why tests are
    skipped).
  - Link related documentation updates; mention resource requirements for GPU-bound
    tests.
  - Add screenshots or log snippets when touching user-facing outputs.
- **Reviews**: Be explicit about follow-ups with `TODO(agent)` comments and track them
  in the PR discussion. Address review feedback with additional commits (no force-push
  once review starts unless requested).
- **Pre-merge**: Ensure pre-commit hooks pass (Ruff lint+format, mdformat, clang-format,
  nbstripout, CLI docs, autoflake). For doc-only changes, run `mdformat --check` on
  touched files.

## Reviewer checklist

- **Scope & requirements**: Confirm the PR maps to a filed issue and the description
  lists clear acceptance criteria. Ensure major behavior changes are covered by tests or
  documented.
- **Testing evidence**: Look for explicit test commands (unit, integration, or docs
  builds). Verify GPU-heavy suites are marked/skipped appropriately with rationale.
- **Asynchrony & concurrency**: For workflow or engine edits, check that async functions
  await all I/O, avoid blocking calls, and keep weight versioning (`set_version`,
  `update_weights`) consistent.
- **Resource awareness**: Ensure configs note memory/GPU expectations, and new
  datasets/models document storage paths or cache requirements.
- **Code style compliance**: Watch for Ruff lint/format alignment, autoflake cleanup,
  clang-format on CUDA/C++, mdformat for docs, logging via `areal.utils.logging`, and
  consistent type hints/dataclass usage.
- **Config & docs**: Validate new knobs land in the right dataclasses/YAMLs with
  defaults explained in docs or README snippets. Cross-check hyperlinks and CLI
  references.
- **Observability**: Confirm metrics integrate with `stats_tracker`/`StatsLogger`, and
  long-running workflows expose dump directories or debugging hooks when warranted.
- **Cleanup & debt**: Reject lingering debug prints, commented code, or unexplained
  `TODO`s (except tagged `TODO(agent)` with context). Ensure migrations include
  recovery/evaluator updates if they impact checkpoints.

## Reference material

- Docs portal: https://inclusionai.github.io/AReaL/
- Quickstart tutorial: `docs/tutorial/quickstart.md`
- Customization guides: `docs/customization/`
- Best practices: `docs/best_practices/` (debugging, OOM handling, rollout tips)
- Release notes: `docs/version_history.md`

Create an issue or discussion if you hit unclear architecture boundaries—this repo
evolves quickly.
