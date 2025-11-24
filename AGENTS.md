<!-- Think of this file as the go-to brief for AI coding agents working on AReaL. -->

# AGENTS.md — AReaL Agent Operations Guide

## TL;DR for coding agents

- **Runtime**: Designed for distributed GPU clusters (FSDP/Megatron + distributed
  communication libraries). Assume containerized execution; no standalone local runs.
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
- **Collaboration**: Before editing code, outline the proposed plan and confirm it with
  the user.

When unsure, leave a `TODO(agent)` comment and note the constraint in your response.

## Repository map

- `areal/` — Core Python package housing APIs, controllers, engines, workflows, and
  shared utilities:
  - `areal/api/` — Contracts for workflows, engines, schedulers, IO structs, and
    CLI/config dataclasses.
  - `areal/controller/` — Distributed batching and controller-side dataset packing
    helpers.
  - `areal/core/` — Async orchestration primitives for task runners, remote inference,
    and workflow execution.
  - `areal/dataset/` — Stateful dataset loaders and utilities that feed rollout jobs
    safely.
  - `areal/engine/` — Training/inference backends (FSDP2, Megatron, PPO actors, remote
    adapters).
  - `areal/experimental/` — Prototype engines/workflows that evolve quickly; expect
    breaking changes.
  - `areal/launcher/` — Launch specs for local, Ray, and Slurm clusters plus container
    guidance.
  - `areal/models/` — Model-specific adapters (Megatron-Core layers, Transformers
    wrappers, custom heads).
  - `areal/platforms/` — Hardware/platform abstractions for CPU/GPU/NPU runtimes and
    device adapters.
  - `areal/reward/` — Built-in reward functions, math parsers, and helpers (wrap slow
    logic with AsyncRewardWrapper).
  - `areal/scheduler/` — Placement and allocation policies aligned with launcher
    configs.
  - `areal/tests/` — Focused unit/integration suites (many require GPUs or mocked
    distributed backends).
  - `areal/thirdparty/` — Vendored integrations (e.g., vLLM/SGLang shims) kept in-tree.
  - `areal/tools/` — Developer utilities and maintenance scripts tied to the core
    package.
  - `areal/utils/` — Cross-cutting helpers for logging, tensor ops, stats tracking,
    checkpoints, and recovery.
  - `areal/workflow/` — Concrete rollout agents (`multi_turn`, `rlvr`, `vision_rlvr`)
    implementing `RolloutWorkflow`.
- `assets/` — Figures and other static assets referenced across docs and blogs.
- `benchmark/` — Regression baselines, benchmark snapshots, and reference metrics (e.g.,
  `verl_v0_3_0_post1_*`).
- `blog/` — Release notes and update write-ups documenting project progress.
- `csrc/` — CUDA/C++ extensions (run `build_ext --inplace` or reinstall editable wheels
  after edits).
- `docs/` — Jupyter Book source for https://inclusionai.github.io/AReaL/ plus CLI
  reference generators.
- `evaluation/` — Offline scoring pipelines (math, code, Elo) and shared
  evaluators/utilities.
- `examples/` — End-to-end wiring scripts for math, RLHF, VLM, multi-turn, search
  agents, and launcher recipes.
- `functioncall/` — Tool-calling scaffolding reused by workflows and evaluation
  harnesses.
- `notebook/` — Reference notebooks (outputs stripped via pre-commit) for quick
  experimentation.
- `patch/` — In-tree patches applied to third-party dependencies (e.g., SGLang
  hotfixes).
- `realhf/` — Legacy integrations kept read-only; do **not** modify or import in new
  code.
- `recipe/` — Deployment recipes and higher-level orchestration configs per target
  environment.

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
  multi-node clusters using distributed communication libraries. If you cannot execute
  them, state that your validation is limited to static analysis/doc updates.
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

- Start from the existing patterns in `areal/workflow/multi_turn.py`, `rlvr.py`, or
  `vision_rlvr.py`, then add a sibling module under `areal/workflow/` that subclasses
  `RolloutWorkflow`.
- In `__init__`, thread through `GenerationHyperparameters`, the tokenizer, reward
  callable, stat scope, and optional `dump_dir`; wrap the reward via
  `AsyncRewardWrapper` exactly like `MultiTurnWorkflow` does.
- Keep `arun_episode` async-only, drive generation through `InferenceEngine.agenerate`,
  and emit tensors using `concat_padded_tensors` so outputs stay
  `[batch, seq_len, ...]`.
- Use `areal/utils/data.py` helpers for padding/broadcasting, `areal/utils/logging` for
  logger plumbing, and `stats_tracker` for reward metrics.
- Persist transcripts under `{dump_dir}/{engine.get_version()}/` (follow the
  `multi_turn` implementation) when debugging is enabled.
- Update whichever entry script or launcher references the workflow (e.g.,
  `examples/multi-turn-math/train.py`, configs in `examples/**/conf/`, or CLI glue) so
  Hydra can import the new module.

### Introduce a reward function

- Create `areal/reward/<name>.py` and implement a callable following
  `areal/api/reward_api.py` (see `geometry3k_reward_fn` for reference).
- Accept `(prompt, completions, prompt_ids, completion_ids, **data)` and return a
  scalar; extract answers deterministically (`math_parser.math_equal`, regex, etc.) and
  avoid blocking I/O.
- Add the identifier to `VALID_REWARD_FN` and branch selection logic in
  `areal/reward/__init__.py` so configs like `reward.path=...` resolve automatically.
- When rewards rely on slow models or external services, keep the heavy code inside the
  reward module but let workflows wrap it with `AsyncRewardWrapper` (as in
  `MultiTurnWorkflow`).
- Document required dataset fields or endpoints in the module docstring/README so launch
  scripts can provision secrets or caches.

### Wire a new dataset

- Mirror the layout in `areal/dataset/gsm8k.py`, `clevr_count_70k.py`, etc.: create
  `areal/dataset/<name>.py` with `get_<name>_<type>_dataset` helpers for SFT/RL
  variants.
- Update `areal/dataset/__init__.py` by appending the dataset to `VALID_DATASETS` and
  adding a dispatch branch inside `_get_custom_dataset`.
- Define the sample schema explicitly (`messages`, `answer`, `image_path`, metadata) and
  validate it before returning; filter/trim sequences with tokenizer-aware checks when
  `max_length` is provided.
- Expose configuration knobs (path, split, type, max_length, processor/tokenizer
  requirements) via the `DatasetConfig` dataclass in `areal/api/cli_args.py`, then
  reference them in the relevant `examples/**/conf` YAML.
- If preprocessing or external storage is required, add a short note beside the loader
  or under `examples/<recipe>/README.md` so other agents know how to stage data.

### Launch training / evaluation

- Choose an existing script in `examples/**` (math, multi-turn, VLM, etc.) that mirrors
  your use case, then replicate its launcher pairing (`areal/launcher/local.py`,
  `ray.py`, `slurm.py`, or `sglang_server.py`).
- Read the example README to collect scheduler requirements, container images,
  environment variables, and any dataset preparation steps before running.
- Keep rollout actors and inference engines version-aligned by propagating
  `WeightUpdateMeta` (as shown in `examples/multi-turn-math/train.py`) and noting
  skipped weight updates explicitly if clusters are unavailable.
- Capture the Hydra/CLI overrides you used
  (`python ... +train_dataset.path=... engine.type=...`) inside the PR/test plan so runs
  are reproducible.
- When cluster access is blocked, document which launcher stages were skipped and what
  validation (unit tests, static checks) you ran instead.

### Publish docs

- Place prose in the right section under `docs/` (tutorial, algorithms, customization,
  lite, etc.) and update `_toc.yml` so Jupyter Book exposes the new page.
- Run `mdformat` (or `mdformat --check`) on edited Markdown plus `ruff format` on
  embedded code blocks when needed.
- Regenerate CLI docs with `python docs/generate_cli_docs.py` whenever
  `areal/api/cli_args.py` or CLI entrypoints change, then restage
  `docs/cli_reference.md`.
- Coordinate a docs build (or explain why it is skipped) and capture the limitation in
  your PR/testing notes if the hosted pipeline cannot run.

### Monitor metrics & artifacts

- Emit rollout/training metrics through `areal/utils/stats_tracker.py`; grab a scoped
  tracker (`stats_tracker.get("rollout")`) and log scalars so downstream `StatsLogger`
  backends (W&B/SwanLab) pick them up automatically.
- When debugging, pass `dump_dir` into workflows so transcripts persist under
  `{dump_dir}/{engine.get_version()}/` like `areal/workflow/multi_turn.py`; scrub
  sensitive data before committing artifacts.
- Checkpoint via `areal/utils/saver.py` and resume with `areal/utils/recover.py`; note
  the checkpoint path and version in your PR/test notes so others can reproduce the
  exact state.

## Testing & validation strategy

### Create or extend unit tests

- Place new tests under `areal/tests/` using `test_<topic>.py` so Pytest auto-discovers
  them (e.g., tensor helpers live in `test_utils.py`, schedulers in
  `test_local_scheduler.py`).
- Reuse fixtures + helpers: copy the pattern from `test_utils.py` (local fixtures
  feeding parametrized cases) or import shared logic from `areal/tests/utils.py`
  (`is_in_ci`, `get_bool_env_var`). Prefer `pytest.fixture` + `pytest.mark.parametrize`
  over ad-hoc loops.
- Keep tests hermetic by mocking engines/workflows similar to
  `test_engine_api_workflow_resolution.py`; avoid spinning up real clusters unless you
  are under `torchrun/` or `experimental/`.
- For GPU/distributed requirements, gate with `pytest.mark.skipif` or custom env checks
  (see `test_fsdp_engine_nccl.py` and `areal/tests/torchrun/`), and document the
  hardware dependency inside the skip reason.
- When tests need sample artifacts (configs, datasets), reuse the examples in
  `areal/tests/sft` or `areal/tests/grpo` rather than downloading new assets. Commit
  only lightweight fixtures.

### Run the right suites

- **Unit suites**: Target the file you touched, e.g.,
  `pytest areal/tests/test_utils.py`. If a full run is infeasible, list the exact
  command you would have executed.
- **Workflow smoke tests**: `areal/tests/grpo` exercises rollout loops and expects CUDA;
  acknowledge when skipped.
- **Distributed/FSDP suites**: `test_fsdp_*`, `test_sglang_engine.py`, RPC/torchrun
  folders require multi-node setups and distributed communication libraries. Call out
  the limitation explicitly.
- **Static checks**: Pre-commit runs Ruff lint/format, mdformat, clang-format,
  nbstripout, CLI doc regeneration, and autoflake. Note if hooks were not run locally
  and why.

Always mention resource requirements in PRs and in agent responses when tests are
skipped.

## Collaboration & review expectations

- **Branches**: Use kebab-case summaries (e.g., `feature/multi-turn-metrics`,
  `bugfix/fsdp-weight-sync`) so PR automation and reviewers can parse intent quickly.
- **Commits**: Follow Conventional Commit prefixes (`feat:`, `fix:`, `docs:`, etc.),
  keep the subject around 72 characters for readable logs (go longer only when the extra
  context is essential), write in imperative voice, and put deeper reasoning in the
  body. Squash noisy WIP commits before opening or updating a PR.
- **Pre-merge checks**: Run the full pre-commit stack (Ruff lint+format, mdformat,
  clang-format, nbstripout, CLI docs, autoflake). For doc-only edits, at least run
  `mdformat --check` on touched files and call out anything you could not run locally.
- **Surface scope upfront**: Tie the PR to a filed issue, summarize acceptance criteria,
  highlight risk areas (breaking changes, performance regressions), and note any
  configs, datasets, or launchers impacted.
- **Testing evidence**: List the exact commands you executed (unit, workflow smoke, docs
  build). When hardware is unavailable, state the skipped suites, why they were skipped,
  and what alternative validation (static analysis, mocks) you performed.
- **Async + resource safety**: When touching workflows/engines, confirm async code
  awaits I/O, avoids blocking calls, and preserves weight versioning
  (`set_version`/`update_weights`). Document memory/GPU expectations and dataset or
  checkpoint storage requirements inside the PR/test notes.
- **Style, config & docs**: Ensure Ruff/clang-format/autoflake output is clean. Thread
  new options through the right dataclasses/YAMLs, update docs/CLI references, and
  verify hyperlinks. Mention any formatting gaps you plan to address later via
  `TODO(agent)`.
- **Observability & cleanup**: Keep metrics flowing through `stats_tracker`/
  `StatsLogger`, expose dump directories when debugging, and remove stray debug prints
  or commented code. Note migrations or recovery steps when checkpoints or evaluators
  change so reviewers know what to verify.

## Reference material

- **Docs portal** (`https://inclusionai.github.io/AReaL/`): Hosted Jupyter Book with the
  full table of contents; use it to cross-check rendered diagrams, formulas, and links.
- **Tutorials & quickstart** (`docs/tutorial/quickstart.md`): End-to-end GSM8K GRPO run
  covering single-node LocalLauncher flows, Ray/Slurm deployment knobs, SkyPilot
  recipes, and the legacy→lite config converter.
- **Lite deep dive** (`docs/lite/gsm8k_grpo.md`): Architecture-level walkthrough of how
  launchers, RemoteSGLangEngine, workflows, and FSDP PPO actors coordinate during
  asynchronous GRPO on GSM8K; great for understanding control flow before editing
  engines or workflows.
- **Customization guides** (`docs/customization/*.md`): Step-by-step patterns for adding
  datasets, authoring new `RolloutWorkflow` subclasses, or wiring custom RL algorithms
  while keeping configs Hydra-friendly.
- **Algorithm notes** (`docs/algorithms/*.md`): Reference math + configuration advice
  for GRPO, DAPO/DAPO-style filters, async RL, GSPO, LitePPO, m2po, rloo, etc.,
  including when to switch between synchronous and asynchronous modes.
- **Best practices** (`docs/best_practices/*.md`): Practical debugging playbooks,
  reward-drift diagnostics, OOM mitigation, and performance profiling checklists you
  should cite when explaining skipped tests or perf limitations.
- **CLI & doc tooling** (`docs/cli_reference.md`): Auto-generated CLI argument catalog
  plus instructions for regenerating docs/CLI output before landing config changes.
- **Benchmarks & reproducibility** (`docs/references/*.md`): Canonical benchmark setups,
  dataset/model combos, and experiment-log expectations to mention in PR validation
  notes.
- **Version history** (`docs/version_history.md`): Release timeline noting major API
  moves, deprecations, and migration steps from legacy AReaL to AReaL-lite.
