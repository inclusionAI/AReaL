# Sandbox Integration Example for AReaL

This directory contains examples of using E2B-compatible sandboxes for
sandboxed code execution in RL training workflows.

## Overview

AReaL's `SandboxToolWorkflow` enables training LLMs with tool-integrated
reasoning (TIR) where generated code is executed in isolated sandbox
instances, providing:

- **KVM-level isolation**: Generated code cannot escape the sandbox
- **< 60ms cold start**: Per-episode sandbox creation without latency overhead
  (with CubeSandbox self-hosted deployment)
- **< 5MB per instance**: Support for thousands of concurrent episodes
- **E2B SDK compatible**: Standard `e2b-code-interpreter` Python API

## Quick Start

### 1. Install dependencies

```bash
pip install e2b-code-interpreter
```

### 2. Start an E2B-compatible service

Use any E2B-compatible backend:

- [E2B Cloud](https://e2b.dev) (managed SaaS)
- [CubeSandbox](https://github.com/TencentCloud/CubeSandbox) (self-hosted,
  recommended for RL training)

### 3. Configure environment

```bash
export SANDBOX_API_URL="http://your-e2b-api:3000"
export SANDBOX_API_KEY="your-api-key"
```

### 4. Run training

```bash
uv run python gsm8k_sandbox_rl.py
```

## Architecture

```
areal/api/sandbox_api.py          ← Protocol + Config (pure abstractions)
areal/infra/sandbox/              ← E2B adapter + pool management
  ├── e2b_sandbox.py              ← E2B SDK wrapper
  ├── local_sandbox.py            ← Local executor (debug only)
  ├── factory.py                  ← Backend factory
  └── manager.py                  ← Per-thread sandbox pool
areal/workflow/sandbox_tool.py    ← SandboxToolWorkflow (core workflow)
```

## Configuration

```yaml
sandbox:
  enabled: true
  backend: e2b           # or "local" for debugging
  api_url: "http://localhost:3000"
  api_key: ""
  template_id: ""        # optional pre-configured environment
  timeout: 30.0          # per-execution timeout (seconds)
  max_tool_turns: 5      # maximum tool call rounds per episode
  pool_size: 0           # 0 = on-demand, N = pre-warmed pool
```

## Backends

| Backend | Isolation | Use Case |
|---------|-----------|----------|
| `e2b`  | KVM (kernel-level) | Production RL training |
| `local` | None (in-process) | Development/debugging only |

## TIR Integration

For existing TIR workflows, you can replace the unsafe `PythonTool` with
the sandboxed `E2BSandboxPythonTool`:

```python
# Before (unsafe)
from tools.python_tool import PythonTool
tool = PythonTool(timeout=30)

# After (sandboxed)
from tools.e2b_sandbox_tool import E2BSandboxPythonTool
tool = E2BSandboxPythonTool(
    timeout=30,
    api_url="http://your-e2b-api:3000",
)
```
