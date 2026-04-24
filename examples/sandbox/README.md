# CubeSandbox Integration Example for AReaL

This directory contains examples of using CubeSandbox for sandboxed code
execution in RL training workflows.

## Overview

AReaL's `SandboxToolWorkflow` enables training LLMs with tool-integrated
reasoning (TIR) where generated code is executed in isolated CubeSandbox
instances, providing:

- **KVM-level isolation**: Generated code cannot escape the sandbox
- **< 60ms cold start**: Per-episode sandbox creation without latency overhead
- **< 5MB per instance**: Support for thousands of concurrent episodes
- **E2B SDK compatible**: Standard `e2b-code-interpreter` Python API

## Quick Start

### 1. Install dependencies

```bash
pip install e2b-code-interpreter
```

### 2. Start CubeSandbox

Follow the [CubeSandbox setup guide](https://github.com/TencentCloud/CubeSandbox)
to deploy a local or cloud instance.

### 3. Configure environment

```bash
export SANDBOX_API_URL="http://your-cubesandbox:3000"
export SANDBOX_API_KEY="your-api-key"
```

### 4. Run training

```bash
uv run python gsm8k_sandbox_rl.py
```

## Architecture

```
areal/api/sandbox_api.py          ← Protocol + Config (pure abstractions)
areal/infra/sandbox/              ← CubeSandbox adapter + pool management
  ├── cube_sandbox.py             ← E2B SDK wrapper
  ├── local_sandbox.py            ← Local executor (debug only)
  ├── factory.py                  ← Backend factory
  └── manager.py                  ← Per-thread sandbox pool
areal/workflow/sandbox_tool.py    ← SandboxToolWorkflow (core workflow)
```

## Configuration

```yaml
sandbox:
  enabled: true
  backend: cube          # or "local" for debugging
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
| `cube`  | KVM (kernel-level) | Production RL training |
| `local` | None (in-process) | Development/debugging only |

## TIR Integration

For existing TIR workflows, you can replace the unsafe `PythonTool` with
the sandboxed `CubeSandboxPythonTool`:

```python
# Before (unsafe)
from tools.python_tool import PythonTool
tool = PythonTool(timeout=30)

# After (sandboxed)
from tools.cube_sandbox_tool import CubeSandboxPythonTool
tool = CubeSandboxPythonTool(
    timeout=30,
    api_url="http://your-cubesandbox:3000",
)
```
