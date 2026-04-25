# Sandbox Agent RL Training

Train a math-solving agent with sandboxed code execution using AReaL's
AgentWorkflow pattern and E2B-compatible sandbox backends.

## Overview

The agent uses OpenAI-compatible **function calling** (via the Agents SDK)
to invoke a `run_python_code` tool. Code is executed in an isolated E2B
sandbox (KVM-level isolation), making it safe for untrusted model-generated
code. AReaL's proxy automatically records all logprobs and computes GRPO
gradients.

Key properties:

- **KVM-level isolation** — generated code cannot escape the sandbox
- **< 60 ms cold start** — with CubeSandbox self-hosted deployment
- **Standard function calling** — model learns the OpenAI tool-use format
- **Deployable** — trained model works with any OpenAI-compatible API

## Quick Start

### 1. Install dependencies

```bash
pip install e2b-code-interpreter openai-agents math-verify
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
python examples/sandbox/train_sandbox_agent.py \
    --config examples/sandbox/gsm8k_agent_sandbox.yaml
```

## Architecture

```
areal/api/sandbox_api.py              ← Protocol + Config (pure abstractions)
areal/infra/sandbox/                  ← E2B backend + local fallback
  ├── e2b_sandbox.py                  ← E2B SDK wrapper
  ├── local_sandbox.py                ← Local executor (debug only)
  └── factory.py                      ← Backend factory

examples/sandbox/                     ← This directory
  ├── sandbox_math_agent.py           ← AgentWorkflow (function calling)
  ├── train_sandbox_agent.py          ← Training entry point
  ├── configs.py                      ← SandboxAgentConfig dataclass
  └── gsm8k_agent_sandbox.yaml        ← YAML config
```

## How It Works

1. AReaL starts SGLang inference engine + OpenAI-compatible proxy
2. `SandboxMathAgent.run()` creates an `AsyncOpenAI` client pointing at
   the proxy
3. The OpenAI Agents SDK runs the agent loop — model generates tool calls,
   agent executes them in the E2B sandbox, feeds results back
4. Final answer is scored with `math_reward_fn` (exact match via
   `math-verify`)
5. AReaL proxy records all logprobs automatically; GRPO computes gradients

## Configuration

The `sandbox:` section in YAML controls sandbox behaviour:

```yaml
sandbox:
  enabled: true
  backend: e2b            # or "local" for debugging (unsafe)
  api_url: ""             # or set SANDBOX_API_URL env var
  api_key: ""             # or set SANDBOX_API_KEY env var
  template_id: ""         # optional pre-configured environment
  ssl_cert_file: ""       # for self-hosted TLS
  timeout: 30.0           # per-execution timeout
```

## Backends

| Backend | Isolation | Use Case |
|---------|-----------|----------|
| `e2b` | KVM (kernel-level) | Production RL training |
| `local` | None (subprocess) | Development / debugging only |

## See Also

- [Agent Workflow docs](../../docs/customization/agent.md)
- [Agent Workflow example](../agent_workflow/) — same pattern, no sandbox
