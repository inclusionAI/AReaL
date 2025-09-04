# Experiments

This directory contains **experimental code** that is offered as-is and should be treated as experimental components, not part of the core tau2 benchmark.

> ⚠️ **Important**: The code in this directory is experimental and may not be fully tested or supported. Use at your own discretion.

## Overview

The `experiments/` folder provides additional tools, experimental agents, model training pipelines, and serving utilities that extend the tau2 benchmark but are not part of the core evaluation framework. These components are provided for research purposes and to enable advanced use cases.

## Directory Structure

### 📊 [eval/](eval/)
Evaluation utilities and analysis tools:
- Result analysis scripts
- Custom evaluation pipelines
- Performance comparison tools

### 🚀 [model_serving/](model_serving/)
Production-ready model serving solutions:

📖 **Main Documentation**: [model_serving/README.md](model_serving/README.md)

#### [ollama_serving/](model_serving/ollama_serving/)
- Multi-GPU Ollama cluster with nginx load balancing
- Production-ready setup with health monitoring
- **Drop-in replacement** running on standard Ollama port (11434)

📖 **Documentation**: [model_serving/ollama_serving/README.md](model_serving/ollama_serving/README.md)

#### [vllm_serving/](model_serving/vllm_serving/)
- Comprehensive vLLM server management with parallelism support
- Tensor, pipeline, and data parallelism configurations
- Tool calling support and GPU optimization

📖 **Documentation**: [model_serving/vllm_serving/README.md](model_serving/vllm_serving/README.md)

#### [litellm_proxy/](model_serving/litellm_proxy/)
- Unified API interface to various LLM providers
- OpenAI-compatible endpoints
- Environment-based configuration management

📖 **Documentation**: [model_serving/litellm_proxy/README.md](model_serving/litellm_proxy/README.md)

#### [external_providers/](model_serving/external_providers/)
- Test scripts for external model providers (Together AI, Baseten)
- Integration examples and API testing utilities

📖 **Documentation**: [model_serving/external_providers/README.md](model_serving/external_providers/README.md)

### 🎯 [model_training/](model_training/)
Complete model training pipelines and utilities:

#### [dataset_prep/](model_training/dataset_prep/)
- **Dataset preparation** from tau2 trajectories for SFT training
- Conversion to OpenAI format for popular training frameworks
- Token analysis and truncation optimization tools

📖 **Documentation**: [model_training/dataset_prep/README.md](model_training/dataset_prep/README.md)

#### [data/](model_training/data/)
- Pre-processed training datasets from tau2 evaluations
- Contains trajectories from GPT-4.1-mini, o4-mini, and Claude-3.5-Sonnet
- Train/test splits with successful trajectory filtering

📖 **Documentation**: [model_training/data/README.md](model_training/data/README.md)

#### [sft_trl/](model_training/sft_trl/) *(Deprecated)*
- Supervised Fine-Tuning using HuggingFace TRL framework
- Multi-GPU training configurations with Accelerate

📖 **Documentation**: [model_training/sft_trl/README.md](model_training/sft_trl/README.md)

#### [rl/](model_training/rl/)
Reinforcement Learning training frameworks:

##### [verl-agent/](model_training/rl/verl-agent/)
- **VERL-based RL training** adapted for tau2-bench
- Requires 4x A100 GPUs (80GB)
- Integration with tau2 gym environment

📖 **Documentation**: [model_training/rl/verl-agent/README_tau.md](model_training/rl/verl-agent/README_tau.md)

##### [areal/](model_training/rl/areal/)
- **GRPO (Group Relative Policy Optimization)** on tau2-bench
- Multi-GPU distributed training
- LoRA support for efficient fine-tuning

📖 **Documentation**: [model_training/rl/areal/README_grpo_tau2.md](model_training/rl/areal/README_grpo_tau2.md)


### 🤖 [agents/](agents/)
Enhanced experimental agent implementations:
- **LLMAgentV2**: State-of-the-art agent with enhanced prompting techniques from 2024-2025 research
- Features: Multi-step reasoning, agentic workflows, enhanced error handling, context optimization
- **Drop-in replacement** for the original LLMAgent with backward compatibility

📖 **Documentation**: [agents/README_V2.md](agents/README_V2.md)


## Quick Start

### Model Serving
Choose your preferred serving method:

```bash
# Ollama cluster (recommended for production)
cd model_serving/ollama_serving/
./setup_ollama_cluster.sh start

# vLLM server (for advanced configurations)
cd model_serving/vllm_serving/
./vllm_start_server.sh start

# LiteLLM proxy (for unified API access)
cd model_serving/litellm_proxy/
./litellm_proxy.sh start
```

### Dataset Preparation
Prepare training data from tau2 trajectories:

```bash
# Create SFT dataset from trajectory results
cd model_training/dataset_prep/
python prepare_dataset.py make \
    --result-dir ../../../data/tau2/results/final \
    --save-dir ../../../data/datasets \
    --name my-sft-dataset \
    --success-only

# Convert to OpenAI format
python prepare_dataset.py to-openai \
    --dataset-path ../../../data/datasets/my-sft-dataset.json \
    --save-dir ../../../data/datasets/my-sft-dataset-openai \
    --split train test
```

### Enhanced Agent
Use the experimental LLMAgentV2 as a drop-in replacement:

```python
from experiments.agents.llm_agent_completion import LLMAgentV2

# Replace LLMAgent with LLMAgentV2
agent = LLMAgentV2(
    tools=your_tools,
    domain_policy=your_policy,
    llm="gpt-4",
    enable_self_consistency=True,
    max_context_tokens=100000
)
```

## Integration with Core tau2

### Using Experimental Agents
Register experimental agents in the main tau2 registry:

```python
# In src/tau2/registry.py
from experiments.agents.llm_agent_completion import LLMAgentV2
registry.register_agent(LLMAgentV2, "llm_agent_v2")
```

Then use in commands:
```bash
tau2 run --agent llm_agent_v2 --domain telecom ...
```

### Using Served Models
Point tau2 to locally served models:

```bash
# Using Ollama cluster
tau2 run --agent-llm "ollama/qwen2.5:7b" --agent-llm-args '{"api_base": "http://localhost:11434"}'

# Using vLLM server
tau2 run --agent-llm "hosted_vllm/Qwen/Qwen2.5-7B" --agent-llm-args '{"api_base": "http://localhost:8000/v1"}'
```

## Development Guidelines

When working with experimental code:

1. **Backward Compatibility**: Maintain compatibility with core tau2 interfaces when possible
2. **Documentation**: Each experimental component should have its own README
3. **Testing**: Include basic testing scripts and examples
4. **Dependencies**: Manage dependencies carefully to avoid conflicts with core tau2
5. **Isolation**: Keep experimental code self-contained within this directory

## Contributing

Experimental contributions are welcome! Please:

1. Add comprehensive documentation in your subfolder's README
2. Include example usage and test scripts
3. Mark any breaking changes or dependencies clearly
4. Consider the experimental nature - code doesn't need to be production-ready

## Support

Since this is experimental code:

- **No guarantees** of stability or continued support
- **Community-driven** - contributions and improvements welcome
- **Use at your own risk** - test thoroughly before production use
- **Documentation-first** - refer to individual README files for detailed usage

For core tau2 benchmark support, see the main project documentation.
