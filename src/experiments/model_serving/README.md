# Model Serving Utilities

Utilities for serving and interacting with LLMs.

## Services

### [ollama_serving/](ollama_serving/)
Production-ready Ollama cluster with nginx load balancing for multi-GPU inference.

### [vllm_serving/](vllm_serving/)
vLLM server management with support for tensor, pipeline, and data parallelism.

### [litellm_proxy/](litellm_proxy/)
LiteLLM proxy setup for unified API interface to various LLM providers.

### [external_providers/](external_providers/)
Test scripts for external model providers (Together AI, Baseten).

## Quick Start

Choose your preferred serving method and follow the README in each directory. 