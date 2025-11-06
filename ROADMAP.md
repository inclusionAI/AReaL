# AReaL Roadmap

This roadmap outlines the planned features and improvements for AReaL in the next
quarter. We welcome community feedback and contributions to help shape the future
direction of the project.

**Latest Release:** Check [releases](https://github.com/inclusionAI/AReaL/releases) for
the most recent version.

## 2025 Q4 Roadmap (due January 31, 2026)

[GitHub Issue #542](https://github.com/inclusionAI/AReaL/issues/542).

This roadmap tracks major planned enhancements through January 31, 2026. Items are
organized into two categories:

- **On-going:** Features currently under active development by the core AReaL team
- **Planned but not in progress:** Features with concrete implementation plans where we
  welcome community contributions

### Backends

**On-going**

- [ ] Single-controller mode: https://github.com/inclusionAI/AReaL/issues/260
- [ ] Detailed profiling for optimal performance across different scales
- [ ] RL training with cross-node vLLM pipeline/context parallelism

**Planned but not in progress**

- [ ] Multi-LLM training (different agents with different parameters)
- [ ] Data transfer optimization in single-controller mode
- [ ] Auto-scaling inference engines in single-controller mode
- [ ] Elastic weight update setup and acceleration
- [ ] Low-precision RL training

### Usability

**Planned but not in progress**

- [ ] Wrap training scripts into trainers
- [ ] Fully respect allocation mode in trainers/training scripts
- [ ] Support distributed training and debugging in Jupyter notebooks
- [ ] Refactor FSDP/Megatron engine/controller APIs to finer granularity
- [ ] Add CI pipeline to build Docker images upon release
- [ ] Example of using a generative or critic-like reward model

### Documentation

**Planned but not in progress**

- [ ] Tutorial on how to write efficient async rollout workflows
- [ ] Benchmarking and profiling guide
- [ ] Use case guides: offline inference, offline evaluation, multi-agent training
- [ ] AReaL performance tuning guide
  - [ ] Device allocation strategies for training and inference
  - [ ] Parallelism strategy configuration for training and inference

## Historical Roadmaps

### 2025 Q3

[GitHub Issue #257](https://github.com/inclusionAI/AReaL/issues/257).

**Backends**

Completed:

- Megatron training backend support
- SGLang large expert parallelism (EP) inference support
- Remote vLLM inference engine
- Ulysses context parallelism & tensor parallelism for FSDP backend
- End-to-end MoE RL training with large EP inference and Megatron expert parallelism
- Distributed weight resharder for Megatron training backend

Canceled:

- Local SGLang inference engine with inference/training colocation (hybrid engine)
- RL training with SGLang pipeline parallelism

**Usability**

Completed:

- OpenAI-compatible client support
- Support RLOO
- Provide benchmarking configuration examples:
  - DAPO
  - Bradley-Terry reward modeling
  - PPO with critic models
  - REINFORCE++

**Documentation**

Completed:

- OpenAI-compatible client documentation
- Out-of-memory (OOM) troubleshooting guide
- AReaL debugging best practices:
  - LLM server-only debugging - How to launch LLM servers independently and debug agent
    workflows
  - Mock data and torchrun debugging - Creating synthetic data and using `torchrun` for
    algorithm debugging
  - Training-free evaluation experiments - Running evaluations without training or
    additional GPUs

## How to Influence the Roadmap

We value community input! Here's how you can help shape AReaL's future:

### 💡 Propose New Features

1. **Check Existing Issues:** Search
   [issues](https://github.com/inclusionAI/AReaL/issues) and
   [discussions](https://github.com/inclusionAI/AReaL/discussions) to see if your idea
   already exists
1. **Create a Feature Request:** Use our
   [feature request template](https://github.com/inclusionAI/AReaL/issues/new?template=feature.md)
1. **Discuss in GitHub Discussions:** Post in
   [Ideas category](https://github.com/inclusionAI/AReaL/discussions/categories/ideas)
   for early feedback
1. **Vote on Features:** Use 👍 reactions on issues to show support

### 🛠️ Contribute Implementation

Check our [contribution guide](CONTRIBUTING.md).

## Release Cycle

**Minor Releases:** Bi-weekly - Bug fixes, small improvements, and new features

**Major Releases:** Quarterly - Important milestones and significant changes

## Historical Milestones

Check [our historical milestone summaries since open-source](docs/version_history.md).

## Long-Term Vision

Our vision for AReaL is to become the **go-to framework for training reasoning and
agentic AI systems** that is:

1. **Accessible:** Easy to get started, whether you're a researcher or practitioner
1. **Scalable:** Scales from laptop to 1000+ GPU clusters seamlessly
1. **Flexible:** Supports diverse algorithms, models, and use cases
1. **Performant:** Industry-leading training speed and efficiency
1. **Open:** Fully open-source with transparent development

______________________________________________________________________

**Last Updated:** 2025-11-06

**Questions about the roadmap?** Open a discussion in
[GitHub Discussions](https://github.com/inclusionAI/AReaL/discussions) or ask in our
[WeChat group](./assets/wechat_qrcode.png).
