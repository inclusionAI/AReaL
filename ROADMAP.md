# AReaL Roadmap

This roadmap outlines the planned features and improvements for AReaL. We welcome
community feedback and contributions to help shape the future direction of the project.

## Current Status

**Latest Release:** Check [releases](https://github.com/inclusionAI/AReaL/releases) for
the most recent version

**Active Development:** AReaL-lite lightweight framework with algorithm-first design

## 2025 Q3 Roadmap

For detailed Q3 2025 roadmap, see
[GitHub Issue #257](https://github.com/inclusionAI/AReaL/issues/257).

### High Priority

**🎯 Core Features**

- [ ] Enhanced context parallelism support for long-context training
- [ ] Improved LoRA training integration with FSDP
- [ ] Zero-bubble pipeline parallelism implementation
- [ ] Expert parallelism optimization for MoE models

**⚡ Performance & Scalability**

- [ ] Weight update latency optimization
- [ ] Asynchronous rollout performance improvements
- [ ] Memory usage optimization for large-scale training
- [ ] Better GPU utilization monitoring and profiling tools

**🔧 Developer Experience**

- [ ] Simplified local development setup
- [ ] Better error messages and debugging tools
- [ ] Interactive tutorial notebooks
- [ ] Video tutorials and walkthroughs

### Medium Priority

**🧠 Algorithm Support**

- [ ] Additional RL algorithms (DAPO variants, LitePPO improvements)
- [ ] Multi-objective RL support
- [ ] Curriculum learning integration
- [ ] Online evaluation during training

**📊 Monitoring & Observability**

- [ ] Real-time training dashboard
- [ ] Enhanced metrics tracking (beyond W&B/SwanLab)
- [ ] Automatic anomaly detection in training
- [ ] Better checkpoint management and recovery

**🌐 Ecosystem Integration**

- [ ] Better HuggingFace Hub integration
- [ ] Support for additional model families (Llama 3.x, Gemma variants)
- [ ] Integration with popular evaluation frameworks
- [ ] OpenAI-compatible API for inference

### Lower Priority / Future Considerations

**🔬 Research Features**

- [ ] Multi-modal RL (vision-language-action)
- [ ] Federated RL support
- [ ] Meta-learning capabilities
- [ ] Offline RL enhancements

**📚 Documentation & Examples**

- [ ] More domain-specific examples (tool use, code generation, search)
- [ ] Case studies from production deployments
- [ ] Performance tuning guides for different hardware setups
- [ ] Migration guides from other frameworks

**🏗️ Infrastructure**

- [ ] Kubernetes deployment support
- [ ] Auto-scaling based on workload
- [ ] Cost optimization tools
- [ ] Multi-cloud support enhancements

## Community Requests

The following features have been requested by the community. Upvote existing issues or
create new ones to help us prioritize:

- **Model Support:** Additional model architectures (check
  [issues with `area/model` label](https://github.com/inclusionAI/AReaL/labels/area%2Fmodel))
- **Dataset Integration:** New benchmark datasets (check
  [issues with `area/dataset` label](https://github.com/inclusionAI/AReaL/labels/area%2Fdataset))
- **Workflow Templates:** Pre-built workflows for common use cases
- **Deployment:** Production deployment best practices and tooling

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

1. **Pick an Issue:** Look for issues labeled
   [`help wanted`](https://github.com/inclusionAI/AReaL/labels/help%20wanted) or
   [`good first issue`](https://github.com/inclusionAI/AReaL/labels/good%20first%20issue)
1. **Discuss Approach:** Comment on the issue to discuss your implementation plan
1. **Submit a PR:** Follow our [contributing guide](CONTRIBUTING.md) to submit your
   changes
1. **Collaborate:** Work with maintainers to refine and merge your contribution

### 📣 Share Feedback

- **What's working well?** Let us know in
  [Discussions](https://github.com/inclusionAI/AReaL/discussions)
- **What's painful?** File bug reports or usability issues
- **What's missing?** Request features you need for your projects

## Release Cycle

**Minor Releases:** Weekly - Bug fixes and small improvements

**Major Releases:** Monthly - New features and significant changes

**Breaking Changes:** Announced well in advance with migration guides

## Completed Milestones

### v0.3 (boba²) - June 2025

- ✅ Fully asynchronous RL training (2.77× speedup)
- ✅ Multi-turn agentic RL training simplified
- ✅ Research paper published ([arxiv](https://arxiv.org/pdf/2505.24298))

### v0.2 (boba) - March 2025

- ✅ SGLang support for faster inference
- ✅ State-of-the-art 7B and 32B math reasoning models
- ✅ Performance optimizations

### v0.1 - February 2025

- ✅ Initial release
- ✅ 1.5B and 7B model training pipelines
- ✅ GRPO, PPO, RLOO algorithm support
- ✅ FSDP and Megatron backend support

### AReaL-lite - July 2025

- ✅ Algorithm-first API redesign
- ✅ 80% code reduction while maintaining 90% functionality
- ✅ Single-file customization support
- ✅ Improved developer experience

## Long-Term Vision

Our vision for AReaL is to become the **go-to framework for training reasoning and
agentic AI systems** that is:

1. **Accessible:** Easy to get started, whether you're a researcher or practitioner
1. **Scalable:** Scales from laptop to 1000+ GPU clusters seamlessly
1. **Flexible:** Supports diverse algorithms, models, and use cases
1. **Performant:** Industry-leading training speed and efficiency
1. **Open:** Fully open-source with transparent development

## Stay Updated

- **GitHub Releases:** [Watch releases](https://github.com/inclusionAI/AReaL/releases)
- **Discussions:** [Join discussions](https://github.com/inclusionAI/AReaL/discussions)
- **WeChat Group:** [Join our community](./assets/wechat_qrcode.png)
- **Documentation:** [Read the docs](https://inclusionai.github.io/AReaL/)

______________________________________________________________________

**Last Updated:** 2025-01-21

**Questions about the roadmap?** Open a discussion in
[GitHub Discussions](https://github.com/inclusionAI/AReaL/discussions) or ask in our
[WeChat group](./assets/wechat_qrcode.png).
