<h1 align="center">
<em>AReaL</em>: A Large-Scale Asynchronous Reinforcement Learning System
</h1>

<p align="center">
| <a href="https://arxiv.org/pdf/2505.24298"><b>Paper</b></a> | <a href="https://inclusionai.github.io/AReaL/"><b>Documentation</b></a> | <a href="https://deepwiki.com/inclusionAI/AReaL"><b>Ask DeepWiki</b></a> | <a href="https://huggingface.co/collections/inclusionAI/areal-boba-2-683f0e819ccb7bb2e1b2f2d5"><b>🤗 Models & Data</b></a> |
<a href="./assets/wechat_qrcode.png" target="_blank"><img src="./assets/wechat_icon.png" width="20" style="vertical-align: middle;"> <b>WeChat (微信) Group</b></a> |
</p>

<img align="right" alt="ReaL" src="/assets/logo.png" width="20%">

AReaL is an open-source **fully asynchronous** reinforcement learning training system
for large **reasoning and agentic models**, developed by members from Tsinghua IIIS and
the AReaL Team at Ant Group. Built upon the open-source project
[ReaLHF](https://github.com/openpsi-project/ReaLHF), we are fully committed to
open-source principles by providing the training details, data, and infrastructure
required to reproduce our results, along with the models themselves. AReaL aims to help
everyone build their own AI agents easily and affordably. Our team loves milk tea
because it's delicious, customizable, and affordable—we hope you enjoy our project just
as much as you'd enjoy real milk tea. Cheers!

**AReaL Highlights**

- ⚡ **Flexibility**: Seamless customization for
  [multi-turn agentic rollout](https://inclusionai.github.io/AReaL/customization/agent.html)
  with
  [other agentic frameworks](https://inclusionai.github.io/AReaL/tutorial/agentic_rl.html).
- 📈 **Scalability**: **Stable** fully asynchronous RL training with **industry-leading
  speed**.
- ✨ **Cutting-Edge Performance**: State-of-the-art [math](/blog/AReaL_v0_2.md),
  [coding](/blog/AReaL_v0_3.md), [search](https://github.com/inclusionAI/ASearcher), and
  [customer service](https://arxiv.org/abs/2601.22607) agents.

## 📰 News

**\[2026/02/06\]** We are delighted to introduce **EigenData**, a self-evolving data
synthesis engine. Combined with RL training on AReaL, the 235B MoE model surpasses
Gemini 3.0 Pro and GPT 5.2 on $\\tau^2$-bench! Check out the
[paper](https://arxiv.org/pdf/2601.22607),
[code](https://github.com/inclusionAI/AReaL/tree/main/examples/tau2), and
[announcement on X](https://x.com/Eigen_AI_Labs/status/2018561648022130805?s=20).

**\[2026/01/15\]** Congrats to our friends at [CAMEL-AI](https://www.camel-ai.org/) for
open-sourcing [SETA](https://github.com/camel-ai/seta), their terminal agent RL project
trained with AReaL! Check out
[their training workflow](https://github.com/camel-ai/seta/tree/main/training/tbench_areal_workflow)
and the [announcement on X](https://x.com/guohao_li/status/2009678513574408636).

**\[2026/01/01\]** Happy New Year! Thanks to the outstanding contribution from
@HwVanICI, we are excited to officially announce stable support for AReaL training on
**Ascend NPU devices**! The code is actively maintained and continuously updated in the
[`ascend` branch](https://github.com/inclusionAI/AReaL/tree/ascend). Check out
[our documentation](https://inclusionai.github.io/AReaL/tutorial/installation_npu.html)
to get started, and feel free to report any issues!

<details>
<summary><b>📋 Previous Releases</b></summary>

**\[2025/08/30\]** Introducing ASearcher, a state-of-the-art search agent built with
AReaL's end-to-end asynchronous RL training. Check out the [paper](assets/paper.pdf) and
the [open-source repository](https://github.com/inclusionAI/ASearcher)!

**\[2025/07/31\] (AReaL-lite)** We introduce AReaL-lite, a **lightweight** version of
AReaL designed specifically for AI researchers and rapid prototyping. AReaL-lite
features an **algorithm-first** API design that prioritizes ease of use and algorithm
development, while natively supporting **fully asynchronous agentic RL**. With 80% fewer
lines of code, AReaL-lite maintains 90% of AReaL's performance and core functionality.
Check out [our AReaL-lite design documentation](/areal/README.md) and
[the quickstart guide](https://inclusionai.github.io/AReaL/tutorial/quickstart.html) to
begin your journey with **AReaL-lite**!

**\[2025/06/03\] (v0.3, boba²)** We release **boba²** (double-boba) for fully
asynchronous RL training, which achieves **2.77× speedup while delivering comparable or
superior training performance** compared to synchronous systems. Furthermore,
asynchronous RL significantly simplifies multi-turn agentic RL training setup! Check out
[our v0.3 overview blog](/blog/AReaL_v0_3.md) and the
[research paper](assets/paper.pdf).

**\[2025/03/31\] (v0.2, boba)** Introducing our milestone release—boba! Please call it
A-ReaL-boba! This release features significantly faster training with SGLang support and
state-of-the-art 7B and 32B models for mathematical reasoning. Check out our
[v0.2 technical blog](/blog/AReaL_v0_2.md).

**\[2025/02/24\] (v0.1)** Our initial release includes reproducible results for 1.5B and
7B Large Reasoning Models (LRMs). Check out our
[v0.1 technical blog](/blog/AReaL_v0_1.md).

</details>

## 🚀 Getting Started

First, install the package:

```bash
git clone https://github.com/inclusionAI/AReaL
cd AReaL
pip install uv
uv sync --extra cuda
```

Our training scripts automatically download the required dataset (openai/gsm8k) and
model (Qwen/Qwen2-1.5B-Instruct). To run on a single node:

```bash
python3 examples/math/gsm8k_rl.py --config examples/math/gsm8k_grpo.yaml scheduler.type=local
```

To run on a Ray cluster with 2 nodes and 8 GPUs per node (remember to update paths in
the YAML file to point to your shared storage):

```bash
python3 examples/math/gsm8k_rl.py --config examples/math/gsm8k_grpo.yaml \
  cluster.n_nodes=2 cluster.n_gpus_per_node=8 \
  scheduler.type=ray
```

For comprehensive setup instructions, see
[our quickstart guide](https://inclusionai.github.io/AReaL/tutorial/quickstart.html).

## 📚 Examples

### Math & Reasoning

| Task                                             | Description                                                                                  | Performance                                                       |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| **[Math](examples/math/)**                       | GSM8K math reasoning with GRPO, PPO, DAPO, REINFORCE, RLOO, LitePPO, DR-GRPO, GSPO, and more | -                                                                 |
| **[Multi-Turn Math](examples/multi_turn_math/)** | Multi-turn math agent with reward discounting across turns                                   | [Training Curve](examples/multi_turn_math/reward_curve.png)       |
| **[LoRA Math](examples/lora/)**                  | Parameter-efficient math training with LoRA (SGLang/vLLM backends)                           | -                                                                 |
| **[Countdown](examples/countdown/)**             | Countdown numbers game with custom rewards                                                   | [Training Curve](examples/countdown/countdown_training_curve.png) |

### Agentic RL

| Task                                                     | Description                                                            | Performance                                                                  |
| -------------------------------------------------------- | ---------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| **[General Agent](examples/agent_workflow/)**            | General agentic training with any agentic frameworks                   | [Guide](docs/tutorial/agentic_rl.md)                                         |
| **[Tau2 Customer Service](examples/tau2/)**              | Customer service agent on Tau2-Bench (retail, airline, telecom)        | [Paper](https://arxiv.org/abs/2601.22607)                                    |
| **[Search Agent](examples/search_agent/)**               | End-to-end search agent with Tongyi-DeepResearch workflow              | [Training Curve](examples/search_agent/tongyi_deepresearch/reward_curve.png) |
| **[Tool-Integrated Reasoning](examples/tir/)**           | Multi-turn tool calling during reasoning (Python executor, calculator) | [Training Curve](examples/tir/figures/task_reward.png)                       |
| **[OpenAI Agents Integration](examples/openai_agents/)** | Integration with OpenAI Agents SDK for agentic workflows               | -                                                                            |
| **[CAMEL-AI Integration](examples/camel/)**              | Integration with CAMEL-AI framework for agentic RL                     | -                                                                            |

### Vision-Language Models

| Task                                | Description                                               | Performance                                     |
| ----------------------------------- | --------------------------------------------------------- | ----------------------------------------------- |
| **[VLM](examples/vlm/)**            | Geometry3K and CLEVR Count 70K visual reasoning with GRPO | -                                               |
| **[VLM on NPU](examples/vlm_npu/)** | VLM training on Huawei NPU hardware                       | [Benchmark Results](examples/vlm_npu/README.md) |

### Alignment & Infrastructure

| Task                                            | Description                                           | Performance                                       |
| ----------------------------------------------- | ----------------------------------------------------- | ------------------------------------------------- |
| **[RLHF Reward Modeling](examples/alignment/)** | Bradley-Terry reward modeling on Anthropic HH-RLHF    | [Training Curve](examples/alignment/rw_curve.png) |
| **[SkyPilot Deployment](examples/skypilot/)**   | Cloud deployment with SkyPilot (GCP, AWS, Kubernetes) | [Screenshots](examples/skypilot/README.md)        |

## 🔧 Support Matrix

### 🧠 Algorithms

All RL algorithms support both asynchronous and synchronous versions by setting
`max_head_offpolicyness=0`. See [Asynchronous RL Guide](docs/algorithms/async.md).

| Algorithm                | Documentation                             | Paper                                          | Configuration                                                |
| ------------------------ | ----------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------ |
| **GRPO**                 | [📖 Docs](docs/algorithms/grpo_series.md) | [📄 Paper](https://arxiv.org/pdf/2402.03300)   | [🔗 GSM8K Example](examples/math/gsm8k_grpo.yaml)            |
| **GSPO**                 | [📖 Docs](docs/algorithms/grpo_series.md) | [📄 Paper](https://arxiv.org/abs/2507.18071)   | [🔗 GSM8K Example](examples/math/gsm8k_gspo.yaml)            |
| **PPO**                  | [📖 Docs](docs/algorithms/grpo_series.md) | [📄 Paper](https://arxiv.org/pdf/2203.02155)   | [🔗 GSM8K Example](examples/math/gsm8k_ppo.yaml)             |
| **DAPO**                 | [📖 Docs](docs/algorithms/grpo_series.md) | [📄 Paper](https://arxiv.org/abs/2503.14476)   | [🔗 GSM8K Example](examples/math/gsm8k_dapo_dynamic_bs.yaml) |
| **LitePPO**              | [📖 Docs](docs/algorithms/grpo_series.md) | [📄 Paper](https://arxiv.org/abs/2508.08221)   | [🔗 GSM8K Example](examples/math/gsm8k_liteppo.yaml)         |
| **Dr.GRPO**              | [📖 Docs](docs/algorithms/grpo_series.md) | [📄 Paper](https://arxiv.org/abs/2503.20783)   | [🔗 GSM8K Example](examples/math/gsm8k_drgrpo.yaml)          |
| **REINFORCE++**          | -                                         | [📄 Paper](https://arxiv.org/pdf/2501.03262)   | [🔗 GSM8K Example](examples/math/gsm8k_reinforce.yaml)       |
| **RLOO**                 | [📖 Docs](docs/algorithms/grpo_series.md) | [📄 Paper](https://arxiv.org/pdf/2402.14740v1) | [🔗 GSM8K Example](examples/math/gsm8k_rloo.yaml)            |
| **SAPO**                 | [📖 Docs](docs/algorithms/grpo_series.md) | [📄 Paper](https://arxiv.org/abs/2511.20347)   | [🔗 GSM8K Example](examples/math/gsm8k_sapo.yaml)            |
| **M2PO**                 | [📖 Docs](docs/algorithms/m2po.md)        | [📄 Paper](https://arxiv.org/abs/2510.01161)   | [🔗 GSM8K Example](examples/math/gsm8k_m2po.yaml)            |
| **RLHF Reward Modeling** | -                                         | -                                              | [🔗 RLHF Example](examples/alignment/)                       |
| **SFT**                  | -                                         | -                                              | [🔗 GSM8K Example](examples/math/gsm8k_sft.py)               |

### Models

| Model Family               | Megatron | PyTorch FSDP | PyTorch Archon | Notes                                                    |
| -------------------------- | -------- | ------------ | -------------- | -------------------------------------------------------- |
| **Qwen2/3**                | ✅       | ✅           | ✅             | -                                                        |
| **Qwen3-MoE**              | ✅       | ✅           | ✅             | -                                                        |
| **Qwen2.5-VL**             | ❌       | ✅           | ❌             | Vision-language model                                    |
| **Qwen3-VL**               | ❌       | ✅           | ❌             | Vision-language model                                    |
| **Gemma 3**                | ❌       | ✅           | ❌             | Vision-language model                                    |
| **Other Hugging Face LLM** | ❌       | ✅           | ❌             | Compatibility depending on the version of `transformers` |

Check the [AI Coding Assistant Guide](docs/reference/ai_assisted_dev.md) and
[Archon Reference](docs/tutorial/archon.md) for how to integrate new models into AReaL.

### Training Backends

| Backend            | DP          | Tensor Parallel | Sequence Parallel within TP | Context Parallel | Pipeline Parallel | Expert Parallel | 1D Sequence Packing | LoRA |
| ------------------ | ----------- | --------------- | --------------------------- | ---------------- | ----------------- | --------------- | ------------------- | ---- |
| **Megatron**       | ✅ (ZeRO-1) | ✅              | ✅                          | ✅               | ✅                | ✅              | ✅                  | ❌   |
| **PyTorch FSDP**   | ✅ (FSDP2)  | ✅              | ✅                          | ✅               | ❌                | ❌              | ✅                  | ✅   |
| **PyTorch Archon** | ✅ (FSDP2)  | ✅              | ✅                          | ✅               | ✅                | ✅              | ✅                  | ❌   |

### Inference Backends

| Backend    | Tensor Parallel | Context Parallel | Pipeline Parallel | Data Parallel Attention | Expert Parallel |
| ---------- | --------------- | ---------------- | ----------------- | ----------------------- | --------------- |
| **vLLM**   | ✅              | ❓               | ✅                | ❓                      | ❓              |
| **SGLang** | ✅              | ❌               | ❌                | ✅                      | ✅              |

## 📖 Resources

### Tutorial

- [Installation](https://inclusionai.github.io/AReaL/tutorial/installation.html)
- [Quickstart](https://inclusionai.github.io/AReaL/tutorial/quickstart.html)
- [Agentic RL](https://inclusionai.github.io/AReaL/tutorial/agentic_rl.html)
- [Evaluation](https://inclusionai.github.io/AReaL/tutorial/eval.html)
- [Large MoE with Megatron](https://inclusionai.github.io/AReaL/tutorial/megatron.html)
- [Large MoE with PyTorch Archon](https://inclusionai.github.io/AReaL/tutorial/archon.html)

### Code Walkthrough

- [Running GRPO on GSM8K dataset](https://inclusionai.github.io/AReaL/tutorial/gsm8k_grpo.html)

### Best Practices

- [Improving Algorithm Performance](https://inclusionai.github.io/AReaL/best_practices/algo_perf.html)
- [Agent Workflow Best Practices](https://inclusionai.github.io/AReaL/best_practices/workflow.html)
- [Debugging](https://inclusionai.github.io/AReaL/best_practices/debugging.html)
- [Handling OOM Issues](https://inclusionai.github.io/AReaL/best_practices/handling_oom.html)
- [Performance Profiling](https://inclusionai.github.io/AReaL/best_practices/perf_profiling.html)

### Customization

- [Customize Dataset](https://inclusionai.github.io/AReaL/customization/dataset.html)
- [Customize Agentic/RVLR Rollout Workflows](https://inclusionai.github.io/AReaL/customization/agent.html)

### Algorithms

- [Asynchronous RL Explained](https://inclusionai.github.io/AReaL/algorithms/async.html)
- [PPO, GRPO, and Related Algorithms](https://inclusionai.github.io/AReaL/algorithms/grpo_series.html)
- [M2PO](https://inclusionai.github.io/AReaL/algorithms/m2po.html)

### Reference

- [CLI Configurations](https://inclusionai.github.io/AReaL/cli_reference.html)
- [Checkpointing](https://inclusionai.github.io/AReaL/reference/checkpointing.html)
- [Metrics Tracking](https://inclusionai.github.io/AReaL/reference/metrics_tracking.html)
- [Allocation Mode](https://inclusionai.github.io/AReaL/reference/alloc_mode.html)
- [Rollout Workflow](https://inclusionai.github.io/AReaL/reference/rollout_workflow.html)
- [Agent Workflow](https://inclusionai.github.io/AReaL/reference/agent_workflow.html)
- [AI-Assisted Development](https://inclusionai.github.io/AReaL/reference/ai_assisted_dev.html)

## 🤝 Contributing

We warmly welcome contributions from the community! Whether you're fixing bugs, adding
features, improving documentation, or helping others, your contribution is valued.
Please check our **[Contributing Guide](CONTRIBUTING.md)** for detailed information.

```bash
# Fork and clone the repository
git clone https://github.com/YOUR-USERNAME/AReaL
cd AReaL

# Install uv and sync dependencies
pip install uv
# Use `--extra cuda` on Linux with CUDA for full functionality
uv sync --extra cuda --group dev
# Or without CUDA support
# uv sync --group dev

# Set up pre-commit hooks for automatic formatting
pre-commit install

# Make changes
git checkout -b feat/gpt-o5
git add .
# `git commit` will automatically format your file
git commit -m "Implement gpt-o5 training loop"
git push
```

## 🗺️ Future Roadmap

- **[Full Roadmap](ROADMAP.md)**
- **[2025 Q4 Roadmap](https://github.com/inclusionAI/AReaL/issues/542)**

AReaL is under active development with planned minor releases weekly and major releases
monthly. We warmly welcome community engagement and contributions. We are also
**actively hiring interns and full-time employees** with open positions in both the US
and China.

## 🙏 Acknowledgments

We gratefully acknowledge that major contributors are from the AReaL Team at the
Institute for Interdisciplinary Information Sciences (IIIS), Tsinghua University and Ant
Group.

We have also received invaluable assistance from the following groups (listed
alphabetically):

- The Data Intelligence Lab at Ant Research for their data support

- @HwVanICI for support on vLLM, LoRA, NPU integration, and more

- The [Relaxed System Lab](https://github.com/Relaxed-System-Lab) at HKUST for seamless
  collaboration on numerous system-related aspects

- The [SGLang team](https://github.com/sgl-project/sglang) for supporting custom weight
  update features and their contributions during AReaL-lite development

- The Super Computing Technology (SCT) team at Ant Group for their expertise in
  large-scale cluster operations and maintenance

- Special thanks to @Lyken17 for providing valuable suggestions throughout the API
  design process

We also deeply appreciate all pioneering work from the community, particularly the
[ReaLHF](https://github.com/openpsi-project/ReaLHF) project from OpenPsi Inc. and other
outstanding projects, including but not limited to
[DeepScaleR](https://github.com/agentica-project/deepscaler),
[Open-Reasoner-Zero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/tree/main),
[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF),
[VeRL](https://github.com/volcengine/verl),
[SGLang](https://github.com/sgl-project/sglang), [QwQ](https://github.com/QwenLM/QwQ),
[Light-R1](https://github.com/Qihoo360/Light-R1), and
[DAPO](https://github.com/BytedTsinghua-SIA/DAPO).

## 📄 Citation

```bibtex
@inproceedings{mei2025real,
  author       = {Mei, Zhiyu and Fu, Wei and Li, Kaiwei and Wang, Guangju and Zhang, Huanchen and Wu, Yi},
  title        = {ReaL: Efficient RLHF Training of Large Language Models with Parameter Reallocation},
  booktitle    = {Proceedings of the Eighth Conference on Machine Learning and Systems,
                  MLSys 2025, Santa Clara, CA, USA, May 12-15, 2025},
  publisher    = {mlsys.org},
  year         = {2025},
}
```

```bibtex
@misc{fu2025areal,
      title={AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning},
      author={Wei Fu and Jiaxuan Gao and Xujie Shen and Chen Zhu and Zhiyu Mei and Chuyi He and Shusheng Xu and Guo Wei and Jun Mei and Jiashu Wang and Tongkai Yang and Binhang Yuan and Yi Wu},
      year={2025},
      eprint={2505.24298},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.24298},
}
```
