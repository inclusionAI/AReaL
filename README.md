<h1 align="center">
<em>AReaL</em>: A Large-Scale Asynchronous Reinforcement Learning System
</h1>

<p align="center">
| <a href="https://arxiv.org/pdf/2505.24298"><b>Paper</b></a> | <a href="https://inclusionai.github.io/AReaL/"><b>Documentation</b></a> | <a href="https://deepwiki.com/inclusionAI/AReaL"><b>Ask DeepWiki</b></a> | <a href="https://huggingface.co/collections/inclusionAI/areal-boba-2-683f0e819ccb7bb2e1b2f2d5"><b>🤗 Models & Data</b></a> |
<a href="./assets/wechat_qrcode.png" target="_blank"><img src="./assets/wechat_icon.png" width="20" style="vertical-align: middle;"> <b>WeChat (微信) Group</b></a> |
</p>

<img align="right" alt="ReaL" src="/assets/logo.png" width="20%">

AReaL is an open-source **fully asynchronous** reinforcement learning training system
for large **reasoning and agentic models**, developed by the AReaL Team at Ant Group.
Built upon the open-source project [ReaLHF](https://github.com/openpsi-project/ReaLHF),
we are fully committed to open-source principles by providing training details, data,
and infrastructure required to reproduce our results along with the models themselves.
AReaL aims to help everyone build their own AI agents easily and affordably. Our team
loves milk tea because it's delicious, customizable, and affordable. We hope you enjoy
our project just as you enjoy real-world milk tea (cheers).

**AReaL Highlights**

- ⚡ **Flexibility**: Seamless customization for
  [multi-turn agentic rollout](https://inclusionai.github.io/AReaL/customization/agent.html)
  workflows within a single file, and smooth integration with other agentic tooling
  frameworks.
- 🚀 **Scalability**: Through algorithm-system co-design, AReaL delivers **stable** fully
  asynchronous RL training with **industry-leading speed**. AReaL seamlessly adapts to
  diverse computational environments, scaling from a single node to 1,000+ GPUs.
- 🔪 **Cutting-Edge Performance**: AReaL produces state-of-the-art
  [math](/blog/AReaL_v0_2.md), [coding](/blog/AReaL_v0_3.md), and
  [search agents](https://github.com/inclusionAI/ASearcher) with exceptional
  capabilities.

## 📰 News

**\[2025/08/30\]** Introducing ASearcher, a state-of-the-art search agent built with
AReaL's end-to-end asynchronous RL training. Check out the
[paper](https://arxiv.org/pdf/2508.07976) and the
[open-source repository](https://github.com/inclusionAI/ASearcher)!

**\[2025/07/31\] (AReaL-lite)** We introduce AReaL-lite, a **lightweight** version of
AReaL designed specifically for AI researchers and rapid prototyping. AReaL-lite
features an **algorithm-first** API design that prioritizes ease of use and algorithm
development, while natively supporting **fully asynchronous agentic RL**. With 80% fewer
lines of code, AReaL-lite maintains 90% of AReaL's performance and core functionality.
Check out [our AReaL-lite design documentation](/areal/README.md) and
[the quickstart guide](https://inclusionai.github.io/AReaL/tutorial/quickstart.html) to
begin your journey with **AReaL-lite**!

<details>
<summary><b>📋 Previous Releases</b></summary>

**\[2025/06/03\] (v0.3, boba²)** We release **boba²** (double-boba) for fully
asynchronous RL training, which achieves **2.77× speedup while delivering comparable or
superior training performance** compared to synchronous systems. Furthermore,
asynchronous RL significantly simplifies multi-turn agentic RL training setup! Check out
[our v0.3 overview blog](/blog/AReaL_v0_3.md) and the
[research paper](https://arxiv.org/pdf/2505.24298).

**\[2025/03/31\] (v0.2, boba)** Introducing our milestone release—boba! Please call it
A-ReaL-boba! This release features significantly faster training with SGLang support and
state-of-the-art 7B and 32B models for mathematical reasoning. Check out our
[v0.2 technical blog](/blog/AReaL_v0_2.md).

**\[2025/02/24\] (v0.1)** Our initial release includes reproducible results for 1.5B and
7B Large Reasoning Models (LRMs). Check out our
[v0.1 technical blog](/blog/AReaL_v0_1.md).

</details>

## 📚 Examples

| Task                                             | Description                                                                          | Performance                                                                       |
| ------------------------------------------------ | ------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------- |
| **[Math](examples/math/)**                       | Mathematical problem solving (SFT, GRPO, or PPO)                                     | TBA                                                                               |
| **[Multi-Turn Math](examples/multi-turn-math/)** | Iterative mathematical problem solving with self-correction                          | [Training Curve](examples/multi-turn-math/reward_curve.png)                       |
| **[LoRA Math](examples/lora/)**                  | Math Agent Trained With LoRA                                                         | TBA                                                                               |
| **[VLM Math](examples/vlm/)**                    | CLEVR visual counting tasks                                                          | TBA                                                                               |
| **[Reasoning](examples/countdown/)**             | Countdown numbers game with custom rewards                                           | [Training Curve](/examples/countdown/countdown_training_curve.png)                |
| **[Search Agent](examples/search-agent/)**       | An agent with end-to-end reasoning, search, browsing, and summarization capabilities | [ASearcher Repo](https://github.com/inclusionAI/ASearcher)                        |
| **[Tool-Integrated Reasoning](examples/tir/)**   | An agent that can invoke tools during reasoning                                      | [TIR Example](https://github.com/inclusionAI/AReaL/tree/main/examples/tir)        |
| **[RLHF](examples/alignment/)**                  | RLHF for LLM Alignment                                                               | [RLHF Example](https://github.com/inclusionAI/AReaL/tree/main/examples/alignment) |

## 🔧 Support Matrix

### 🧠 Algorithms

| Algorithm                | Documentation                         | Paper                                          | Configuration                                                |
| ------------------------ | ------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------ |
| **GRPO**                 | [📖 Docs](docs/algorithms/grpo.md)    | [📄 Paper](https://arxiv.org/pdf/2402.03300)   | [🔗 GSM8K Example](examples/math/gsm8k_grpo.yaml)            |
| **PPO**                  | -                                     | [📄 Paper](https://arxiv.org/pdf/2203.02155)   | [🔗 GSM8K Example](examples/math/gsm8k_ppo.yaml)             |
| **DAPO**                 | [📖 Docs](docs/algorithms/dapo.md)    | [📄 Paper](https://arxiv.org/abs/2503.14476)   | [🔗 GSM8K Example](examples/experimental/dapo/gsm8k_dapo.py) |
| **LitePPO**              | [📖 Docs](docs/algorithms/litePPO.md) | [📄 Paper](https://arxiv.org/abs/2508.08221)   | -                                                            |
| **Dr.GRPO**              | [📖 Docs](docs/algorithms/dr.GRPO.md) | [📄 Paper](https://arxiv.org/abs/2503.20783)   | -                                                            |
| **REINFORCE++**          | -                                     | [📄 Paper](https://arxiv.org/pdf/2501.03262)   | [🔗 GSM8K Example](examples/math/gsm8k_reinforce.yaml)       |
| **RLOO**                 | [📖 Docs](docs/algorithms/rloo.md)    | [📄 Paper](https://arxiv.org/pdf/2402.14740v1) | [🔗 GSM8K Example](examples/math/gsm8k_rloo.yaml)            |
| **RLHF Reward Modeling** | -                                     | -                                              | [🔗 RLHF Example](examples/alignment/)                       |
| **SFT**                  | -                                     | -                                              | [🔗 GSM8K Example](examples/math/gsm8k_sft.py)               |

### Models

| Model Family               | Megatron | PyTorch FSDP | Notes                                                    |
| -------------------------- | -------- | ------------ | -------------------------------------------------------- |
| **Qwen2/3**                | ✅       | ✅           | -                                                        |
| **Qwen3-MoE**              | ✅       | ✅           | -                                                        |
| **Qwen2.5-VL**             | ❌       | ✅           | Vision-language model                                    |
| **Gemma 3**                | ❌       | ✅           | Vision-language model                                    |
| **Other Hugging Face LLM** | ❌       | ✅           | Compatibility depending on the version of `transformers` |

### Training Backends

| Backend          | DP          | Tensor Parallel | Sequence Parallel within TP | Context Parallel | Pipeline Parallel | Expert Parallel | 1D Sequence Packing | LoRA |
| ---------------- | ----------- | --------------- | --------------------------- | ---------------- | ----------------- | --------------- | ------------------- | ---- |
| **Megatron**     | ✅ (ZeRO-1) | ✅              | ✅                          | ✅               | ✅                | ✅              | ✅                  | ❌   |
| **PyTorch FSDP** | ✅ (FSDP2)  | ✅              | ✅                          | ✅               | ❌                | ❌              | ✅                  | ✅   |

### Inference Backends

| Backend    | Tensor Parallel | Context Parallel | Pipeline Parallel | Data Parallel Attention | Expert Parallel |
| ---------- | --------------- | ---------------- | ----------------- | ----------------------- | --------------- |
| **vLLM**   | ✅              | ❓               | ❓                | ❓                      | ❓              |
| **SGLang** | ✅              | ❌               | ❌                | ✅                      | ✅              |

## 🚀 Getting Started

Our training scripts automatically download the required dataset (openai/gsm8k) and
model (Qwen/Qwen2-1.5B-Instruct). To run on a single node:

```bash
python3 -m areal.launcher.local \
  examples/math/gsm8k_grpo.py \
  --config examples/math/gsm8k_grpo.yaml
```

To run on a Ray cluster with 2 nodes and 8 GPUs per node (remember to update paths in
the YAML file to point to your shared storage):

```bash
python3 -m areal.launcher.ray \
  examples/math/gsm8k_grpo.py \
  --config examples/math/gsm8k_grpo.yaml \
  cluster.n_nodes=2 \
  cluster.n_gpus_per_node=8
```

For comprehensive setup instructions, see
[our quickstart guide](https://inclusionai.github.io/AReaL/tutorial/quickstart.html).

## 📖 Resources

- [Installation](https://inclusionai.github.io/AReaL/tutorial/installation.html)
- [Quickstart](https://inclusionai.github.io/AReaL/tutorial/quickstart.html)
- [CLI Configurations](https://inclusionai.github.io/AReaL/cli_reference.html)
- [Debugging Best Practices](https://inclusionai.github.io/AReaL/best_practices/debugging.html)
- [Handling OOM Issues](https://inclusionai.github.io/AReaL/best_practices/handling_oom.html)
- [Contributing](https://inclusionai.github.io/AReaL/contrib.html)

### Code Walkthrough

- [Running GRPO on GSM8K dataset with AReaL-lite](https://inclusionai.github.io/AReaL/lite/gsm8k_grpo.html)

### Customization

- [Customize dataset with AReaL-lite](https://inclusionai.github.io/AReaL/customization/dataset.html)
- [Customize Agentic/RVLR rollout workflows with AReaL-lite](https://inclusionai.github.io/AReaL/customization/agent.html)
- [Customize algorithms with AReaL-lite](https://inclusionai.github.io/AReaL/customization/algorithm.html)

## 🗺️ Future Roadmap

- [2025 Q3 Roadmap](https://github.com/inclusionAI/AReaL/issues/257)

AReaL is under active development with planned minor releases weekly and major releases
monthly. We warmly welcome community engagement and contributions. We are also
**actively hiring interns and full-time employees** with open positions in both the US
and China.

## 🙏 Acknowledgments

We gratefully acknowledge that major contributors are from the AReaL Team at Ant Group
and the Institute for Interdisciplinary Information Sciences, Tsinghua University.

We have also received invaluable assistance from the following groups (listed
alphabetically):

- The Data Intelligence Lab at Ant Research for their data support

- The [Relaxed System Lab](https://github.com/Relaxed-System-Lab) from HKUST for
  seamless collaboration on numerous system-related aspects

- The [SGLang team](https://github.com/sgl-project/sglang) for supporting custom weight
  update features and their contributions during AReaL-lite development

- The Super Computing Technology (SCT) team at Ant Group for their expertise in
  large-scale cluster operations and maintenance

- Special thanks to @Lyken17 for providing valuable suggestions throughout our
  development process

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
