<h1 align="center">
<em>AReaL</em>: Ant Reasoning Reinforcement Learning for LLMs
</h1>

<p align="center">
| <a href="https://arxiv.org/pdf/2505.24298"><b>Paper</b></a> | <a href="https://inclusionai.github.io/AReaL/"><b>Documentation</b></a> | <a href="https://deepwiki.com/inclusionAI/AReaL"><b>Ask DeepWiki</b></a> | <a href="https://huggingface.co/collections/inclusionAI/areal-boba-2-683f0e819ccb7bb2e1b2f2d5"><b>🤗 Models & Data</b></a> |
<a href="./assets/wechat_qrcode.png" target="_blank"><img src="./assets/wechat_icon.png" width="20" style="vertical-align: middle;"> <b>WeChat (微信) Group</b></a> |
</p>

<img align="right" alt="ReaL" src="/assets/logo.png" width="20%">

AReaL is an open-source **fully asynchronous reinforcement learning training system**
for large reasoning and agentic models developed by **the AReaL Team, Ant Group**. Built
upon the open-source project [ReaLHF](https://github.com/openpsi-project/ReaLHF), we are
fully committed to open-source by providing training details, data, and infrastructure
required to reproduce results along with the model itself. AReaL aims to help everyone
build their own AI agents easily and affordably. Our team loves milk tea because it's
delicious, customizable, and affordable. We hope you enjoy our project just like how you
enjoy real-world milk tea (cheers).

**AReaL Highlights**

- ⚡ **Flexibility**: Smooth customization for multi-turn agentic rollout workflows
  within a single file, and smooth integration with other agentic tooling framework.
  Check [here](https://inclusionai.github.io/AReaL/customization/agent.html) for rollout
  customization and
  [here](https://inclusionai.github.io/AReaL/customization/algorithm.html) for algorithm
  customization.
- 🚀 **Scalability**: With algorithm-system co-design, AReaL supports fully asynchronous
  RL for **the fastest training speed**! AReaL can seamlessly adapt to different
  computational resource settings, ranging from a single node to 1K GPUs.
- 🔪 **Cutting-Edge Performance:** AReaL can produce [math](/blog/AReaL_v0_2.md),
  [coding](/blog/AReaL_v0_3.md), and
  [search agents](https://github.com/inclusionAI/ASearcher) with cutting-edge
  capabilities.

## News

**\[2025/08/30\]** Introducing ASearcher, a state-of-the-art search agent built with
AReaL's end-to-end asynchronous RL training. Checkout the
[paper](https://arxiv.org/pdf/2508.07976) and the
[open-source repo](https://github.com/inclusionAI/ASearcher)!

<!--> TODO: fold the following news into a tab <!-->

**\[2025/07/31\] (AReaL-lite)** We introduce AReaL-lite, a **light-weight** version of
AReaL designed specifically for AI researchers and rapid prototyping. AReaL-lite
features an **algorithm-first** API design that prioritizes ease of use and algorithm
development, while inherently supporting **fully asynchronous agentic RL**. With 80%
fewer lines of code, AReaL-lite maintains 90% of AReaL's high performance and core
functionality. Check out [our AReaL-lite design doc](/areal/README.md) and
[the quickstart guide](https://inclusionai.github.io/AReaL/tutorial/quickstart.html) to
begin your journey with **AReaL-lite**!

**\[2025/06/03\] (v0.3, boba²)** We release **boba²** (double-boba) for fully
asynchronous RL training, which achieves a **2.77x speedup while obtaining on-par or
even better training performance** compared to synchronous systems. Moreover,
asynchronous RL makes it extremely easy to set up multi-turn agentic RL training! Check
out [our v0.3 overview blog](/blog/AReaL_v0_3.md) and the
[research paper](https://arxiv.org/pdf/2505.24298).

**\[2025/03/31\] (v0.2, boba)** Here comes our next milestone release - boba! Please
call it A-ReaL-boba! This release includes much faster training with SGLang support and
SOTA 7B and 32B models on math reasoning. Check our
[v0.2 technical blog](/blog/AReaL_v0_2.md).

**\[2025/02/24\] (v0.1)** Our initial release includes reproducible results for 1.5B and
7B LRMs. Check our [v0.1 technical blog](/blog/AReaL_v0_1.md).

## Getting Started

Our training scripts will automatically download the dataset (openai/gsm8k) and model
(Qwen/Qwen2-1.5B-Instruct). On a single node, run:

```bash
python3 -m areal.launcher.local \
  examples/math/gsm8k_grpo.py \
  --config examples/math/gsm8k_grpo.yaml
```

On a Ray cluster with 2 nodes & 8 GPUs each node, runs (remember to change paths in the
YAML file to your own shared storage):

```bash
python3 -m areal.launcher.ray \
  examples/math/gsm8k_grpo.py \
  --config examples/math/gsm8k_grpo.yaml \
  cluster.n_nodes=2 \
  cluster.n_gpus_per_node=8
```

For more details, please check out
[our quickstart guide](https://inclusionai.github.io/AReaL/tutorial/quickstart.html)!

## Resources

- [Documentation](https://inclusionai.github.io/AReaL/)
- [Contributing](https://inclusionai.github.io/AReaL/contrib.html)

### Quickstart

- [Installation](https://inclusionai.github.io/AReaL/tutorial/installation.html)
- [AReaL-lite Quickstart](https://inclusionai.github.io/AReaL/tutorial/quickstart.html)

### Code Walkthrough

- [Running GRPO on GSM8K dataset with AReaL-lite](https://inclusionai.github.io/AReaL/lite/gsm8k_grpo.html)

### Customization

- [Customize dataset with AReaL-lite](https://inclusionai.github.io/AReaL/customization/dataset.html)
- [Customize Agentic/RVLR rollout workflows with AReaL-lite](https://inclusionai.github.io/AReaL/customization/agent.html)
- [Customize algorithms with AReaL-lite](https://inclusionai.github.io/AReaL/customization/algorithm.html)

### Advanced Usage

- [Debugging Best Practice](https://inclusionai.github.io/AReaL/best_practices/debugging.html)
- [Handling OOM Issues](https://inclusionai.github.io/AReaL/best_practices/handling_oom.html)

## Future Plan

AReaL is under active development. We plan to have minor releases weekly and major
releases monthly. Community engagement and contributions are extremely welcome. We are
also **hiring interns and full-time employees** with open positions in both the US and
China.

For the research and development plan already in place, please see the following list:

### System Development

- [x] Support for SGLang
- [x] RL training with coding problems
- [x] Asynchronous generation and RL training
- [ ] Optimizations for distributed training: expert parallel for MOE and zero-bubble
  pipelining
- [x] RL for vision-language models (VLM)
- [x] Multi-turn agentic RL
- [x] Function calling and tool use

### Algorithm Development

- [x] RL training recipes for 1.5B and 7B models
- [x] A complete RL training recipe for 32B models
- [ ] Sample-efficient multi-task RL algorithms
- [x] Agentic capabilities with end-to-end RL
- [ ] Stable RL training for larger MOE models

## Acknowledgement

We would like to note that major contributors are from the AReaL Team at Ant Group and
the Institute for Interdisciplinary Information Sciences, Tsinghua University.

Our team has also received invaluable assistance from the following groups (listed in
alphabetical order):

- The Data Intelligence Lab at Ant Research for their data support

- The [Relaxed System Lab](https://github.com/Relaxed-System-Lab) from HKUST for
  seamless cooperation on many system-related aspects

- The [SGLang team](https://github.com/sgl-project/sglang) for supporting customized
  features for updating weights and their contributions during the development of
  AReaL-lite

- The Super Computing Technology (SCT) team at Ant Group, particularly for their
  expertise in large-scale cluster operations and maintenance

- Special thanks to @Lyken17 for providing suggestions across our development process

We also appreciate all the pioneering works from the community, particularly the
[ReaLHF](https://github.com/openpsi-project/ReaLHF) project from OpenPsi Inc. and other
projects, including but not limited to
[DeepScaleR](https://github.com/agentica-project/deepscaler),
[Open-Reasoner-Zero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/tree/main),
[OpenRLHF](https://github.com/OpenRLHF/OpenRLHF),
[VeRL](https://github.com/volcengine/verl),
[SGLang](https://github.com/sgl-project/sglang), [QwQ](https://github.com/QwenLM/QwQ),
[Light-R1](https://github.com/Qihoo360/Light-R1) and
[DAPO](https://github.com/BytedTsinghua-SIA/DAPO).

## Citation

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
