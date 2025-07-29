<h1 align="center">
<em>AReaL</em>: Ant Reasoning Reinforcement Learning for LLMs
</h1>

<p align="center">
| <a href="https://arxiv.org/pdf/2505.24298"><b>Paper</b></a> | <a href="https://inclusionai.github.io/AReaL/"><b>Documentation</b></a> | <a href="https://deepwiki.com/inclusionAI/AReaL"><b>Ask DeepWiki</b></a> | <a href="https://huggingface.co/collections/inclusionAI/areal-boba-2-683f0e819ccb7bb2e1b2f2d5"><b>🤗 Models & Data</b></a> |
<a href="./assets/wechat_qrcode.png" target="_blank"><b>WeChat Group</b></a> |
</p>

<img align="right" alt="ReaL" src="/assets/logo.png" width="20%">

AReaL (Ant Reasoning RL) is an open-source **fully asynchronous reinforcement learning
training system** for large reasoning models developed at **the RL Lab, Ant Research**.
Built upon the open-source project [RealHF](https://github.com/openpsi-project/ReaLHF),
we are fully committed to open-source by providing training details, data, and
infrastructure required to reproduce results along with the model itself. AReaL aims to
help everyone build their own AI agents easily and affordably. Our team loves milk tea
because it's delicious, customizable, and affordable. We hope you enjoy our project just
like how you enjoy real-world milk tea (cheers).

**AReaL Highlights**

- 🔥 **Asynchronous RL**: With algorithm-system co-design, AReaL supports fully
  asynchronous RL for **the fastest training**! Experimental support for multi-turn
  agentic RL is also provided.
- ⚡ **\[NEW\] Light-weight & AI-centric:** In our new release AReaLite, we deliver
  **90%** of AReaL functionalities with only **20%** # lines of code! AReaLite also
  follows an **AI-centric** design that make users build their own **agentic** and
  **RLVR** training workflows with much less effort.
- 🛠️ **Open & Reproducible**: We continuously release _all code, datasets, and training
  recipes_ for RL training of LLMs.
- 🚀 **Scalability**: AReaL can seamlessly adapt to different computational resource
  settings, ranging from a single node to 1K GPUs.
- 🔪 **Cutting-Edge Performance:** AReaL can produce models with cutting-edge reasoning
  capabilities in math and coding. We are also actively working on agentic tasks.

## News

**\[2025/07/31\] (v0.4, AReaLite)** We introduce **AReaLite**, a **light-weight**
version of AReaL with an **AI-centric** API design that inherently supports fully
asynchronous **agentic RL**. Check out [our AReaLite Design Doc](/arealite/README.md)
and [the quickstart guide](/docs/tutorial/quickstart.md) to begin your journey with
**AReaLite**!

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

## Release Highlights

New highlights in AReaLite:

- Follows an *AI-centric* API design instead of the *system-centric* architecture in old
  AReaL, which make it easier for AI researchers to adopt, understand, and develop
  effectively and efficiently. To learn more about the design principles of AReaL,
  please read [AReaLite Design Doc](/arealite/README.md)!

- A much more *light-weight* codebase compared to old AReaL codebase with only **20%** #
  lines of code, with a detailed [code walkthrough](/docs/arealite/gsm8k_grpo.md) on an
  GRPO-on-GSM8K example. Save your time & efforts for code reading!

- Smoother customization for your own **algorithms** and **agentic & RLVR rollout** RL
  within a single file! Check [here](/docs/customization/agent.md) for agent & RLVR
  customization and [here](/docs/customization/algorithm.md) for algorithm
  customization.

Good old stuff from AReaL:

- High performance and scalability with fully asynchronous RL training. Check our
  [boba² (v0.3) blog](/blog/AReaL_v0_3.md) for details.

- A single command line to launch an experiment, no matter on a single node or a
  large-scale distributed cluster.

Now, let us run an example experiment with AReaLite following the quickstart guide
below!

## Getting Started with AReaLite

Our training scripts will automatically download the dataset (openai/gsm8k) and model
(Qwen/Qwen2-1.5B-Instruct). On a single node, runs:

```
python3 -m arealite.launcher.local examples/arealite/gsm8k_grpo.py --config examples/arealite/configs/gsm8k_grpo.yaml
```

On a ray cluster with 2 nodes & 8 GPUs each node, runs (Remember to change paths in YAML
file to your own shared storage):

```
python3 -m arealite.launcher.ray examples/arealite/gsm8k_grpo.py --config examples/arealite/configs/gsm8k_grpo.yaml \
  cluster.n_nodes=2 \
  cluster.n_gpus_per_node=8
```

<!-- TBD: not finished yet -->

Evaluation (on a single node):

```
python3 -m arealite.launcher.local examples/arealite/eval.py --config examples/arealite/configs/eval.yaml
```

## Switching from AReaL to AReaLite

We also provide a convenient script to convert your AReaL YAML config into AReaLite
config in one command line. First you need to locate your AReaL config either modified
from files from `examples` folder, or generated when you run your experiments in
`<fileroot>/<expr_name>/<trial_name>` folder. Runs:

```
python3 examples/arealite/convert_config.py -f <config_path> -o <output_path>
```

You can easily convert your config to

## Resources

- [Documentation](https://inclusionai.github.io/AReaL/)
- [Contributing](https://inclusionai.github.io/AReaL/contrib.html)

### Quickstart

- [Installation](https://inclusionai.github.io/AReaL/tutorial/installation.html)
- [AReaLite Quickstart](/docs/tutorial/quickstart.md)

### Code Walkthrough

- [Running GRPO on GSM8K dataset with AReaLite](/docs/arealite/gsm8k_grpo.md)

### Customization

- [Customize dataset with AReaLite](../customization/dataset.md)
- [Customize Agentic/RVLR rollout workflows with AReaLite](../customization/agent.md)
- [Customize algorithms with AReaLite](../customization/algorithm.md)

### AReaL Legacy

For old AReaL documentation, check the legacy sections in our
[Documentation](https://inclusionai.github.io/AReaL/). To reproduce AReaL boba & boba²
results, check our
[reproduction guide with legacy AReaL](https://inclusionai.github.io/AReaL/references/reproduce.html).

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
- [ ] RL for vision-language models (VLM)
- [x] Multi-turn agentic RL
- [ ] Function calling and tool use

### Algorithm Development

- [x] RL training recipes for 1.5B and 7B models
- [x] A complete RL training recipe for 32B models
- [ ] Sample-efficient multi-task RL algorithms
- [ ] Agentic capabilities with end-to-end RL
- [ ] Stable RL training for larger MOE models

## Acknowledgement

We would like to note that major contributors are from the RL Lab at Ant Research and
the Institute for Interdisciplinary Information Sciences, Tsinghua University.

Our team has also received invaluable assistance from the Data Intelligence Lab at Ant
Research for data support and from the Super Computing Technology (SCT) team at Ant
Group, particularly in the realm of large-scale cluster operations and maintenance.

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
