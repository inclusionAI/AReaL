<h1 align="center">
<em>AReaL</em>: Ant Reasoning Reinforcement Learning for LLMs
</h1>

<p align="center">
| <a href="https://arxiv.org/pdf/2505.24298"><b>Paper</b></a> | <a href="https://inclusionai.github.io/AReaL/"><b>Documentation</b></a> | <a href="https://deepwiki.com/inclusionAI/AReaL"><b>Ask DeepWiki</b></a> | <a href="https://huggingface.co/collections/inclusionAI/areal-boba-2-683f0e819ccb7bb2e1b2f2d5"><b>🤗 Models & Data</b></a> |
</p>

<img align="right" alt="ReaL" src="/assets/logo.png" width="20%">

AReaL (Ant Reasoning RL) is an open-sourced **fully asynchronous reinforcement learning training system** for large reasoning models developed at **the RL Lab, Ant Research**, built upon the open-source project [RealHF](https://github.com/openpsi-project/ReaLHF). We fully commit to open-source by opening training details, data and infra required to reproduce the results along with the model itself. AReaL aims to help everyone build their own AI agents easily and affordably. Our team loves milk tea as it is delicious, customizable, and affordable. We hope you all enjoy our project just like how you enjoy a real-world milk tea (cheers).

**AReaL Highlights**

+ 🔥 **[NEW] Asynchronous RL:** With algorithm-system co-design, AReaL supports fully asynchronous RL for **the fastest training**! Experimental support for multi-turn agentic RL is also provided.
+ 🛠️ **Open & Reproducible**: We will continuously release _all code, datasets, and training recipes_ for RL training LLMs.
+ 🚀 **Scalability**: AReaL can seamlessly adapt to different computational resource settings, ranging from 1 single node to 1K GPUs.
+ 🔪 **Cutting-Edge Performances:** AReaL can produce models with cutting-edge reasoning capabilities in math and coding. We are also actively working on agentic tasks.

## News

**[2025/06/03] (v0.3, boba²)** We release **boba²** (double-boba) for fully asynchronous RL training, which achieves a **2.77x speedup while obtaining on-par or even better training performance** compared to synchronous systems. Moreover, asynchronous RL makes it extremely easy to set up multi-turn agentic RL training! Check out [our v0.3 overview blog](/blog/AReaL_v0_3.md) and the [research paper](https://arxiv.org/pdf/2505.24298).

**[2025/03/31] (v0.2, Boba)** Here comes our next milestone release - Boba! Please call it A-ReaL-Boba! This release includes much accelerated training with SGLang support and SOTA 7B and 32B models on math reasoning. Check our [v0.2 technical blog](/blog/AReaL_v0_2.md).

**[2025/02/24] (v0.1)** Our initial release includes reproducible results for 1.5B and 7B LRMs. Check our [v0.1 technical blog](/blog/AReaL_v0_1.md).

## Release Highlights

In our AReaL-boba² (A-ReaL-double-boba) release, we highlight the top 3 most important features:

+ A fully asynchronous RL training pipeline with **system and RL algorithm co-design**, achieving [over 2.77x speedup](https://github.com/inclusionAI/AReaL/tree/main/benchmark) without any performance drop.
+ SOTA coding models, i.e., a 14B model with a **69.1 score on LCB-v5**. [Reproducible results](https://inclusionai.github.io/AReaL/references/reproduce.html) with fully open-sourced datasets are also provided.
+ Experimental support for **multi-turn** agentic RL training.

For the complete system design and training details, please check [our v0.3 blog](/blog/AReaL_v0_3.md) and our [research paper](about:blank) for a more comprehensive presentation of our system design.

### Overview of Asynchronous RL Training

During the synchronous RL training process, a generation step must wait until the longest sequence completes within the batch of LLM outputs. Due to the varying output lengths for LRM, a synchronous RL system suffers from massive GPU idle time, leading to training inefficiency. Some recent works ([DeepCoder](https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51), [Intellect](https://www.primeintellect.ai/blog/intellect-2)) propose to overlap a single training step with a single generation step to accelerate training. However, the largest bottleneck remains unchanged: the samples within a batch are still from the same model version, leading to waiting and GPU idle time.

![Synchronous vs One-step Overlap RL](/assets/sync_one_step_gen.png)

*Fig.1. Left: Execution timeline of a synchronous RL training. Right: Execution timeline of one-step overlap RL system.*

AReaL adopts a fully asynchronous RL training framework that completely decouples generation from training. In AReaL, LLM generation runs in a streaming manner, with each rollout worker continuously producing outputs without waiting. Meanwhile, trainer workers perform parallel model updates upon receiving training batches.

![Asynchronous RL Training](/assets/async_timeline.png)

*Fig 2. Execution timeline of our fully asynchronous RL system.*

AReaL follows a system-algorithm co-design principle: on the system side, AReaL efficiently syncs up model parameters and carefully controls the staleness of each training sample; on the algorithm side, AReaL improves the objective of PPO to make async-RL stable.

We compare the scalability of **asynchronous RL** training based on our AReaL-boba² system with **classical synchronous RL** training (we adopt the fastest open-sourced system veRL, main branch on 05/07/2025) across different model sizes and different numbers of H800 GPUs. AReaL demonstrates the much improved scaling capabilities w.r.t. training throughput. This is also partially due to that AReaL decouples training and generation, leading to much fewer GPU memory fragments. (Check the [benchmark directory](/benchmark) for detailed benchmark guide.)

![Scaling Comparison](/assets/async_scaling_vs_verl.png)

*Fig.3 The scaling trend of asynchronous RL (based on AReaL-boba2) and classical synchronous RL (based on veRL) with different model sizes. Dotted lines indicate ideal linear scaling.*

### SOTA Code Generation Model by AReaL-boba²

We use **Qwen3** as our base model. After asynchronous RL training, we achieve SOTA results on LiveCodeBench, Codeforce, and CodeContests benchmarks.

| **Model (8B)** | **LiveCodeBench v5**<br/>**(2024.10-2025.2)** | **Codeforce** | **CodeContests** |
| :---: | :---: | :---: | :---: |
| Qwen3-8B | 58.8 | 1879/96.7% | 31.4 |
| DeepSeek-R1-0528-Qwen3-8B | 58.4 | 1945/97.3% | 31.0 |
| [🤗 AReaL-boba²-8B-Open](https://huggingface.co/inclusionAI/AReaL-boba-2-8B-subset) | 62.0 | 1933/97.2% | **41.4** |
| [🤗 AReaL-boba²-8B](https://huggingface.co/inclusionAI/AReaL-boba-2-8B) | **63.0** | **1962/97.5%** | 40.8 |

| **Model (14B)** | **LiveCodeBench v5**<br/>**(2024.10-2025.2)** | **Codeforce** | **CodeContests** |
| :---: | :---: | :---: | :---: |
| Qwen3-14B | 65.4 | 1978/97.7% | 38.3 |
| DeepCoder-14B-Preview | 60.6 | 1936/95.3% | 40.1 |
| [🤗 AReaL-boba²-14B-Open](https://huggingface.co/inclusionAI/AReaL-boba-2-14B-subset) | 67.3 | 1990/97.8% | **46.2** |
| [🤗 AReal-boba²-14B](https://huggingface.co/inclusionAI/AReaL-boba-2-14B) | **69.1** | **2044/98.2%** | 46.1 |

| **Larger Models** | **LiveCodeBench v5**<br/>**(2024.10-2025.2)** | **Codeforce** | **Codecontest** |
| :---: | :---: | :---: | :---: |
| Qwen3-235B | 70.7 | 2056 | - |
| DeepSeek-R1 | 64.3 | 2029 | - |
| OpenAI-o3-mini (Medium) | 66.3 | 2036 | - |

*Table 1: Coding Task Performance Comparison. AReaL-boba²-8B/14B-Open denotes training results on open-sourced data，AReaL-boba²-8B/14B models are trained with an additional small amount of internal data could achieve SOTA performance on LiveCodeBench, Codeforce & CodeContests*

We highlight the tutorials about the following key features for synchronous training in AReaL-boba²:

+ [Streaming generation and reward computation](https://inclusionai.github.io/AReaL/developer/rollout/rollout_worker.html)
+ [Interruptible rollout](https://inclusionai.github.io/AReaL/developer/rollout/gserver.html)
+ [Data staleness control with the rollout controller](https://inclusionai.github.io/AReaL/developer/rollout/gserver.html)
+ [The adoption of decoupled PPO loss](https://inclusionai.github.io/AReaL/customization/algorithm.html)

We provide a step-by-step [code walkthrough](https://inclusionai.github.io/AReaL/developer/overview.html) and [customization guide](https://inclusionai.github.io/AReaL/customization/dataset.html) for these features and recommend users to walk through the corresponding documentation.

### RL Training for Multi-turn Agent

AReaL-boba² allows you to independently customize the [dataset](https://inclusionai.github.io/AReaL/customization/dataset.html), [rollout behavior](https://inclusionai.github.io/AReaL/customization/agent.html), and the [training algorithm](https://inclusionai.github.io/AReaL/customization/algorithm.html), without the need to modify the heavy system-level code.

In particular, we show a simple example to develop a multi-turn math agent for RL training. Please see the learning curve below and reference the [step-by-step guide](https://inclusionai.github.io/AReaL/customization/agent.html) if you want to implement your own agentic RL project.

![Multi-turn Agent Learning Curve](/assets/multiturn_reward.png)


## Getting Started

### Quick Start

Train Qwen3 1.7B locally:

```bash
bash examples/env/scripts/setup-pip-deps.sh
python3 training/main_async_ppo.py \
    n_nodes=1 n_gpus_per_node=8 \
    allocation_mode=sglang.d4p1m1+d2p2m1 \
    cluster.fileroot=/storage/testing/experiments \
    actor.type._class=qwen3 \
    actor.path=Qwen/Qwen3-1.7B \
    ref.type._class=qwen3 \
    ref.path=Qwen/Qwen3-1.7B \
    dataset.path=/path/to/dataset/boba_106k_0319.jsonl \
    dataset.train_bs_n_seqs=32 \
    group_size=8 \
    ppo.gen.max_new_tokens=4096 \
    ppo.ppo_n_minibatches=4 \
    actor_train.mb_spec.max_tokens_per_mb=32768 \
    actor_inf.mb_spec.max_tokens_per_mb=32768 \
    max_concurrent_rollouts=16 \
    max_head_offpolicyness=4
```

Evaluation:

```bash
bash examples/env/scripts/setup-eval-pip-deps.sh
cd evaluation
# Evaluate the model
python eval_and_aggregate.py \
  --model_path ${MODEL_PATH} \
  --output_path ${OUTPUT_PATH} \
  --data_names aime24,aime25 \
  --max_gen_tokens 32768 \
  --data_names codeforces,lcb_v5 \
  --prompt_type qwen3-think-pure \
  --temperature 1.0
```

### Resources

+ [Installation](https://inclusionai.github.io/AReaL/tutorial/installation.html)
+ [Quickstart](https://inclusionai.github.io/AReaL/tutorial/quickstart.html)
+ [Code Walkthrough](https://inclusionai.github.io/AReaL/developer/overview.html)
+ **Customization Guide**
    - [Dataset](https://inclusionai.github.io/AReaL/customization/dataset.html)
    - [Rollout Behavior (Agentic RL)](https://inclusionai.github.io/AReaL/customization/agent.html)
    - [Training Algorithm](https://inclusionai.github.io/AReaL/customization/algorithm.html)
+ [Contributing](https://inclusionai.github.io/AReaL/contrib.html)

## Future Plan

AReaL is under active development. We plan to have minor releases in a weekly manner and major releases in a monthly manner. Community engagements and contributions are extremely welcomed. We are also **hiring interns and full-timers** with open positions in both the US and China.

For the research and development plan already in place, please see the following list:

### System Development

- [x] Support for SGLang.
- [x] RL training with coding problems.
- [x] Asynchronous generation and RL training.
- [ ] Optimizations for distributed training: expert parallel for MOE and zero-bubble pipelining.
- [ ] RL for vision-language models (VLM).
- [x] Multi-turn agentic RL.
- [ ] Function calling and tool use.

### Algorithm Development

- [x] RL training receipes for 1.5B and 7B models.
- [x] A complete RL training receipe for 32B models.
- [ ] Sample-efficient multi-task RL algorithms.
- [ ] Agentic capabilities with end-to-end RL.
- [ ] Stable RL training for larger MOE models.

## Acknowledgement
We would like to remark that major contributors are from the RL Lab at Ant Research and the Institute for Interdisciplinary Information Sciences, Tsinghua University.

Our team has also received invaluable assistance from the Data Intelligence Lab at Ant Research for data support and from the Super Computing Technology (SCT) team at Ant Group, particularly in the realm of large-scale cluster operations and maintenance.

We also appreciate all the pioneering works from the community, particularly the [ReaLHF](https://github.com/openpsi-project/ReaLHF) project from OpenPsi Inc. and those other projects, including but not limited to, [DeepScaleR](https://github.com/agentica-project/deepscaler), [Open-Reasoner-Zero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/tree/main), [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [VeRL](https://github.com/volcengine/verl), [SGLang](https://github.com/sgl-project/sglang), [QwQ](https://github.com/QwenLM/QwQ), [Light-R1](https://github.com/Qihoo360/Light-R1) and [DAPO](https://github.com/BytedTsinghua-SIA/DAPO).

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
@misc{areal2025,
  author = {RL Lab, Ant Research},
  title = {AReaL: Ant Reasoning RL},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/inclusionAI/AReaL}},
}
```

