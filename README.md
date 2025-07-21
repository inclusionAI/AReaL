<h1 align="center">
<em>AReaL</em>: Ant Reasoning Reinforcement Learning for LLMs
</h1>

<p align="center">
| <a href="https://arxiv.org/pdf/2505.24298"><b>Paper</b></a> | <a href="https://inclusionai.github.io/AReaL/"><b>Documentation</b></a> | <a href="https://deepwiki.com/inclusionAI/AReaL"><b>Ask DeepWiki</b></a> | <a href="https://huggingface.co/collections/inclusionAI/areal-boba-2-683f0e819ccb7bb2e1b2f2d5"><b>🤗 Models & Data</b></a> |
<a href="./assets/wechat_qrcode.png" target="_blank"><b>WeChat Group</b></a> |
</p>

<img align="right" alt="ReaL" src="/assets/logo.png" width="20%">

AReaL (Ant Reasoning RL) is an open-source **fully asynchronous reinforcement learning training system** for large reasoning models developed at **the RL Lab, Ant Research**. Built upon the open-source project [RealHF](https://github.com/openpsi-project/ReaLHF), we are fully committed to open-source by providing training details, data, and infrastructure required to reproduce results along with the model itself. AReaL aims to help everyone build their own AI agents easily and affordably. Our team loves milk tea because it's delicious, customizable, and affordable. We hope you enjoy our project just like how you enjoy real-world milk tea (cheers).

**AReaL Highlights**

+ 🔥 <span style="color: red; font-weight: bold;">**[NEW] Asynchronous RL:**</span> With algorithm-system co-design, AReaL supports fully asynchronous RL for **the fastest training**! Experimental support for multi-turn agentic RL is also provided.
+ 🛠️ **Open & Reproducible**: We continuously release _all code, datasets, and training recipes_ for RL training of LLMs.
+ 🚀 **Scalability**: AReaL can seamlessly adapt to different computational resource settings, ranging from a single node to 1K GPUs.
+ 🔪 **Cutting-Edge Performance:** AReaL can produce models with cutting-edge reasoning capabilities in math and coding. We are also actively working on agentic tasks.

## News

**[2025/06/03] (v0.3, boba²)** We release **boba²** (double-boba) for fully asynchronous RL training, which achieves a **2.77x speedup while obtaining on-par or even better training performance** compared to synchronous systems. Moreover, asynchronous RL makes it extremely easy to set up multi-turn agentic RL training! Check out [our v0.3 overview blog](/blog/AReaL_v0_3.md) and the [research paper](https://arxiv.org/pdf/2505.24298).

**[2025/03/31] (v0.2, boba)** Here comes our next milestone release - boba! Please call it A-ReaL-boba! This release includes much faster training with SGLang support and SOTA 7B and 32B models on math reasoning. Check our [v0.2 technical blog](/blog/AReaL_v0_2.md).

**[2025/02/24] (v0.1)** Our initial release includes reproducible results for 1.5B and 7B LRMs. Check our [v0.1 technical blog](/blog/AReaL_v0_1.md).

## Release Highlights

In our AReaL-boba² (A-ReaL-double-boba) release, we highlight the top 3 most important features:

+ A fully asynchronous RL training pipeline with **system and RL algorithm co-design**, achieving over 2.77x speedup without any performance drop. Check the [benchmark scripts and instructions here](https://github.com/inclusionAI/AReaL/tree/main/benchmark/verl_v0_3_0_post1_76084d3).

+ SOTA coding models, i.e., a 14B model with a **69.1 score on LCB-v5**. To reproduce, check the [configs](https://github.com/inclusionAI/AReaL/tree/main/examples/configs/v0.3-qwen3-code) and [instructions](https://inclusionai.github.io/AReaL/references/reproduce.html).

+ Experimental support for **multi-turn** agentic RL training. Check our [complete example](https://inclusionai.github.io/AReaL/customization/agent.html).

For the complete system design and more training details, please check [our v0.3 blog](/blog/AReaL_v0_3.md) and our [research paper](https://arxiv.org/pdf/2505.24298).

**Jump to the [quickstart section](https://github.com/inclusionAI/AReaL?tab=readme-ov-file#getting-started) if you want to quickly run an experiment and get your hands dirty!** 😈

### Overview of Asynchronous RL Training

During the synchronous RL training process, a generation step must wait until the longest sequence completes within the batch of LLM outputs. Due to the varying output lengths for LRMs, a synchronous RL system suffers from massive GPU idle time, leading to training inefficiency. Some recent works ([DeepCoder](https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51), [Intellect](https://www.primeintellect.ai/blog/intellect-2)) propose overlapping a single training step with a single generation step to accelerate training. However, the largest bottleneck remains unchanged: the samples within a batch are still from the same model version, leading to waiting and GPU idle time.

![Synchronous vs One-step Overlap RL](/assets/sync_one_step_gen.png)

*Fig.1. Left: Execution timeline of synchronous RL training. Right: Execution timeline of one-step overlap RL system.*

AReaL adopts a fully asynchronous RL training framework that completely decouples generation from training. In AReaL, LLM generation runs in a streaming manner, with each rollout worker continuously producing outputs without waiting. Meanwhile, trainer workers perform parallel model updates upon receiving training batches.

![Asynchronous RL Training](/assets/async_timeline.png)

*Fig 2. Execution timeline of our fully asynchronous RL system.*

AReaL follows a system-algorithm co-design principle: on the system side, AReaL efficiently syncs model parameters and carefully controls the staleness of each training sample; on the algorithm side, AReaL improves the objective of PPO to make async-RL stable.

We compare the scalability of **asynchronous RL** training based on our AReaL-boba² system with **classical synchronous RL** training (we adopt the fastest open-source system veRL, main branch on 05/07/2025) across different model sizes and different numbers of H800 GPUs. AReaL demonstrates much improved scaling capabilities with respect to training throughput. This is also partially due to AReaL decoupling training and generation, leading to much fewer GPU memory fragments.

![Scaling Comparison](/assets/async_scaling_vs_verl.png)

*Fig.3 The scaling trend of asynchronous RL (based on AReaL-boba2) and classical synchronous RL (based on veRL) with different model sizes. Dotted lines indicate ideal linear scaling.*

### SOTA Code Generation Model by AReaL-boba²

We use **Qwen3** as our base model. After asynchronous RL training, we achieve SOTA results on LiveCodeBench, Codeforces, and CodeContests benchmarks.

| **Model (8B)** | **LiveCodeBench v5**<br/>**(2024.10-2025.2)** | **Codeforces** | **CodeContests** |
| :---: | :---: | :---: | :---: |
| Qwen3-8B | 58.8 | 1879/96.7% | 31.4 |
| DeepSeek-R1-0528-Qwen3-8B | 58.4 | 1945/97.3% | 31.0 |
| [🤗 AReaL-boba²-8B-Open](https://huggingface.co/inclusionAI/AReaL-boba-2-8B-subset) | 62.0 | 1933/97.2% | **41.4** |
| [🤗 AReaL-boba²-8B](https://huggingface.co/inclusionAI/AReaL-boba-2-8B) | **63.0** | **1962/97.5%** | 40.8 |

| **Model (14B)** | **LiveCodeBench v5**<br/>**(2024.10-2025.2)** | **Codeforces** | **CodeContests** |
| :---: | :---: | :---: | :---: |
| Qwen3-14B | 65.4 | 1978/97.7% | 38.3 |
| DeepCoder-14B-Preview | 60.6 | 1936/95.3% | 40.1 |
| [🤗 AReaL-boba²-14B-Open](https://huggingface.co/inclusionAI/AReaL-boba-2-14B-subset) | 67.3 | 1990/97.8% | **46.2** |
| [🤗 AReal-boba²-14B](https://huggingface.co/inclusionAI/AReaL-boba-2-14B) | **69.1** | **2044/98.2%** | 46.1 |

| **Larger Models** | **LiveCodeBench v5**<br/>**(2024.10-2025.2)** | **Codeforces** | **CodeContests** |
| :---: | :---: | :---: | :---: |
| Qwen3-235B | 70.7 | 2056 | - |
| DeepSeek-R1 | 64.3 | 2029 | - |
| OpenAI-o3-mini (Medium) | 66.3 | 2036 | - |

*Table 1: Coding Task Performance Comparison. AReaL-boba²-8B/14B-Open denotes training results on open-source data. AReaL-boba²-8B/14B models are trained with an additional small amount of internal data and achieve SOTA performance on LiveCodeBench, Codeforces & CodeContests.*

We highlight the [tutorials](https://inclusionai.github.io/AReaL/customization/dataset.html) and [code walkthroughs](https://inclusionai.github.io/AReaL/developer/overview.html) about the following key features for asynchronous training:

+ [Streaming generation and reward computation](https://inclusionai.github.io/AReaL/developer/rollout/rollout_worker.html)
+ [Interruptible rollout](https://inclusionai.github.io/AReaL/developer/rollout/gserver.html)
+ [Data staleness control with the rollout controller](https://inclusionai.github.io/AReaL/developer/rollout/gserver.html)
+ [The adoption of decoupled PPO loss](https://inclusionai.github.io/AReaL/customization/algorithm.html)

### RL Training for Multi-turn Agent

AReaL-boba² allows you to independently customize the [dataset](https://inclusionai.github.io/AReaL/customization/dataset.html), [rollout behavior](https://inclusionai.github.io/AReaL/customization/agent.html), and the [training algorithm](https://inclusionai.github.io/AReaL/customization/algorithm.html), without needing to modify the heavy system-level code.

In particular, we show a simple example to develop a multi-turn math agent for RL training. Please see the learning curve below and reference the [step-by-step guide](https://inclusionai.github.io/AReaL/customization/agent.html) if you want to implement your own agentic RL project.

## Getting Started
Obtain the training data:
- [Math](https://huggingface.co/datasets/inclusionAI/AReaL-boba-Data) 
- [Code](https://huggingface.co/datasets/inclusionAI/AReaL-boba-2-RL-Code)

For code training data, a simple preprocessing script was provided in `examples/data_preprocess/preprocess_training_data.py`:
```bash
python3 preprocess_training_data.py --data_path $original_data_path --output_path $training_data_path
```

Train Qwen3 1.7B locally (Remember to modify `dataset.path` in the script below):
```bash
bash examples/run_async_ppo.sh
```

Evaluation:

```bash
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

## Resources

+ [Documentation](https://inclusionai.github.io/AReaL/)
+ [Contributing](https://inclusionai.github.io/AReaL/contrib.html)

### Quickstart

+ [Installation](https://inclusionai.github.io/AReaL/tutorial/installation.html)
+ [Example: Improving the math capability of Qwen3 with PPO](https://inclusionai.github.io/AReaL/tutorial/quickstart.html)

### Benchmark and Reproduction

+ **Reproduce boba² Code Models**
  - 🤗 **Model weights**: [8B-code](https://huggingface.co/inclusionAI/AReaL-boba-2-8B), [14B-code](https://huggingface.co/inclusionAI/AReaL-boba-2-14B), [8B-code-open](https://huggingface.co/inclusionAI/AReaL-boba-2-8B-subset), [14B-code-open](https://huggingface.co/inclusionAI/AReaL-boba-2-14B-subset)
  - [Evaluation Guide](https://inclusionai.github.io/AReaL/tutorial/eval.html)
  - [Training configs](https://github.com/inclusionAI/AReaL/tree/main/examples/configs/v0.3-qwen3-code) and [instructions](https://inclusionai.github.io/AReaL/references/reproduce.html)
+ [Scripts for Benchmark Training Throughput](https://github.com/inclusionAI/AReaL/tree/main/benchmark/verl_v0_3_0_post1_76084d3)

### Customization Guide

- [Use your own dataset](https://inclusionai.github.io/AReaL/customization/dataset.html)
- [Modifying the reward function and rollout behavior (multi-turn agentic RL)](https://inclusionai.github.io/AReaL/customization/agent.html)
- [Modifying PPO to GRPO](https://inclusionai.github.io/AReaL/customization/algorithm.html#grouped-advantage-normalization)
- [Developing the decoupled PPO loss](https://inclusionai.github.io/AReaL/customization/algorithm.html#the-decoupled-ppo-loss)

### System Code Walkthrough

+ [Trainer](https://inclusionai.github.io/AReaL/developer/trainer/model_worker.html)
+ [Model Backend and Algorithm Interface](https://inclusionai.github.io/AReaL/developer/trainer/algo_interface.html)
+ [Rollout Controller](https://inclusionai.github.io/AReaL/developer/rollout/gserver.html)
+ [Streaming generation and reward computation](https://inclusionai.github.io/AReaL/developer/rollout/rollout_worker.html)

## Future Plan

AReaL is under active development. We plan to have minor releases weekly and major releases monthly. Community engagement and contributions are extremely welcome. We are also **hiring interns and full-time employees** with open positions in both the US and China.

For the research and development plan already in place, please see the following list:

### System Development

- [x] Support for SGLang
- [x] RL training with coding problems
- [x] Asynchronous generation and RL training
- [ ] Optimizations for distributed training: expert parallel for MOE and zero-bubble pipelining
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

We would like to note that major contributors are from the RL Lab at Ant Research and the Institute for Interdisciplinary Information Sciences, Tsinghua University.

Our team has also received invaluable assistance from the Data Intelligence Lab at Ant Research for data support and from the Super Computing Technology (SCT) team at Ant Group, particularly in the realm of large-scale cluster operations and maintenance.

We also appreciate all the pioneering works from the community, particularly the [ReaLHF](https://github.com/openpsi-project/ReaLHF) project from OpenPsi Inc. and other projects, including but not limited to [DeepScaleR](https://github.com/agentica-project/deepscaler), [Open-Reasoner-Zero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/tree/main), [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [VeRL](https://github.com/volcengine/verl), [SGLang](https://github.com/sgl-project/sglang), [QwQ](https://github.com/QwenLM/QwQ), [Light-R1](https://github.com/Qihoo360/Light-R1) and [DAPO](https://github.com/BytedTsinghua-SIA/DAPO).

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
