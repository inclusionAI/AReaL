<h1 align="center">
<em>AReaL</em>: Ant Reasoning Reinforcement Learning for LLMs
</h1>

<p align="center">
| <a href="https://arxiv.org/pdf/2505.24298"><b>Paper</b></a> | <a href="https://inclusionai.github.io/AReaL/"><b>Documentation</b></a> | <a href="https://deepwiki.com/inclusionAI/AReaL"><b>Ask DeepWiki</b></a> |
</p>

<img align="right" alt="ReaL" src="/assets/logo.png" width="20%">

AReaL (Ant Reasoning RL) is a fully open-sourced, scalable, and efficient reinforcement learning training system for large language models developed at **the RL Lab, Ant Research**, built upon the open-source project [RealHF](https://github.com/openpsi-project/ReaLHF). We fully commit to open-source by releasing training details, data, and infra required to reproduce all the models with the desired performances. AReaL aims to help everyone build their own AI agents easily and affordably. Our team loves milk tea as it is delicious, customizable, and affordable. We hope you all enjoy our project just like how you enjoy A-ReaL-milk-tea.🧋 

**AReaL Highlights**

+ 🛠️ **Open & Reproducible**: We will continuously release _all code, datasets, and training recipes_ for RL training LLMs .
+ 🚀 **Scalability**: AReaL can seamlessly adapt to different computational resource settings, ranging from 1 single node to 1K GPUs.
+ 🔪 **Cutting-Edge Performances:** AReaL can produce models with cutting-edge reasoning capabilities. We are actively working on other domains, such as coding and agent, as well. 

## News

**[2025/03/31]** **(v0.2, Boba)** Our milestone release Boba! Please call it A-ReaL-Boba! This release includes much accelerated training with SGLang support and SOTA 7B and 32B models on math reasoning. 

**[2025/02/24]** **(v0.1)** Our initial release includes reproducible results for 1.5B and 7B LRMs. Check our [v0.1 technical blog](/blog/AReaL_v0_1.md).

## AReaL-boba Milestones and Highlights
In our boba release, we highlight the 3 most important milestones:

+ Full SGLang support and a collection of efficiency improvements
+ A SOTA 7B math reasoning model [AReaL-boba-RL-7B](https://huggingface.co/inclusionAI/AReaL-boba-RL-7B) and the corresponing training data [AReaL-boba-106k](https://huggingface.co/datasets/inclusionAI/AReaL-boba-Data/blob/main/AReaL-boba-106k.jsonl).
+ A particularly competitive 32B model [AReaL-boba-SFT-32B](https://huggingface.co/inclusionAI/AReaL-boba-SFT-32B) that can be trained with extremely low cost. (Training Data: [AReaL-boba-SFT-200](https://huggingface.co/datasets/inclusionAI/AReaL-boba-Data/blob/main/AReaL-boba-SFT-200.jsonl))

For the complete training and model details, please check [our v0.2 technical blog](/blog/AReaL_v0_2.md). 

#### SGLang support with 1.5x speedup on 7B Training

![throughput_comparision_with_v0.1.0.png](/assets/thpt_comparison.png) 

Thanks to a series of system-level optimizations, AReaL v0.2 improves its end-to-end training performance by up to 73%. 

In the following table, we show the convergence time under different resource settings:

| **Model Size** | **1.5B** | **1.5B** | **1.5B** | **7B** | **7B** |**32B (SFT)** |
| --- |:--------:|:--------:|:--------:|:------:|:------:|:-------:|
| #GPU |    8     |    32    |   128    |   32   |  128   |  64 |
| Step |   250    |   250    |   250    |  400   |  400   |  300 |
| Time (h) |   ~240   |   ~69    |   ~27    |  ~252  |  ~90   |  ~3.5 |


#### SOTA 7B model using RL in math reasoning
| **Model** | **AIME 2024** | **AIME 2025** | **GPQA-Diamond** |
| :---: | :---: | :---: | :---: |
| O1-Preview | 56.7 | - |  |
| R1-Distill-Qwen-7B | 55.0 | 39.7 | 47.1 |
| Light-R1-7B-DS | 56.7 | 44.9 | 40.9 |
| [AReaL-boba-RL-7B 🤗](https://huggingface.co/inclusionAI/AReaL-boba-RL-7B) | **61.9** | **48.3** | **47.6** |


We use **R1-Distill-Qwen-7B** as our base model. After RL training, the pass@1 scores on AIME 2024 and AIME 2025 improve by 6.9 and 8.6 points, respectively, achieving SOTA performance among 7B models in mathematical reasoning. We have released the training data at [AReaL-boba-106k](https://huggingface.co/datasets/inclusionAI/AReaL-boba-Data/blob/main/AReaL-boba-106k.jsonl).  

Although our dataset primarily consists of math and logic problems, we observed that RL training led to measurable improvements on the challenging STEM benchmark GPQA. We plan to open-source more datasets in the future, including code, STEM, and other domains. All the reported numbers are re-evaluated using our evaluation code with more details in our [blog](blog/AReaL_v0_2.md#eval_detail). 

#### Approaching QwQ-32B performances using only 200 data samples
| **Model** | **AIME 2024** |
| :---: | :---: |
| R1-Distill-Qwen-32B | 72.6 |
| QwQ-32B | 78.9 |
| [AReaL-boba-SFT-32B 🤗](https://huggingface.co/inclusionAI/AReaL-boba-SFT-32B) | 78.8 |


Building upon **R1-Distill-Qwen-32B**, we replicate **QwQ-32B's** inference performance on AIME 2024 using just **200 data points** via Supervised Fine-Tuning (SFT). We have released the training data at [AReaL-boba-SFT-200](https://huggingface.co/datasets/inclusionAI/AReaL-boba-Data/blob/main/AReaL-boba-SFT-200.jsonl).

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

Evaluation

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
  + [Dataset](https://inclusionai.github.io/AReaL/customization/dataset.html)
  + [Rollout Behavior (Agentic RL)](https://inclusionai.github.io/AReaL/customization/agent.html)
  + [Training Algorithm](https://inclusionai.github.io/AReaL/customization/algorithm.html)
+ [Contributing](https://inclusionai.github.io/AReaL/contrib.html)

## Future Plan
AReaL is under active development. We will have major releases in a weekly manner. We also highly appreciate efforts from the community as well. Here we highlight our future research and development plan. 

### System Development
- [x] Support for SGLang.
- [x] RL training with coding problems.
- [x] Asynchronous generation and RL training.
- [ ] Optimizations for distributed training: expert parallel and zero-bubble pipelining.
- [ ] RL for vision-language models (VLM).
- [ ] Function calling and agent capabilities.

### Algorithm Development
- [x] RL training receipes for 1.5B and 7B models.
- [x] A complete RL training receipe for 32B models.
- [ ] Sample-efficient multi-task RL algorithms.
- [ ] Agentic capabilities with end-to-end RL.
- [ ] Stable RL training for larger MOE models.

## Acknowledgement
We would like to remark that major contributors are from **RL Lab at Ant Research** and **Institute for Interdisciplinary Information Sciences, Tsinghua University**.

Our team has also received invaluable assistance from the Super Computing Technology (SCT) team at Ant Group, particularly in large-scale cluster operations and maintenance. 

We also appreciate all the pioneer works from the community, particularly the [ReaLHF](https://github.com/openpsi-project/ReaLHF) project from OpenPsi Inc. and many other projects, including but not limited to, [DeepScaleR](https://github.com/agentica-project/deepscaler), [Open-Reasoner-Zero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/), [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [verl](https://github.com/volcengine/verl), [SGLang](https://github.com/sgl-project/sglang), [QwQ](https://github.com/QwenLM/QwQ),  [Light-R1](https://github.com/Qihoo360/Light-R1), and [DAPO](https://github.com/BytedTsinghua-SIA/DAPO).

## Citation
```plain
@inproceedings{mei2025real,
  author       = {Mei, Zhiyu and Fu, Wei and Li, Kaiwei and Wang, Guangju and Zhang, Huanchen and Wu, Yi},
  title        = {ReaL: Efficient RLHF Training of Large Language Models with Parameter Reallocation},
  booktitle    = {Proceedings of the Eighth Conference on Machine Learning and Systems,
                  MLSys 2025, Santa Clara, CA, USA, May 12-15, 2025},
  publisher    = {mlsys.org},
  year         = {2025},
}
```

```plain
@misc{areal2025,
  author = {RL Lab, Ant Research},
  title = {AReaL: Ant Reasoning RL},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/inclusionAI/AReaL}},
}
```

