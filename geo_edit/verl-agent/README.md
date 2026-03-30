<p align="center">
    <img src="./docs/gigpo/logo-verl-agent.png" alt="logo" width="55%">
</p>


<h3 align="center">
<b>Group-in-Group Policy Optimization for LLM Agent Training</b>
<br>
<b>NeurIPS 2025</b>
</h3>


<p align="center">
  <a href="https://arxiv.org/abs/2505.10978">
    <img src="https://img.shields.io/badge/arXiv-Paper-red?style=flat-square&logo=arxiv" alt="arXiv Paper"></a>
  &nbsp;
  <a href="https://github.com/langfengQ/verl-agent">
    <img src="https://img.shields.io/badge/GitHub-Project-181717?style=flat-square&logo=github" alt="GitHub Project"></a>
  &nbsp;
  <a href="https://huggingface.co/collections/langfeng01/verl-agent-684970e8f51babe2a6d98554">
    <img src="https://img.shields.io/badge/HuggingFace-Models-yellow?style=flat-square&logo=huggingface" alt="HuggingFace Models"></a>
  &nbsp;
  <a href="https://x.com/langfengq/status/1930848580505620677">
    <img src="https://img.shields.io/badge/Twitter-Channel-000000?style=flat-square&logo=x" alt="X Channel"></a>
  &nbsp;
  <a href="https://github.com/langfengQ/verl-agent/blob/master/LICENSE">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg?style=flat-square" alt="License"></a>
  &nbsp;
  <a href="https://github.com/langfengQ/verl-agent/issues">
    <img src="https://img.shields.io/github/issues/langfengQ/verl-agent?style=flat-square&color=green" alt="GitHub issues"></a>
  &nbsp;
  <a href="https://github.com/langfengQ/verl-agent/stargazers">
    <img src="https://img.shields.io/github/stars/langfengQ/verl-agent?style=social" alt="Repo stars"></a>
  &nbsp;
</p>

`verl-agent` is an extension of [veRL](https://github.com/volcengine/verl), specifically designed for training **large language model (LLM) agents via reinforcement learning (RL)**. 

Unlike prior approaches that simply concatenate full interaction histories, `verl-agent` proposes **step-independent multi-turn rollout mechanism**, which allows for **fully customizable** per-step input structures, history management, and memory modules. This design makes `verl-agent` **highly scalable for very long-horizon, multi-turn RL training** (e.g., tasks in ALFWorld can require up to 50 steps to complete).

`verl-agent` provides a **diverse set of RL algorithms** (including our new algorithm GiGPO) and a **rich suite of agent environments**, enabling the development of reasoning agents in both visual and text-based tasks.

# News
- [2026.02] `HGPO` accepted at [ICLR 2026](https://iclr.cc/)! 🎉🎉🎉 [[Paper](https://openreview.net/forum?id=T8Dev99qnz)] [[Code](https://github.com/langfengQ/verl-agent/tree/master/recipe/hgpo)]
- [2026.02] 🔥 We open-source [Dr. MAS](https://github.com/langfengQ/DrMAS), which supports stable end-to-end RL post-training of **multi-agent LLM systems**! [[Paper](https://arxiv.org/pdf/2602.08847)] [[Code](https://github.com/langfengQ/DrMAS)]
- [2025.12] `Qwen3-VL` is supported! See example [here](./examples/gigpo_trainer/run_sokoban_qwen3vl.sh).
- [2025.09] `GiGPO` is now supported by [ROLL](https://github.com/alibaba/ROLL)! [[Document](https://alibaba.github.io/ROLL/docs/English/UserGuide/agentic/agentic_GiGPO)] [[Train Curves](https://github.com/alibaba/ROLL/issues/173#issuecomment-3332106534)].
- [2025.09] `verl-agent`-style training pipeline is now supported by [OpenManus-RL](https://github.com/OpenManus/OpenManus-RL)!
- [2025.09] [GiGPO](https://arxiv.org/abs/2505.10978) accepted at [NeurIPS 2025](https://neurips.cc/)! 🎉🎉🎉
- [2025.08] Add **Search-R1 experiments** and **similarity-based GiGPO**! Check out GiGPO's superior performance in Search-R1 experiments [here](#results).
- [2025.07] `GiGPO` & `verl-agent` talks at [Agent for SWE meetup](https://lu.ma/e498qhsi) by LF AI & Data Singapore on 7/11.
- [2025.07] Add modular memory manager. See [here](./agent_system/memory).
- [2025.06] ***Major update***: Merge all features from the latest [veRL](https://github.com/volcengine/verl). For example, `verl-agent` now supports Qwen3, LoRA, REINFORCE++, and more. Feel free to explore!
- [2025.05] Code released and paper on `GiGPO` released.

# Quick Feature Summary
| Feature Category | Supported Capabilities|
| - | - |
| **Interaction**          | ✅ Multi-turn Agent-Environment interaction<br>✅ Step-wise interaction<br>✅ Scalable for long-horizon tasks |
| **Memory**               | ✅ Fully customizable memory module<br>✅ Flexible history management|
| **Input Flexibility**    | ✅ Fully customizable per-step input structures |
| **Execution**            | ✅ Parallelized Gym environments<br>✅ Group environments support (for group-based RL)|
| **Model Support**        | ✅ Qwen3<br>✅ Qwen3-VL<br>✅ Qwen2.5<br>✅ Qwen2.5-VL<br>✅ LLaMA3.2<br>and more |
| **Modality**             | ✅ Text-only<br>✅ Text + Image (multi-modal) |
| **Lightweight Training** | ✅ Supports LoRA training |
| **Environments**         | ✅ ALFWorld<br>✅ WebShop<br> ✅ Search (Tool Calling)<br> ✅ Sokoban<br>✅ Gym Cards<br>✅ AppWorld |
| **RL Algorithms**        | ✅ GiGPO<br>✅ GRPO<br>✅ PPO<br>✅ DAPO<br>✅ GSPO<br>✅ RLOO<br>✅ REINFORCE++<br>✅ Dynamic sampling & clip-higher supported <br> and more |
| **Prompt-based Agent**   | ✅ GPT-4o prompt-based agent  |

# Framework Comparison
<p align="center">
    <img src="./docs/gigpo/framework-comparison.png" alt="framework" width="100%">
</p>


# Table of Contents

- [Key Features](#key-features)
- [Results](#results)  
- [Installation](#installation)  
  - [Install veRL](#install-verl)  
  - [Install Supported Environments](#install-supported-environments)  
    - [1. ALFWorld](#1-alfworld)  
    - [2. WebShop](#2-webshop)
    - [3. Search](#3-search)  
    - [4. Sokoban](#4-sokoban)  
    - [5. Gym Cards](#5-gym-cards)  
    - [6. AppWorld (Experimental)](#6-appworld-experimental)  
- [Run Examples](#run-examples)  
  - [RL Training](#rl-training)  
    - [1. GiGPO](#1-gigpo)  
    - [2. GRPO](#2-grpo)  
    - [3. PPO](#3-ppo)  
    - [4. RLOO](#4-rloo)  
    - [5. DAPO](#5-dapo)  
    - [6. GiGPO (dynamic)](#6-gigpo-dynamic)
  - [LoRA](#lora)
  - [Prompt-based Agent with GPT-4o](#prompt-based-agent-with-gpt-4o)
- [FAQ](#faq)
  - [1. Customize Memory Module](#1-customize-memory-module)
  - [2. Data Preparation](#2-data-preparation)
  - [3. Customize Your Own Prompts](#3-customize-your-own-prompts)
  - [4. Add New Environments](#4-add-new-environments)
- [Contributing](#contributing)
- [Acknowledgement](#acknowledgement)
- [Awesome Work Powered by verl-agent & GiGPO](#awesome-work-powered-by-verl-agent--gigpo)
- [Citation](#citation)
- [Star History](#star-history)

# Key Features

- **Multi-Turn Agent-Environment Interaction**

  `verl-agent` supports multi-step interactive loops between agents and environments. Agents perceive environmental feedback after each step, forming the basis for reinforcement learning.

- **Fully Customizable Memory Module & Per-Step Input Structure**

  `verl-agent` features a **customizable memory module** (see [here](./agent_system/memory)) that allows for flexibly choosing what history to include for each step. The input typically consists of the current observation along with a concise history summary at each step (see prompt [here](./agent_system/environments/prompts/webshop.py)). Developers can **freely define what to include, such as recent steps, key events, summaries, or external knowledge**. There's no requirement to concatenate the full history, and the input structure for each step is ***fully customizable***.

- **Scalable for Very Long-Horizon Optimization**

  Prior works like [RAGEN](https://github.com/RAGEN-AI/RAGEN) and [Search-R1](https://github.com/PeterGriffinJin/Search-R1) concatenate the entire history of states and responses. This causes the context length to grow rapidly with the number of turns, making them difficult to scale to long-horizon scenarios. In contrast, `verl-agent` constructs inputs step-by-step. Each input is concise and customizable. This design keeps the context length almost constant over time, making `verl-agent` highly scalable for long-horizon scenarios (e.g., 30–50 steps in ALFWorld) without running into token limits or inefficiency.
  
- **Parallelized Gym-Style Environments and Group Environments**

  `verl-agent` provides a gym-style interface with support for parallelized environments. This enables high-throughput rollouts, speeding up training. In addition, `verl-agent` introduces the concept of group environments. All environments within a group share identical initial states during `reset()`. This is especially useful for algorithms like GRPO and DAPO that require multiple rollouts on the same state. You can configure the number of rollouts per group using the `env.rollout.n` in [ppo_trainer.yaml](./verl/trainer/config/ppo_trainer.yaml) config file.

- **Support for Various Models**

  `verl-agent` supports a wide range of LLMs, including `Qwen3`, `Qwen3-VL`, `Qwen2.5`, `LLaMA3.2`, `Qwen2.5-VL`, and others, allowing flexibility for various deployment needs.

- **LoRA Fine-Tuning Support**

  `verl-agent` provides support for [LoRA](https://arxiv.org/abs/2106.09685) (Low-Rank Adaptation), significantly reducing computational cost. Now, `verl-agent` supports training 7B models using 2 H100 GPUs.

- **Vision-Language Agent Support**

  Beyond text-based agents, `verl-agent` also supports training vision-language agents. This enables multi-modal reasoning in environments where both visual perception and language understanding are required.

- **Rich Suite of Environments**
  
  `verl-agent` offers a diverse set of interactive environments including [Search-R1](https://github.com/PeterGriffinJin/Search-R1) experiment, embodied AI environments like [ALFWorld](https://github.com/alfworld/alfworld), visual games such as [Sokoban](https://github.com/mpSchrader/gym-sokoban) and [Gym Cards](https://github.com/RL4VLM/RL4VLM/blob/main/gym-cards/README.md), and digital interface control tasks like [WebShop](https://github.com/princeton-nlp/WebShop) and [AppWorld](https://github.com/stonybrooknlp/appworld/) (experimental). 

- **Diverse RL Algorithms**

  `verl-agent` includes implementations of various RL algorithms, such as [GRPO](https://arxiv.org/abs/2402.03300), [PPO](https://arxiv.org/abs/1707.06347), [DAPO](https://arxiv.org/abs/2503.14476), [GSPO](https://arxiv.org/abs/2507.18071), [RLOO](https://arxiv.org/abs/2402.14740) and our new state-of-the-art algorithm [GiGPO](https://arxiv.org/abs/2505.10978). It also supports several variants enhanced with dynamic sampling and clip-higher techniques.

# Results
> ⚠️ Note: The performance of GiGPO has improved slightly after the "[2025.06.03] Major Update." To reproduce the original paper results, please use the version released prior to the "[2025.06.03] Major Update."

| Algorithm          | Task         | Model      | Success Rate (Paper) | Training Log |
|-------------------|--------------|--------------------------|-----------------------|-------------------------|
| GiGPO | ALFWorld     | Qwen2.5-1.5B-Instruct    | 86.7%   |  [![wandb](https://img.shields.io/badge/W%26B-view-FFBE00?logo=wandb)](https://api.wandb.ai/links/langfeng-cs-nanyang-technological-university-singapore/78zz4sc9) |
| GiGPO | ALFWorld     | Qwen2.5-7B-Instruct      | 90.8%   |  [![wandb](https://img.shields.io/badge/W%26B-view-FFBE00?logo=wandb)](https://api.wandb.ai/links/langfeng-cs-nanyang-technological-university-singapore/78zz4sc9) |
| GiGPO | WebShop      | Qwen2.5-1.5B-Instruct    | 67.4%   |  [![wandb](https://img.shields.io/badge/W%26B-view-FFBE00?logo=wandb)](https://api.wandb.ai/links/langfeng-cs-nanyang-technological-university-singapore/zfnvpvxe) |
| GiGPO | WebShop      | Qwen2.5-7B-Instruct      | 75.2%   |  [![wandb](https://img.shields.io/badge/W%26B-view-FFBE00?logo=wandb)](https://api.wandb.ai/links/langfeng-cs-nanyang-technological-university-singapore/zfnvpvxe) |
| GiGPO | Sokoban [6x6]| Qwen2.5-VL-3B-Instruct   | 81.0%   | [![wandb](https://img.shields.io/badge/W%26B-view-FFBE00?logo=wandb)](https://api.wandb.ai/links/langfeng-cs-nanyang-technological-university-singapore/xm92tyea) |
| GiGPO | EZPoints     | Qwen2.5-VL-3B-Instruct   | 100.0%  |  [![wandb](https://img.shields.io/badge/W%26B-view-FFBE00?logo=wandb)](https://api.wandb.ai/links/langfeng-cs-nanyang-technological-university-singapore/k0y51zei) |
| GiGPO | NumberLine   | Qwen2-VL-2B-Instruct     | 100.0%  | [![wandb](https://img.shields.io/badge/W%26B-view-FFBE00?logo=wandb)](https://api.wandb.ai/links/langfeng-cs-nanyang-technological-university-singapore/81qzsc3n) |


<table border="1" cellspacing="0" cellpadding="5">
  <thead>
    <tr>
      <th>Date</th>
      <th>Method</th>
      <th>NQ†</th>
      <th>TriviaQA*</th>
      <th>PopQA*</th>
      <th>HotpotQA†</th>
      <th>2Wiki*</th>
      <th>MuSiQue*</th>
      <th>Bamboogle*</th>
      <th>Avg.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan="10" align="center"><b>Qwen2.5-3B-Instruct</b></td>
    </tr>
    <tr>
      <td>2025.03</td><td>R1-Instruct</td><td>27.0</td><td>53.7</td><td>19.9</td><td>23.7</td><td>29.2</td><td>7.2</td><td>29.3</td><td>27.1</td>
    </tr>
    <tr>
      <td>2025.03</td><td>Search-R1</td><td>34.1</td><td>54.5</td><td>37.8</td><td>32.4</td><td>31.9</td><td>10.3</td><td>26.4</td><td>32.5</td>
    </tr>
    <tr>
      <td>2025.05</td><td>ZeroSearch</td><td>41.4</td><td>57.4</td><td>44.8</td><td>27.4</td><td>30.0</td><td>9.8</td><td>11.1</td><td>31.7</td>
    </tr>
    <tr>
      <td>2025.05</td><td>StepSearch</td><td>-</td><td>-</td><td>-</td><td>34.5</td><td>32.0</td><td>17.4</td><td>34.4</td><td>-</td>
    </tr>
    <tr>
      <td>2025.05</td><td><b>GiGPO</b><a href="https://api.wandb.ai/links/langfeng-cs-nanyang-technological-university-singapore/1dd48ymw" target="_blank">
      <img src="https://img.shields.io/badge/W%26B-view-FFBE00?logo=wandb" alt="wandb link"/>
    </a></td><td>42.0</td><td>59.5</td><td>42.4</td><td>36.9</td><td>37.0</td><td>12.6</td><td>64.1</td><td>42.1</td>
    </tr>
    <tr>
      <td colspan="10" align="center"><b>Qwen2.5-7B-Instruct</b></td>
    </tr>
    <tr>
      <td>2025.03</td><td>R1-Instruct</td><td>21.0</td><td>44.9</td><td>17.1</td><td>20.8</td><td>27.5</td><td>6.0</td><td>19.2</td><td>22.4</td>
    </tr>
    <tr>
      <td>2025.03</td><td>Search-R1</td><td>39.3</td><td>61.0</td><td>39.7</td><td>37.0</td><td>40.1</td><td>14.6</td><td>36.8</td><td>38.5</td>
    </tr>
    <tr>
      <td>2025.05</td><td>ZeroSearch</td><td>43.6</td><td>61.8</td><td>51.5</td><td>34.6</td><td>35.2</td><td>18.4</td><td>27.8</td><td>39.1</td>
    </tr>
    <tr>
      <td>2025.05</td><td>StepSearch</td><td>-</td><td>-</td><td>-</td><td>38.6</td><td>36.6</td><td>22.6</td><td>40.0</td><td>-</td>
    </tr>
    <tr>
      <td>2025.05</td><td><b>GiGPO</b><a href="https://api.wandb.ai/links/langfeng-cs-nanyang-technological-university-singapore/1dd48ymw" target="_blank">
      <img src="https://img.shields.io/badge/W%26B-view-FFBE00?logo=wandb" alt="wandb link"/>
    </a></td><td>46.4</td><td>64.7</td><td>46.1</td><td>41.6</td><td>43.6</td><td>18.9</td><td>68.9</td><td>47.2</td>
    </tr>
  </tbody>
</table>


We have released our models on [HuggingFace](https://huggingface.co/collections/langfeng01/verl-agent-684970e8f51babe2a6d98554).

# Installation
## Install veRL
```bash
conda create -n verl-agent python==3.12 -y
conda activate verl-agent

pip3 install vllm==0.11.0

pip3 install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install -e .
```

## Install Supported Environments
> ⚠️ **Important:** 
To run an agent in any of these environments, you must first install and configure the corresponding environment. We strongly recommend installing ***each environment in its own dedicated conda environment*** to avoid potential package version conflicts.

### 1. ALFWorld
Install with pip:
```bash
pip3 install gymnasium==0.29.1
pip3 install stable-baselines3==2.6.0
pip install alfworld
```

Download PDDL & Game files and pre-trained MaskRCNN detector (will be stored in `~/.cache/alfworld/`):
```bash
alfworld-download -f
```

Use `--extra` to download pre-trained checkpoints and seq2seq data.

Play a Textworld game:
```bash
alfworld-play-tw
```
---

### 2. WebShop
WebShop requires Python <=3.10, so begin by creating a new `verl-agent-webshop` environment
```bash
conda create -n verl-agent-webshop python==3.10 -y
conda activate verl-agent-webshop
```

Install WebShop
```bash
cd ./agent_system/environments/env_package/webshop/webshop
./setup.sh -d all
```

Note: If you encounter issues with gdown, you may need to visit `https://drive.google.com/`, get your Google Drive cookie, and paste it into `.cache/gdown/cookies.txt`.
Or you may need to manually download the files.

After WebShop is installed, return to the root directory of the repository and install the verl package in `verl-agent`:
```bash
cd repo_root/
pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn==2.7.4.post1 --no-build-isolation
pip3 install -e .
pip3 install vllm==0.8.2
# spacy 3.7.2 requires typer<0.10.0,>=0.3.0, but you have typer 0.15.2 which is incompatible.
# weasel 0.3.4 requires typer<0.10.0,>=0.3.0, but you have typer 0.15.2 which is incompatible.
```
The warnings can be safely ignored.

---

### 3. Search
```bash
cd ./agent_system/environments/env_package/search/third_party
pip install -e .
pip install gym==0.26.2
```

Prepare dataset (data will be saved at `~/data/searchR1_processed_direct`):
```bash
cd repo_root/
python examples/data_preprocess/preprocess_search_r1_dataset.py
```


Since faiss-gpu is not available via pip, we setup a separate conda environment for the local retrieval server. Running this server will use around 6GB of GPU memory per GPU, so make sure to account for this in your training run configuration. Build Retriever environments:
```bash
# Create and activate the retriever environment with Python 3.10
conda create -n retriever python=3.10 -y
conda activate retriever

# Install PyTorch (with GPU support) and related libraries
conda install numpy==1.26.4 # needed to stop incompatible version of numpy from being installed via pip
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install other Python packages
pip install transformers datasets pyserini huggingface_hub

# Install the GPU version of faiss
conda install faiss-gpu==1.8.0 -c pytorch -c nvidia -y

# Install the API service framework
pip install uvicorn fastapi
```

Download the index:
```bash
conda activate retriever

local_dir=~/data/searchR1
python examples/search/searchr1_download.py --local_dir $local_dir
cat $local_dir/part_* > $local_dir/e5_Flat.index
gzip -d $local_dir/wiki-18.jsonl.gz
```

Start the local flat e5 retrieval server: 
```bash
conda activate retriever

# redirect the output to a file to avoid cluttering the terminal
# we have observed outputting to the terminal causing spikes in server response times
bash examples/search/retriever/retrieval_launch.sh > retrieval_server.log 
```

### 4. Sokoban
```bash
pip install matplotlib
pip install gym==0.26.2
pip install gym_sokoban==0.0.6
```
---
### 5. Gym Cards

```bash
cd repo_root/
pip3 install -e ./agent_system/environments/env_package/gym_cards/gym-cards/
pip3 install gymnasium==0.29.1
pip3 install stable-baselines3==2.6.0
```
---
### 6. AppWorld (Experimental)
Install AppWorld package
```bash
cd repo_root/
pip install git+https://github.com/StonyBrookNLP/appworld.git
appworld install
pip install -e .
```
You can ignore the warning of incompatibility for appworld, because we don't run appworld in `verl-agent` environment.

Create a dedicated conda environment `appworld` for the AppWorld server:
```bash
conda create -n appworld python=3.12 -y
conda activate appworld
pip install git+https://github.com/StonyBrookNLP/appworld.git
appworld install
appworld download data
```


<!-- > ⚠️ **Important:**  
To run an agent in any of these environments, you must first install and configure the corresponding environment. Please refer to the [Environment Setup Guide](agent_system/environments/README.md) for step-by-step installation instructions. -->

# Run Examples
## RL Training
We provide out-of-the-box scripts in the ["examples/"](./examples/) directory for training agents in different environments.

Here are some examples:
### 1. GiGPO
GiGPO is our novel algorithm designed to support fine-grained credit assignment in long-horizon LLM agent training. It introduces a two-level grouping mechanism:
- Episode-level groups capture overall task success via total returns (like GRPO).
- Step-level groups gather repeated states across trajectories to compute relative advantages for individual actions.

GiGPO is fully critic-free, maintains the same GPU memory footprint and LLM rollout cost as GRPO, yet achieves significantly better training efficiency and performance.

```bash
bash examples/gigpo_trainer/run_alfworld.sh # ALFWorld
```
```bash
bash examples/gigpo_trainer/run_webshop.sh # WebShop
```
```bash
bash examples/gigpo_trainer/run_search.sh # Search
```
```bash
bash examples/gigpo_trainer/run_sokoban.sh # Sokoban
```
### 2. GRPO
GRPO is a critic-free algorithm that estimates relative advantages based on a group of full episode trajectories.
```bash
bash examples/grpo_trainer/run_alfworld.sh # ALFWorld
```
```bash
bash examples/grpo_trainer/run_webshop.sh # WebShop
```
### 3. PPO
PPO is a classic actor-critic algorithm that updates the policy using a clipped objective to ensure stable learning. It requires a separate value network (critic) to estimate state values.
```bash
bash examples/ppo_trainer/run_alfworld.sh # ALFWorld
```
```bash
bash examples/ppo_trainer/run_webshop.sh # WebShop
```
### 4. RLOO
For RLOO, we use a leave-one-out estimate and the PPO-clip update (instead of the REINFORCE update), making it closer to [LOOP](https://arxiv.org/abs/2502.01600).
```bash
bash examples/rloo_trainer/run_alfworld.sh # ALFWorld
```
```bash
bash examples/rloo_trainer/run_webshop.sh # WebShop
```
### 5. DAPO
DAPO enhances GRPO with techniques like dynamic sampling and clip-higher.
```bash
bash examples/dapo_trainer/run_alfworld.sh # ALFWorld
```
```bash
bash examples/dapo_trainer/run_webshop.sh # WebShop
```
### 6. GiGPO (dynamic)
GiGPO uses dynamic sampling and clip-higher from DAPO
```bash
bash examples/gigpo_dynamic_trainer/run_alfworld.sh # ALFWorld
```
```bash
bash examples/gigpo_dynamic_trainer/run_webshop.sh # WebShop
```

## LoRA
```bash
bash examples/gigpo_trainer/run_alfworld_lora.sh
```

## Prompt-based Agent with GPT-4o
We also provide a prompt-based GPT-4o agent.
```bash
bash examples/prompt_agent/run_gpt4o_agent.sh
```

# FAQ

## 1. Customize Memory Module
`verl-agent` supports a customizable and flexible memory system for managing and formatting interaction history between the agent and the environment. We provide a [SimpleMemory](./agent_system/memory/memory.py) implementation as a default starting point. This memory module is invoked within [env_manager.py](./agent_system/environments/env_manager.py) (i.e., `build_text_obs()`) to construct the observation at each step. 

Developers are encouraged to extend this module with custom memory strategies, such as dynamic summarization, selective memory retention, or external knowledge integration, to improve the handling of long-horizon interaction histories.

## 2. Data Preparation
For most environments (e.g., AFLWorld, WebShop, Sokoban), we only use data preparation to indicate the modality, either "text" or "visual". For example, if the task is purely text-based, the data will just be an empty string "". If it involves visual input, it will be "\<image\>". As for agent input (including task instruction, observation and prompt), we follow the classical RL pipeline. That means the input of LLM agent comes from the environment's feedback through `env.step()`. In the case of search-r1 experiments where tasks are drawn from a dataset, we leverage the [env_kwargs](./examples/data_preprocess/preprocess_search_r1_dataset.py#L90) parameter to pass tasks into the environment, using: [envs.reset(kwargs=gen_batch.non_tensor_batch.pop('env_kwargs', None))](./agent_system/multi_turn_rollout/rollout_loop.py#L301).

## 3. Customize Your Own Prompts  
We adopt a simple and minimal prompt format in our implementation. For example, in the WebShop environment:
```
You are an expert autonomous agent operating in the WebShop e‑commerce environment.
Your task is to: {task_description}. Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}. You are now at step {current_step} and your current observation is: {current_observation}. Your admissible actions of the current situation are: [{available_actions}].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags. Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
```
If you wish to further enhance or customize them, you can find and edit them in: [agent_system/environments/prompts](./agent_system/environments/prompts/).


## 4. Add New Environments
To add a new environment, 
1. Create your environment package (gym-style interface and multi-process execution) in [agent_system/environments/env_package/](./agent_system/environments/env_package/)
2. Define the corresponding prompt files in [agent_system/environments/prompts](./agent_system/environments/prompts/). 
3. Register your new environment in [env_manager.py](./agent_system/environments/env_manager.py), following the structure defined by [EnvironmentManagerBase](./agent_system/environments/base.py#L19). 

For a reference implementation, see the webshop environment:
1. Environment package: [webshop package](./agent_system/environments/env_package/webshop)
2. Prompts: [webshop prompts](./agent_system/environments/prompts/webshop.py)
3. Environment Manager: [webshop env manager](./agent_system/environments/env_manager.py#L304)


# Contributing

We welcome and appreciate all contributions! If you have ideas to improve `verl-agent`, please feel free to submit a pull request (PR).

Example contributions include:
- **AppWorld Bug Fixes**: Fixed compatibility issues and ensured stable integration with the experimental AppWorld environment.
- **Asynchronous Rollout**: Improved training efficiency and throughput by supporting asynchronous rollout pipelines.
- **Additional Environments**: Added support for additional interactive environments to expand the benchmark coverage and task diversity.

# Acknowledgement

`verl-agent` codebase is built upon [veRL](https://github.com/volcengine/verl). 
The supported environments are adapted from [ALFWorld](https://github.com/alfworld/alfworld), [Sokoban](https://github.com/mpSchrader/gym-sokoban), [SkyRL-Gym](https://github.com/NovaSky-AI/SkyRL/tree/main/skyrl-gym), [Search-R1](https://github.com/PeterGriffinJin/Search-R1), [Gym Cards](https://github.com/RL4VLM/RL4VLM/tree/main/gym-cards), [WebShop](https://github.com/princeton-nlp/WebShop), and [AppWorld](https://github.com/stonybrooknlp/appworld). We extend our gratitude to the authors and contributors of these projects for their valuable work.

We would also like to thank the following contributors for their specific improvements to this project: WebShop bug fix ([@YSLIU627](https://github.com/YSLIU627)), GSPO support ([@MakeKJ](https://github.com/MakeKJ)), Qwen3-VL support ([@FabianSchuetze](https://github.com/FabianSchuetze)).

# Awesome Work Powered by verl-agent & GiGPO

- [HGPO](https://arxiv.org/pdf/2602.22817): **Hierarchy-of-Groups** Policy Optimization for Long-Horizon Agentic Tasks. 
- [Dr. MAS](https://arxiv.org/abs/2602.08847): Stable **end-to-end RL** post-training for **multi-agent LLM systems**. [![[code]](https://img.shields.io/github/stars/langfengQ/DrMAS)](https://github.com/langfengQ/DrMAS)
- [AgentOCR](https://arxiv.org/abs/2601.04786): Efficient token compression by rendering multi-turn agent history into images and adopting agentic self-compression.
- [OpenManus-RL](https://github.com/OpenManus/OpenManus-RL): An open-source framework for live-stream reinforcement learning tuning of LLM agents. [![[code]](https://img.shields.io/github/stars/OpenManus/OpenManus-RL)](https://github.com/OpenManus/OpenManus-RL)
- [RLVMR](https://github.com/Tencent/DigitalHuman/tree/main/RLVMR): Providing agents with fine-grained meta-reasoning rewards in long-horizon tasks. [![[code]](https://img.shields.io/github/stars/Tencent/DigitalHuman)](https://github.com/Tencent/DigitalHuman/tree/main/RLVMR)
- [UI-S1](https://github.com/X-PLUG/MobileAgent/tree/main/UI-S1): A GUI automation model using semi-online reinforcement learning for stable long-horizon task execution. [![[code]](https://img.shields.io/github/stars/X-PLUG/MobileAgent)](https://github.com/X-PLUG/MobileAgent/tree/main/UI-S1)
- [Agent Learning via Early Experience](https://arxiv.org/pdf/2510.08558): A scalable, reward-free paradigm that bridges imitation learning and RL via implicit world modeling and self-reflection.
- [SPEAR](https://github.com/TencentYoutuResearch/SPEAR): **Self-imitation** with **Progressive Exploration** for Agentic Reinforcement Learning (ICLR 2026). [![[code]](https://img.shields.io/github/stars/TencentYoutuResearch/SPEAR)](https://github.com/TencentYoutuResearch/SPEAR/tree/main/)

# Citation
If you find `verl-agent` and `GiGPO` useful in your research or applications, we would appreciate it if you could cite our work:

```
@article{feng2025group,
  title={Group-in-Group Policy Optimization for LLM Agent Training},
  author={Feng, Lang and Xue, Zhenghai and Liu, Tingcong and An, Bo},
  journal={arXiv preprint arXiv:2505.10978},
  year={2025}
}
```

# Star History

[![Star History Chart](https://api.star-history.com/svg?repos=langfengQ/verl-agent&type=Date)](https://www.star-history.com/#langfengQ/verl-agent&Date)
