这是 AReaL 跑 gem minesweeper 的代码

# 一些准备

需要事先编译好 `multitask_agent/gem_train/minesweeper/a.cpp`，并且把可执行文件的路径放到训练脚本里。

gem 目前有一个 bug，他判断游戏结束的标准是所以安全的格子开了，所有有雷的格子标记了。但是他只会在你开一个格子的时候进行这个判断，标记格子的时候不会进行判断。所以我对 gem 的代码做了一点修改，只要开了所有安全的格子游戏就结束。

```
diff --git a/gem/envs/game_env/minesweeper.py b/gem/envs/game_env/minesweeper.py
index 06b5791..31f8be8 100644
--- a/gem/envs/game_env/minesweeper.py
+++ b/gem/envs/game_env/minesweeper.py
@@ -260,7 +260,7 @@ class MinesweeperEnv(Env):
             bool: True if the board is in a solved state, False otherwise.
         """
         return all(
-            (self.grid[r][c] == -1 and self.flags[r][c])
+            self.grid[r][c] == -1
             or (self.grid[r][c] != -1 and self.revealed[r][c])
             for r in range(self.rows)
             for c in range(self.cols)
```

# 训练

```bash
bash scripts/launch/minesweeper-1.7b-local.sh
# or, in a ray cluster
bash scripts/launch/minesweeper-1.7b-ray-2nodes.sh
```

# 测试

## 生成 Rollout

修改 `scripts/launch/eval.sh`，填入你的 checkpoint 的路径。

```bash
bash scripts/launch/eval.sh
```

## 计算 success rate

在 log 目录下运行
```bash
python3 /path/to/project/scripts/utils/gem_sr.py generated/0
```

# 一些事情

训练使用的环境是 `+env_name=game:Minesweeper-v0-easy-with-template`，`-with-template` 的环境，同一个 group 会使用同样的开局，且开局就是走了一步的状态，固定开中间的格子。如果设置 `+minesweeper_random_first_move=True`，开局走的那一步就是随机的，但是同一个 group 里会是同一个。如果直接使用 gem 原本的环境，即 `+env_name=game:Minesweeper-v0-easy`，那么开局不会固定走一步，且同一个 group 里的 setup 也是不一样的。

我试了一下，1.7b 直接跑的话，模型会学会生成一堆不合法的 action，比如开一个已经开过的格子，开边界外的格子，导致 traj 过长。所以我在脚本里加上了 `+invalid_action_reward=-0.05`，给不合法的 action 一个 penalty。

---

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
  workflows within a single file, and smooth integration with
  [other agentic tooling frameworks](https://inclusionai.github.io/AReaL/tutorial/agentic_rl.html).
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

| Algorithm                | Documentation                         | Paper                                          | Configuration                                                  |
| ------------------------ | ------------------------------------- | ---------------------------------------------- | -------------------------------------------------------------- |
| **GRPO**                 | [📖 Docs](docs/algorithms/grpo.md)    | [📄 Paper](https://arxiv.org/pdf/2402.03300)   | [🔗 GSM8K Example](examples/math/gsm8k_grpo.yaml)              |
| **GSPO**                 | [📖 Docs](docs/algorithms/gspo.md)    | [📄 Paper](https://arxiv.org/abs/2507.18071)   | [🔗 GSM8K Example](examples/experimental/gspo/gsm8k_gspo.yaml) |
| **PPO**                  | -                                     | [📄 Paper](https://arxiv.org/pdf/2203.02155)   | [🔗 GSM8K Example](examples/math/gsm8k_ppo.yaml)               |
| **DAPO**                 | [📖 Docs](docs/algorithms/dapo.md)    | [📄 Paper](https://arxiv.org/abs/2503.14476)   | [🔗 GSM8K Example](examples/experimental/dapo/gsm8k_dapo.py)   |
| **LitePPO**              | [📖 Docs](docs/algorithms/litePPO.md) | [📄 Paper](https://arxiv.org/abs/2508.08221)   | -                                                              |
| **Dr.GRPO**              | [📖 Docs](docs/algorithms/dr.GRPO.md) | [📄 Paper](https://arxiv.org/abs/2503.20783)   | -                                                              |
| **REINFORCE++**          | -                                     | [📄 Paper](https://arxiv.org/pdf/2501.03262)   | [🔗 GSM8K Example](examples/math/gsm8k_reinforce.yaml)         |
| **RLOO**                 | [📖 Docs](docs/algorithms/rloo.md)    | [📄 Paper](https://arxiv.org/pdf/2402.14740v1) | [🔗 GSM8K Example](examples/math/gsm8k_rloo.yaml)              |
| **RLHF Reward Modeling** | -                                     | -                                              | [🔗 RLHF Example](examples/alignment/)                         |
| **SFT**                  | -                                     | -                                              | [🔗 GSM8K Example](examples/math/gsm8k_sft.py)                 |

### Models

| Model Family               | Megatron | PyTorch FSDP | Notes                                                    |
| -------------------------- | -------- | ------------ | -------------------------------------------------------- |
| **Qwen2/3**                | ✅       | ✅           | -                                                        |
| **Qwen3-MoE**              | ✅       | ✅           | -                                                        |
| **Qwen2.5-VL**             | ❌       | ✅           | Vision-language model                                    |
| **Qwen3-VL**               | ❌       | ✅           | Vision-language model                                    |
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
| **vLLM**   | ✅              | ❓               | ✅                | ❓                      | ❓              |
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
- [Asynchronous RL Explained](https://inclusionai.github.io/AReaL/algorithms/async.html)
- [Fine-Tuning Large MoE](https://inclusionai.github.io/AReaL/tutorial/megatron.html)
- [Agentic RL](https://inclusionai.github.io/AReaL/tutorial/agentic_rl.html)
- [Debugging Best Practices](https://inclusionai.github.io/AReaL/best_practices/debugging.html)
- [Handling OOM Issues](https://inclusionai.github.io/AReaL/best_practices/handling_oom.html)

### Code Walkthrough

- [Running GRPO on GSM8K dataset with AReaL-lite](https://inclusionai.github.io/AReaL/lite/gsm8k_grpo.html)

### Customization

- [Customize dataset with AReaL-lite](https://inclusionai.github.io/AReaL/customization/dataset.html)
- [Customize Agentic/RVLR rollout workflows with AReaL-lite](https://inclusionai.github.io/AReaL/customization/agent.html)
- [Customize algorithms with AReaL-lite](https://inclusionai.github.io/AReaL/customization/algorithm.html)

## 🤝 Contributing

We warmly welcome contributions from the community! Whether you're fixing bugs, adding
features, improving documentation, or helping others, your contribution is valued.
Please check our **[Contributing Guide](CONTRIBUTING.md)** for detailed information.

```bash
# Fork and clone the repository
git clone https://github.com/YOUR-USERNAME/AReaL
cd AReaL

# Install in development mode
pip install -e ".[dev]"

# Set up pre-commit hooks for automatic formatting
pip install pre-commit
pre-commit install

# Make changes
git checkout -b feat/gpt-o5
git add .
# `git commit` will automatically format your file
git commit -m "Implement gpt-o5 training loop"
git push
```

### 💬 Community & Support

- **[GitHub Discussions](https://github.com/inclusionAI/AReaL/discussions)** - Ask
  questions, share ideas, and connect with the community
- **[WeChat Group](./assets/wechat_qrcode.png)** - Join our WeChat community (微信群)
- **[Project Roadmap](ROADMAP.md)** - See what we're working on and what's planned

## 🗺️ Future Roadmap

- **[Full Roadmap](ROADMAP.md)**
- **[2025 Q4 Roadmap](https://github.com/inclusionAI/AReaL/issues/542)**

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
