# 利用AReaL提升大语言模型推理能力：中文教程

# 前置要求

## 硬件要求

为了能正常完成训练流程，请参照下表确认你的硬件是否满足要求：

| **模型大小**  | **1.5B** | **1.5B** |**1.5B** | **7B**                    |**7B** | **32B** |
|-----------|---|---|---|---------------------------|---|---|
| 节点        | 1    | 4    | 16    | 4                         | 16    | 16 |
| GPU       | 8 张 H800 |    每节点 8 张 H800    |每节点 8 张 H800    | 每节点 8 张 H800              |每节点 8 张 H800    |每节点 8 张 H800    |
| CPU       | 48 核    |   每节点 48 核    |每节点 48 核    | 每节点 48 核                  |每节点 48 核    | 每节点 48 核    |
| 内存        | 1 TB    |每节点 1 TB|每节点 1 TB    | 每节点 1 TB                  |每节点 1 TB    | 每节点 1 TB    |
| 通信        | NVSwitch |NVSwitch+RoCE 带宽 3.2 Tbps|NVSwitch+RoCE 带宽 3.2 Tbps| NVSwitch+RoCE 带宽 3.2 Tbps |NVSwitch+RoCE 带宽 3.2 Tbps| NVSwitch+RoCE 带宽 3.2 Tbps|
| 存储        | 1TB    |共享存储（NAS）10TB |共享存储（NAS）10TB | 共享存储（NAS）10TB             |共享存储（NAS）10TB | 共享存储（NAS）10TB |
| 单步训练时间（秒） | **3461** | **997** | **391** | **2275**  | **815** | **6707**|

关于硬件要求的说明：

-  GPU 需要 80GB 显存，可以选择同级别其他 GPU 型号。

-  单节点训练时可以使用本地存储，但多节点训练必须要提供共享存储，否则无法进行训练。

-  1.5B 和 7B 的训练均采用 18K 的 Context Length， 32B 的训练采用 29K 的 Context Length。


## 软件要求

本教程提供 Docker镜像。以下是经过测试的软件版本，可以参考如下软件版本进行配置。

||版本说明|
|---|---|
|OS|CentOS 7 / Ubuntu 22.04 或其他满足下方软件运行的系统|
|NVIDIA Driver|版本：550.127.08|
|CUDA|版本：12.5|
|Git LFS|参考：[Git LFS 安装指南](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage) 主要用于下载模型，数据集，AReaL 工程代码|
|Docker|版本：27.5.1|
|NVIDIA Container Toolkit|[NVIDIA Container Toolkit 安装指南](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)|
|镜像|ghcr.io/inclusionai/areal-runtime:v0.2.0 这个镜像中包含运行依赖和 Ray 的相关组件|


由于 NVIDIA Driver 和 CUDA 的安装以及共享存储的挂载与节点和系统版本有关，请自行完成安装，本教程不进行介绍。

如果是多节点训练，请先将共享存储挂载到每个节点的 `/storage` 目录上，后续下载的内容都将放在这个目录下，并且 AReaL 容器也会将该目录挂载到容器的 `/storage`，以便训练时访问。
 

# 一键搭建环境并启动训练

本节提供一个一键安装脚本，自动完成节点的环境配置工作：
1. 安装 Docker，Git LFS，NVIDIA Container Toolkit
2. 在每个节点上拉取 AReaL 镜像
3. 下载 AReaL 代码，模型，数据集
4. 搭建 Ray 集群
5. 【可选】在 Ray 集群中启动一个训练任务

请选择任意一个节点执行如下操作：

```bash
mkdir -p /storage/codes
cd /storage/codes/
git clone https://github.com/inclusionAI/AReaL.git
cd /storage/codes/AReaL

python ./examples/env/setup_env_and_start_train.py setup --private_key_file /path/to/ssh_key --ssh_port 22 --username root --hostnames NODE_IP_1 NODE_IP_2 NODE_IP_3 NODE_IP_4 --train_param 1.5B_n1
```

`setup_env_and_start_train.py setup` 参数说明：

- `private_key_file`：SSH 私钥文件，用于连接节点
- `ssh_port`：SSH 端口
- `username`：SSH 用户名
- `hostnames`：IP 列表，用空格分割。可以是 1/4/16 个节点 IP
- `train_param`：【可选】训练参数，用于在完成环境搭建后直接启动一个训练任务。可选值为 `1.5B_n1`，`1.5B_n4`，`1.5B_n16`，`7B_n4`，`7B_n16`

如果因为环境差异，无法运行本节中的脚本或运行出现错误，也可以按照本教程后续章节的内容手动完成环境配置和启动训练。

# 环境配置

由于使用了共享存储，下载操作只需要在一个节点上完成。

## 代码
将 AReaL 项目代码克隆到 `/storage/codes` 中：


```bash
mkdir -p /storage/codes
cd /storage/codes/
git clone https://github.com/inclusionAI/AReaL.git
```

## 数据集

我们提供了用于训练的数据集，请下载数据集并放置在 /storage/datasets/
```bash
mkdir -p /storage/datasets/
cd /storage/datasets/
wget https://huggingface.co/datasets/inclusionAI/AReaL-RL-Data/resolve/main/data/boba_106k_0319.jsonl?download=true
wget https://huggingface.co/datasets/inclusionAI/AReaL-RL-Data/resolve/main/data/orz-zero_56k_0319.jsonl?download=true
```

## 模型
我们基于开源模型进行训练，该模型可以从 HuggingFace Hub 直接下载（请确保已经安装了 Git LFS）：

```
mkdir -p /storage/models
cd /storage/models
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
cd DeepSeek-R1-Distill-Qwen-7B
git lfs pull
```

你也可以在安装 PyPI 和 huggingface_hub 后利用 huggingface CLI 进行下载，具体请参考[官方文档](https://huggingface.co/docs/huggingface_hub/guides/cli)


## 启动 Ray 集群

在执行这一步之前，请先拉取 AReaL 环境镜像，这个镜像中已经包含了 Ray 相关的组件。

在第一个节点上执行如下命令启动 Ray Head：

```bash
docker run -d --name r1-ray-head --privileged --gpus all --network host --shm-size 700g -v /storage:/storage ghcr.io/inclusionai/areal-runtime:v0.2.0 /bin/bash -c "ray start --head --port=6379 && tail -f /dev/null"
```

在除了第一个节点以外的每个节点上执行如下命令启动 Ray Worker（如果只有一个节点，这一步就不用执行了）：

```bash
# RAY_HEAD_IP 是第一个节点的 IP
RAY_HEAD_IP=xxx.xxx.xxx.xxx
docker run -d --name r1-ray-worker --privileged --gpus all --network host --shm-size 700g -v /storage:/storage ghcr.io/inclusionai/areal-runtime:v0.2.0 /bin/bash -c "ray start --address=$RAY_HEAD_IP:6379 && tail -f /dev/null"
```

全部启动完成后，在第一个节点上通过 docker exec 进入容器，查看 Ray 集群的状态：

```bash
docker exec -it r1-ray-head bash
ray status
```

可以看到 Ray 的资源情况，输出如下（这是一个 16 节点 128 卡的集群，根据你的节点数量，这里的输出会有所不同）：

```
======== Autoscaler status: 2025-02-22 14:08:51.061250 ========
Node status
---------------------------------------------------------------
Active:
 1 node_d5634ae61bfe6732d957811bed65c8a39f13ece07e0326f941acbc4e
 1 node_23b0c08045c9a39bc4c454cae298ee531d9a474215ac5e77a5b01e74
 1 node_bc1016320658e92645f29cecb8aaf51c0b7e01a44e8ac9c814dfee59
 1 node_4e7d15e9cee9ee0da5d65e45f1e346228c52bc0c557511c6eeab40dc
 1 node_c5bcf15e28a00515be5d2a7e8e33d71f0f57cdfaf1003db9e0c74788
 1 node_ec3f6ee8f6fdf3a5392bb4dac244668da75d094e084dcbb520ce2525
 1 node_dc2f1eef88126ae4ac7902574714af9ab74b78ba037217e73e063639
 1 node_a4728608c1fda187dc33bb24e831c42fe5c8a582ad428b6e595933bc
 1 node_970379a3ba750ee3b13e31612b6a6b758d50bd4943555b2a13d1bd61
 1 node_bf6b658bea9e437fcb642a2d881425662a689d668c92fe1545899b36
 1 node_2c69511f410d9360f1d05893fde2c97dd32240e0315afea9b2d286a3
 1 node_e4c90c17cc48ad469d123041d3302dcff1f7a82a4805279300812b19
 1 node_3f772cbffb206c30b6ccedade83789d78397804bab874ee59563cb96
 1 node_429bd5115b5590b612590bb455f2d3ed4f77055d746a184baf807655
 1 node_75071820f2c16dc51fa271316b72cd45335ec877c06450d292ab7d54
 1 node_6f4323f9038248d82b91321e2c4ca5fa99e65efa2d976c0b896a8964
Pending:
 (no pending nodes)
Recent failures:
 (no failures)

Resources
---------------------------------------------------------------
Usage:
 0.0/2128.0 CPU
 0.0/128.0 GPU
 0B/21.08TiB memory
 0B/2.91TiB object_store_memory

Demands:
 (no resource demands)
```

# RL训练

在进行分布式训练之前，请确保已经启动了 Ray 集群，并且集群状态正常。
然后在第一个节点（Ray Head 所在节点），进入容器：

```
docker exec -it r1-ray-head bash
cd /storage/codes/AReaL
```

选择匹配硬件环境的一个配置运行即可：

```bash
python3 -m realhf.apps.quickstart ppo-math --config ./examples/configs/7B-distill/ppo-7B-distill-gpus-128.yaml
```

启动后，在终端可以看到启动日志：
```
              ╭─────────────────────────────────────────────────╮               
              │ Setting PPOMATHConfig with the Following Values │               
              ╰─────────────────────────────────────────────────╯               

───────────────────────── Current Configuration Begin ──────────────────────────
actor (ModelTrainEvalConfig)
    actor.type (ModelFamily)
        actor.type._class (str) - qwen2
        actor.type.size (int) - 7
        actor.type.is_critic (bool) - False
...
────────────────────────── Current Configuration End ───────────────────────────
 
20250222-10:26:34.877 quickstart INFO: Running ppo-math experiment.
20250222-10:44:15.581 quickstart INFO: Logs will be dumped to /storage/ray/experiments/logs/root/ppo-7B-distill-gpus-128/512x16
20250222-10:44:15.581 quickstart INFO: Model checkpoints will be saved to /storage/ray/experiments/checkpoints/root/ppo-7B-distill-gpus-128/512x16
20250222-10:26:36.408 quickstart INFO: Launching experiments with RAY...
```

如果运行过程中出现错误（比如出现 Error 关键字），请参考Troubleshooting解决。

## Commandline Options

```bash
python3 -m realhf.apps.quickstart ppo-math --help
```

其中重要的参数的说明如下：

+ mode：总是为 ray，参考本教程进行训练时不要改成其他值。
+ {actor|critic|ref}.path：模型的路径
+ dataset.path：数据集 jsonl 文件的路径
+ external_configs.cluster_config：设置 cluster_config 的配置，比如 fileroot 是存放训练输出的根目录。

+ n_nodes：节点数量
+ n_gpus_per_node：每个节点的 GPU 数量
+ allocation_mode：实验中模型的 GPU 分配和 3D 并行策略，推荐的策略有以下形式:
	+ `sglang.d${DP1}m${TP1}p${PP1}+d${DP2}m${TP2}p${PP2}`: 分别配置 SGLang 生成和训练的并行策略，生成和训练分离，使用两部分不同的 GPU。二者所用的GPU数量相加要等于总的 GPU 数量，即 DP1xTP1xPP1+DP2xTP2xPP2=#GPUs。

+ exp_ctrl.total_train_epochs：训练的 epoch 数量（即迭代整个数据集的次数）
+ exp_ctrl.save_freq_{epochs|steps|secs}：保存持久化存储模型参数的频率，如果设成 null 会不保存模型
+ exp_ctrl.ckpt_freq_{epochs|steps|secs}：保存临时参数用于重启的频率
+ dataset.train_bs_n_seqs：训练的批量大小，即每次训练需要采样的 prompt 数量
+ group_size：每个 prompt 需要采样的答案数量
+ {actor_train|ref_inf}.mb_spec.max_tokens_per_mb：reference模型推理和actor模型训练每次forward/backward数据中最大的token数量，可以减小以避免OOM错误。这些数据会累积梯度进行一次参数更新。
+ ppo.ppo_n_minibatches：每次PPO更新中会把所有数据划分成多少份以此进行loss计算和参数更新。
+ ppo.gen.max_new_tokens：每条prompt生成的最大token数，默认训练脚本中为16k。
+ ppo.gen.min_new_tokens：每条prompt生成的最小token数，默认为0。


## 过程观测
这里以 16 节点的运行日志为例（1 节点和 4 节点也一样），说明几个观察训练进度和效果的方法。

### 查看训练进度

搜索日志中的 Epoch 关键字，查看总的 Epoch 数量和 Step 数量：

```bash
(master_worker/0 pid=96390, ip=xxx.xxx.xxx.xxx) 20250222-11:11:56.997 master worker INFO: Epoch 1/1 step 1/19 (global step 1) finishes. Average #tokens per batch is 111847. #End to end# execution time: *2124.429*s. Total time consumption: 2283.862s. 
(master_worker/0 pid=96390, ip=xxx.xxx.xxx.xxx) 20250222-11:52:02.719 master worker INFO: Epoch 1/1 step 2/19 (global step 2) finishes. Average #tokens per batch is 111847. #End to end# execution time: *2405.716*s. Total time consumption: 4689.584s. 
(master_worker/0 pid=96390, ip=xxx.xxx.xxx.xxx) 20250222-12:27:25.084 master worker INFO: Epoch 1/1 step 3/19 (global step 3) finishes. Average #tokens per batch is 111847. #End to end# execution time: *2122.318*s. Total time consumption: 6811.949s. Estimated remaining time: 33957.093s. 
(master_worker/0 pid=96390, ip=xxx.xxx.xxx.xxx) 20250222-13:05:58.246 master worker INFO: Epoch 1/1 step 4/19 (global step 4) finishes. Average #tokens per batch is 111847. #End to end# execution time: *2313.134*s. Total time consumption: 9125.111s. Estimated remaining time: 33265.891s. 
(master_worker/0 pid=96390, ip=xxx.xxx.xxx.xxx) 20250222-13:44:14.349 master worker INFO: Epoch 1/1 step 5/19 (global step 5) finishes. Average #tokens per batch is 111847. #End to end# execution time: *2296.076*s. Total time consumption: 11421.214s. Estimated remaining time: 31413.800s. 
(master_worker/0 pid=96390, ip=xxx.xxx.xxx.xxx) 20250222-14:22:33.864 master worker INFO: Epoch 1/1 step 6/19 (global step 6) finishes. Average #tokens per batch is 111847. #End to end# execution time: *2299.448*s. Total time consumption: 13720.729s. Estimated remaining time: 29350.673s.
```

出现了 6 条日志信息，以最后一条信息的内容说明各个字段的含义：
+ `Epoch 1/1`：表示总共需要训练 1 个 Epochs，当前在训练第 1 个。这里作为例子总共只训练 1 个 Epoch，正常训练应该是 10 个 Epochs 或者更多。
+ `step 6/19`：表示当前 Epoch 有 19 个 Steps，当前在训练第 6 个
+ `global step 6`： 表示当前 Step 在所有 Epochs 的 Steps 里的序号
+ `#End to end# execution time: *2299.448*s`：表示当前 Step 训练耗费了 2299.448 秒
+ `Total time consumption: 13720.729s`：从训练启动开始一共耗费了 13720.729 秒
+ `Estimated remaining time: 29350.673s`：预计完成训练还需要 29350.673 秒


### 查看训练的效果

搜索日志中的 `task_reward` 关键字

```bash
(master_worker/0 pid=96390, ip=xxx.xxx.xxx.xxx) 20250222-11:11:56.991 master worker INFO: RPC name actor_train returns {'ppo_approx_kl': -2.2640759198111482e-05, 'actor_loss': 1.1128166761409375e-06, 'actor_clip_ratio': 2.1122002635820536e-07, 'importance_weight': 1.0000014305114746, 'task_reward': -0.2996826171875, 'kl_reward': -2.27004832709099e-07, 'final_reward': -0.30145370960235596, 'advantage': 0.003593671601265669, 'avg_seq_len': 7907.8955078125, 'avg_prompt_len': 105.845703125, 'n_tokens': 127828786.0, 'n_valid_tokens': 127828786.0, 'n_seqs': 16384.0, 'no_eos_ratio': 0.122802734375, 'disable_value': 1.0, 'mask_no_eos_with_zero': 0.0}
(master_worker/0 pid=96390, ip=xxx.xxx.xxx.xxx) 20250222-11:52:02.712 master worker INFO: RPC name actor_train returns {'ppo_approx_kl': -2.493159263394773e-05, 'actor_loss': -3.846728588996484e-07, 'actor_clip_ratio': 3.16789424914532e-07, 'importance_weight': 0.9999996423721313, 'task_reward': -0.6793212890625, 'kl_reward': -2.536311853873485e-07, 'final_reward': -0.6813737154006958, 'advantage': 0.004844569601118565, 'avg_seq_len': 8203.9453125, 'avg_prompt_len': 111.892578125, 'n_tokens': 132580185.0, 'n_valid_tokens': 132580185.0, 'n_seqs': 16384.0, 'no_eos_ratio': 0.13812255859375, 'disable_value': 1.0, 'mask_no_eos_with_zero': 0.0}
(master_worker/0 pid=96390, ip=xxx.xxx.xxx.xxx) 20250222-12:27:25.077 master worker INFO: RPC name actor_train returns {'ppo_approx_kl': -2.572356243035756e-05, 'actor_loss': -5.036404786551429e-07, 'actor_clip_ratio': 1.8960582792715286e-07, 'importance_weight': 0.9999992251396179, 'task_reward': -0.6280517578125, 'kl_reward': -2.988609537624143e-07, 'final_reward': -0.6303607225418091, 'advantage': 0.004505862481892109, 'avg_seq_len': 7834.6328125, 'avg_prompt_len': 108.900390625, 'n_tokens': 126578395.0, 'n_valid_tokens': 126578395.0, 'n_seqs': 16384.0, 'no_eos_ratio': 0.11761474609375, 'disable_value': 1.0, 'mask_no_eos_with_zero': 0.0}
(master_worker/0 pid=96390, ip=xxx.xxx.xxx.xxx) 20250222-13:05:58.239 master worker INFO: RPC name actor_train returns {'ppo_approx_kl': -2.4861981728463434e-05, 'actor_loss': 1.3935685672095133e-07, 'actor_clip_ratio': 3.02603467616791e-07, 'importance_weight': 0.9999998807907104, 'task_reward': -0.78857421875, 'kl_reward': -3.672174671009998e-07, 'final_reward': -0.791388750076294, 'advantage': 0.005053278990089893, 'avg_seq_len': 7773.39404296875, 'avg_prompt_len': 108.7890625, 'n_tokens': 125576883.0, 'n_valid_tokens': 125576883.0, 'n_seqs': 16384.0, 'no_eos_ratio': 0.117919921875, 'disable_value': 1.0, 'mask_no_eos_with_zero': 0.0}
(master_worker/0 pid=96390, ip=xxx.xxx.xxx.xxx) 20250222-13:44:14.342 master worker INFO: RPC name actor_train returns {'ppo_approx_kl': -2.516058702894952e-05, 'actor_loss': -7.665488510610885e-07, 'actor_clip_ratio': 1.9505058901359007e-07, 'importance_weight': 0.9999997615814209, 'task_reward': -0.6158447265625, 'kl_reward': -4.6867208425283025e-07, 'final_reward': -0.6195111274719238, 'advantage': 0.004475570283830166, 'avg_seq_len': 7928.50830078125, 'avg_prompt_len': 105.517578125, 'n_tokens': 128171874.0, 'n_valid_tokens': 128171874.0, 'n_seqs': 16384.0, 'no_eos_ratio': 0.12353515625, 'disable_value': 1.0, 'mask_no_eos_with_zero': 0.0}
(master_worker/0 pid=96390, ip=xxx.xxx.xxx.xxx) 20250222-14:22:33.857 master worker INFO: RPC name actor_train returns {'ppo_approx_kl': -2.4821250917739235e-05, 'actor_loss': -3.922649227661168e-07, 'actor_clip_ratio': 3.323623900541861e-07, 'importance_weight': 1.0000001192092896, 'task_reward': -0.7025146484375, 'kl_reward': -5.863367960046162e-07, 'final_reward': -0.7071446776390076, 'advantage': 0.004277692176401615, 'avg_seq_len': 8002.4873046875, 'avg_prompt_len': 105.951171875, 'n_tokens': 129376851.0, 'n_valid_tokens': 129376851.0, 'n_seqs': 16384.0, 'no_eos_ratio': 0.12286376953125, 'disable_value': 1.0, 'mask_no_eos_with_zero': 0.0}
```

以最后一条说明其中几个重点字段的含义：
+ `task_reward`：这个step中采样的所有答案的平均奖励值，训练稳步进行的话这个值会持续上升，最终维持不变
+ `importance_weight`: PPO loss中重要性采样比率在所有token上的平均值，通常接近1。
+ `actor_clip_ratio`: PPO loss中被clip掉的token占所有token的比率，通常小于0.1。
+ `actor_loss`: PPO loss，**不会随着训练过程有明显的上升或下降趋势**，不应作为模型表现的参考。
+ `avg_seq_len`: 这一步中采样的所有序列（即提示词和答案相加）的平均长度。在完整的多阶段训练中，这个值会先下降再上升。
+ `no_eos_ratio`: 这一步中采样的所有答案因为超出最大生成长度被截断的比率。这个值上升也代表了答案的平均长度在上升。

# 评估

## 评估流程

评估代码包含在仓库的`evaluation`文件夹中。按照以上的教程，训练得到的checkpoint会保存在`/storage/ray/experiments/checkpoints/root/`路径下，例如`/storage/ray/experiments/checkpoints/root/ppo-zero-distill-7B-n16/1024x16-n16/actor/epoch1epochstep20globalstep20/`。

启动一个新的容器用于运行评估脚本（评估需要更新部分 python 库，请不要在训练容器中进行）：
```
docker run -d --name r1-eval --privileged --gpus all --network host --shm-size 700g -v /storage:/storage ghcr.io/inclusionai/areal-runtime:v0.2.0 /bin/bash -c "tail -f /dev/null"
docker exec -it r1-eval bash
```

在docker容器内部运行以下脚本进行评估：

```bash
cd /storage/codes/AReaL/evaluation
cd latex2sympy
pip install -e .
cd ..
pip install -r requirements.txt 
pip install vllm --no-build-isolation
pip install transformers==4.47.0
pip install prettytable timeout_decorator
mkdir /storage/ray/eval_output/
nohup python eval_and_aggregate.py \
	--model_path /storage/ray/experiments/checkpoints/root/ppo-zero-distill-7B-n16/1024x16-n16/actor/epoch1epochstep20globalstep20/ \
	--output_path /storage/ray/eval_output/ \
	--data_names "math_500,aime24,amc23" \
	--max_gen_tokens 32768 &> /storage/ray/eval_output/eval_and_aggregate_parallel.log &
```

+ `--model_path`：模型参数的保存路径
+ `--output_path`：评估过程中生成的答案和日志文件路径
+ `--data_names`: 可以指定评测某个数据，多个数据集用逗号隔开，默认为 math_500, aime24, amc23 
+ `--max_gen_tokens`：最长的答案生成长度，默认值 32768

## 评估结果

评估脚本运行完后会在 /storage/ray/eval_output/eval_and_aggregate_parallel.log 日志文件输出一个表格，例如：

```
+----------+---------------+---------------+---------------+------------+---------------+--------+---------+
| dataset  | num_questions | greedy_length | sample_length | greedy_acc | sample_pass@1 | pass@8 | pass@16 |
+----------+---------------+---------------+---------------+------------+---------------+--------+---------+
| math_500 |      500      |     6757.4    |     4139.5    |    84.4    |      92.7     |  97.3  |   97.7  |
|  aime24  |       30      |    19328.0    |    13663.5    |    50.0    |      50.4     |  77.3  |   80.0  |
|  amc23   |       40      |     8850.0    |     6526.2    |    80.0    |      90.5     |  96.8  |   98.8  |
+----------+---------------+---------------+---------------+------------+---------------+--------+---------+
```

+ `{greedy|sample}_length`: 在greedy或随机采样策略下生成的平均答案长度
+ `greedy_acc`：在greedy采样下的平均准确率
+ `sample_pass@{k}`：在随机采样下平均每k个答案产生正确答案的概率

## 额外说明

### 关键参数

+ 我们提供的评估脚本默认采样32次取平均值，采样温度值为0.6
+ 我们发现vLLM的`enforce_eager`参数很大程度影响评估性能，当`enforce_eager=True`时我们才能够复现先前工作汇报的模型表现，否则评估结果会低于先前工作汇报的结果，因此我们会在执行 `eval_and_aggregate_parallel.py` 时将`enforce_eager`强制开启。

由于以上原因，评估过程通常会消耗较长时间。

### 运行时间
评估的运行时间取决于最长生成长度、数据集的题目数量和模型大小等等。在1台8*H100机器上，7B模型，数据集为`math_500,aime24,amc23`，生成长度为32768，评估脚本运行时间为 5 个小时。


# Troubleshooting

如果以下内容没有解答你的问题，欢迎在 GitHub Issue 中进行提问。

## 自动恢复

当设置了 `recover_mode=auto` 并且训练配置和之前相同，AReaL 会尝试找到之前生成的 checkpoints 并且从这个 checkpoints 恢复训练。

如果自动恢复失败，有这些可能性：

+ 训练配置里的 `experiment_name` 和 `trial_name` 与之前的不一样
+ Batch Size（参数里的 `dataset.train_bs_n_seqs`），Group Size（参数里的 `group_size`），节点数（参数里的 `n_nodes`）三个值发生了变化
+ 之前的训练没有创建过 recover checkpoint 。默认的 recover checkpoint 规则有 2 个：
	+ 从第 2 个 step 完成后才生成 recover checkpoint
	+ 一个 step 训练完成，且距离上次 recover checkpoint 时间超过 600s，则生成一个新的 recover checkpoint。这个参数在 `./examples/configs/*/*.yaml` 文件里，参数名为 ：`exp_ctrl.ckpt_freq_secs=600`。


可以通过搜索 `Dumped recover` 确认是否生成过 recover checkpoint

```bash
(master_worker/0 pid=96390, ip=xxx.xxx.xxx.xxx) 20250222-11:52:02.760 master worker INFO: Dumped recover info to file.
(master_worker/0 pid=96390, ip=xxx.xxx.xxx.xxx) 20250222-12:27:25.105 master worker INFO: Dumped recover info to file.
(master_worker/0 pid=96390, ip=xxx.xxx.xxx.xxx) 20250222-13:05:58.264 master worker INFO: Dumped recover info to file.
(master_worker/0 pid=96390, ip=xxx.xxx.xxx.xxx) 20250222-13:44:14.411 master worker INFO: Dumped recover info to file.
(master_worker/0 pid=96390, ip=xxx.xxx.xxx.xxx) 20250222-14:22:33.883 master worker INFO: Dumped recover info to file.
(master_worker/0 pid=96390, ip=xxx.xxx.xxx.xxx) 20250222-14:59:44.925 master worker INFO: Dumped recover info to file.
```

## 一系列OutOfMemory错误

我们提供的脚本已经尽最大努力避免了OOM错误的发生，但是OOM问题仍然会随着训练进行，在内存碎片增加和生成序列长度越来越长时偶尔发生。虽然这些问题通常可以通过自动重启解决，当重启频繁时，用户还可以尝试以下针对性的解决方式。

### torch.cuda.CudaOutOfMemoryError

解决这个问题的关键是定位错误发生的阶段。

- 如果发生在初始化阶段（在进入到actor_gen之前）:
	- 检查当前GPU上是否存在残留进程。在分布式场景下，可以通过重启ray cluster解决；在单机场景下，可以通过pkill解决。
- 该错误通常不会发生在actor_gen阶段。
- 如果发生在ref_inf或actor_train阶段
	- 改变相应计算任务的microbatch大小，例如`actor_train.mb_spec.max_tokens_per_mb=20480`，这个参数代表每次模型forward/backward的数据最多只会包含20480个token，这个值最小可以设为生成序列的最长长度（包括prompt）
	- 改变模型的并行策略，即`allocation_mode`，可以尝试减少数据并行的大小，增加张量或流水线并行的大小。

### CUDA error: out of memory

这个问题可能会发生在vLLM初始化CPU KV cache时，表示每台机器的内存不够了。可以减小`actor.vllm.swap_space`解决。

### RuntimeError: Aborted due to the lack of CPU swap space.

问题的原因是序列长、对KV cache需求大，在GPU显存不够时KV cache会被卸载到内存，而内存中设置的swap space不够。这个问题和[Preemption的报错](https://docs.vllm.ai/en/latest/performance/optimization.html)紧密相关。解决方案是增加`actor.vllm.swap_space`，如果同样的错误出现，请减少`actor.vllm.max_num_seqs`并参考[vLLM官方文档](https://docs.vllm.ai/en/latest/performance/optimization.html)。

### CUDA error: an illegal memory access was encountered

通常会在vLLM生成阶段出现，同样是显存不足的一种表现。解决方案包括：


+ 减小训练batch size或者每个prompt生成的答案数量，但减小后会降低样本效率、延长训练时间
+ [将vLLM的attention backend换成xformers](https://github.com/vllm-project/vllm/issues/5376)

