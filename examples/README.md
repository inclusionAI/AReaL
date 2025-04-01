# Improving LLM's Reasoning Capabilities with AReaL: A Complete Guide

# Prerequisites 
## Hardware Requirements
Check if your hardware meets these minimum requirements:


|**Model Size**| **1.5B** |**1.5B**|**1.5B**|          **7B**           | **7B** | **32B** |
|---|:---:|:---:|:---:|:-------------------------:|:---:|:---:|
| **Nodes** | **1** | **4** | **16** |           **4**           | **16** | **16** |
| GPU | 8x H800 |8x H800 per node| 8x H800 per node |     8x H800 per node      | 8x H800 per node | 8x H800 per node |
| CPU | 48 cores |48 cores per node|48 cores per node|     48 cores per node     | 48 cores per node| 48 cores per node|
| Memory | 1 TB |1 TB per node|1 TB per node|       1 TB per node       | 1 TB per node| 1 TB per node|
| Network | NVSwitch |NVSwitch + RoCE 3.2 Tbps|NVSwitch + RoCE 3.2 Tbps| NVSwitch + RoCE 3.2 Tbps  | NVSwitch + RoCE 3.2 Tbps| NVSwitch + RoCE 3.2 Tbps|
| Storage | 1TB |Shared storage (NAS) 10TB|Shared storage (NAS) 10TB| Shared storage (NAS) 10TB |Shared storage (NAS) 10TB| Shared storage (NAS) 10TB|
| batchsize*groupsize | 512x16 | 512x16 | 512x16 | 512x16  | 512x16 | 512*32|
| **Single-step Time (seconds)** | **3461** | **997** | **391** |         **2275**     | **815** | **6707**|
| **#Steps Until Convergence**           | **~250**  |**~250**  |**~250**  |**~400**  |**~400**  | -  |
| **Total Time (Hours)**          |   **~240**   |   **~69**    |   **~27**    |  **~252**  |  **~90**   | - |

Notes:
- GPUs need to have 80GB memory. Other GPU models with similar specs are acceptable.
- Single-node training can use local storage, but multi-node training requires shared storage.
- We haven't successfully train a powerful 32B model, so we cannot estimation the required steps and time.

## Software Requirements
This tutorial provides a Docker image. Below are the tested software versions:

| | Version |
|---|:---:|
| OS | CentOS 7 / Ubuntu 22.04 or any other system that meets the software requirements below |
| NVIDIA Driver | 550.127.08 |
| CUDA | 12.5 |
| Git LFS | Refer to: https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage. Mainly used for downloading models, datasets, and AReaL project code. |
| Docker | 27.5.1 |
|NVIDIA Container Toolkit|[Installing the NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)|
| AReaL Image | `ghcr.io/inclusionai/areal-runtime:v0.2.0`. This image includes AReaL's runtime dependencies and Ray components. |

Since the installation of NVIDIA Drivers and CUDA, as well as the mounting of shared storage, depends on node configurations and system versions, please complete these installations independently. This tutorial does not cover their setup.

For multi-node training, ensure that the shared storage is mounted to the `/storage` directory on every node. All subsequent downloads and resources will be stored in this directory. The AReaL container will also mount this directory to `/storage` within the container, enabling seamless access during training.


# One-Click Environment Setup and Training Launch

This section provides a one-click setup script to automatically configure the node environment:

1. Install Docker, Git LFS, and NVIDIA Container Toolkit
2. Pull the AReaL image on each node
3. Download AReaL code, models, and datasets
4. Set up a Ray cluster
5. [Optional] Launch a training task within the Ray cluster

Please perform the following operations on any chosen node:

```bash
mkdir -p /storage/codes
cd /storage/codes/
git clone https://github.com/inclusionAI/AReaL.git
cd /storage/codes/AReaL

python ./examples/env/setup_env_and_start_train.py setup --private_key_file /path/to/ssh_key --ssh_port 22 --username root --hostnames NODE_IP_1 NODE_IP_2 NODE_IP_3 NODE_IP_4 --train_param 1.5B_n1
```

`setup_env_and_start_train.py setup` arguments：

- `private_key_file`: SSH secret key. Using by connecting nodes.
- `ssh_port`: SSH port
- `username`: SSH username
- `hostnames`: IP list. Split with space. Can be 1, 4, or 16 node IPs
- `train_param`: [Optional] Training parameters used to launch a training task immediately after environment setup. Valid options are: `1.5B_n1`, `1.5B_n4`, `1.5B_n16`, `7B_n4`, `7B_n16`

If the script in this section fails to execute or encounters errors due to environmental discrepancies, you may manually configure the environment and launch training by following the instructions in the subsequent sections of this tutorial.

# Environment Setup

Since shared storage is used, downloading only needs to be done on one node.

## Code
Clone the AReaL project code to `/storage/codes`:


```bash
mkdir -p /storage/codes
cd /storage/codes/
git clone https://github.com/inclusionAI/AReaL
```

## Dataset

We provide a dataset for training. Download the dataset and place it in `/storage/datasets/`:

```bash
mkdir -p /storage/datasets/
cd /storage/datasets/
wget https://huggingface.co/datasets/inclusionAI/AReaL-RL-Data/resolve/main/data/boba_106k_0319.jsonl?download=true
wget https://huggingface.co/datasets/inclusionAI/AReaL-RL-Data/resolve/main/data/orz-zero_56k_0319.jsonl?download=true
```

## Model

We train based on open-source models, which can be downloaded directly from HuggingFaceHub (Please ensure that Git LFS is installed):

```
mkdir -p /storage/models
cd /storage/models
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
cd DeepSeek-R1-Distill-Qwen-7B
git lfs pull
```

You can also use the HuggingFace CLI to download after installing PyPI and huggingface_hub. Refer to the [official documentation](https://huggingface.co/docs/huggingface_hub/guides/cli) for details.

## Launch the Ray Cluster

Before proceeding, pull the AReaL environment image, which already includes Ray components.

On the first node, start the Ray Head with the following command:

```bash
docker run -d --name r1-ray-head --privileged --gpus all --network host --shm-size 700g -v /storage:/storage ghcr.io/inclusionai/areal-runtime:v0.2.0 /bin/bash -c "ray start --head --port=6379 && tail -f /dev/null"
```

On all other nodes, start the Ray Worker with the following command (skip this step if you only have one node):

```bash
# RAY_HEAD_IP is the IP of the first node
RAY_HEAD_IP=xxx.xxx.xxx.xxx
docker run -d --name r1-ray-worker --privileged --gpus all --network host --shm-size 700g -v /storage:/storage ghcr.io/inclusionai/areal-runtime:v0.2.0 /bin/bash -c "ray start --address=$RAY_HEAD_IP:6379 && tail -f /dev/null"
```

Once all nodes are up, check the Ray cluster status by entering the container on the first node:

```bash
docker exec -it r1-ray-head bash
ray status
```

You should see the Ray resource status. The output will vary depending on your node count (e.g., a 16-node, 128-GPU cluster will show the following results).

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

# RL Trainig

Before starting distributed training, ensure the Ray cluster is up and running properly.
Then, on the first node (where the Ray Head is located), enter the container:

```
docker exec -it r1-ray-head bash
cd /storage/codes/AReaL
```

Choose a config file that matches your hardware environment and run it:

```bash
python3 -m realhf.apps.quickstart ppo-math --config ./examples/configs/7B-distill/ppo-7B-distill-gpus-128.yaml
```

After starting, check the training launch information:

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

If errors occur during execution (e.g., keywords like "Error" appear), refer to the troubleshooting section.

## Commandline Options

```bash
python3 -m realhf.apps.quickstart ppo-math --help
```

The descriptions of the important parameters are as follows:


+ `mode`: It is always `ray`, and do not change it to other values when referring to this tutorial for training.
+ `{actor|critic|ref}.path`: The path of the model.
+ `dataset.path`: The path of the dataset jsonl file
+ `external_configs.cluster_config`: Set config for cluster_config. e.g. fileroot is the root path for saving traning outputs.

+ `n_nodes`: The number of nodes
+ `n_gpus_per_node`: The number of GPUs per node
+ `allocation_mode`: The GPU allocation and 3D parallel strategy of the model in the experiment, mainly in the following form:
	+ `sglang.d${DP1}m${TP1}p${PP1}+d${DP2}m${TP2}p${PP2}`: Configure the parallel strategies for SGLang generation and training respectively. The generation and training use disjoint sets of GPUs, and the sum of the number of GPUs used by the two should be equal to the total number of GPUs, i.e DP1xTP1xPP1+DP2xTP2xPP2=#GPUs.

+ `exp_ctrl.total_train_epochs`: The number of training epochs (i.e., the number of times to iterate over the entire dataset)
+ `exp_ctrl.save_freq_{epochs|steps|secs}`: The frequency of saving the model parameters in persistent storage. If it is set to null, the model will not be saved.
+ `exp_ctrl.ckpt_freq_{epochs|steps|secs}`: The frequency of saving temporary parameters for restart
+ `dataset.train_bs_n_seqs`: The training batch size, that is, the number of prompts to be sampled each time during training
+ `group_size`: The number of answers to be sampled for each prompt
+ `{actor_train|ref_inf}.mb_spec.max_tokens_per_mb`: The maximum number of tokens in the data for each forward/backward pass during the inference of the reference model and the training of the actor model. It can be reduced to avoid OOM errors. These data will accumulate gradients for a single parameter update.
+ `ppo.ppo_n_minibatches`: The number of parts into which all the data will be divided for each PPO update to calculate the loss and update the parameters.
+ `ppo.gen.max_new_tokens`: The maximum number of tokens to be generated for a single prompt, default to 16k.
+ `ppo.gen.min_new_tokens`: The minimum number of tokens to be generated for a single prompt, default to 0.

## Monitoring the Training Process

Here, we use the logs from a 16-node run (the same applies to 1-node and 4-node setups) to explain several methods for observing training progress and results.

### Training Progress

Search for the keyword `Epoch` in the logs to see the total number of Epochs and Steps:

```bash
(master_worker/0 pid=96390, ip=xxx.xxx.xxx.xxx) 20250222-11:11:56.997 master worker INFO: Epoch 1/1 step 1/19 (global step 1) finishes. Average #tokens per batch is 111847. #End to end# execution time: *2124.429*s. Total time consumption: 2283.862s. 
(master_worker/0 pid=96390, ip=xxx.xxx.xxx.xxx) 20250222-11:52:02.719 master worker INFO: Epoch 1/1 step 2/19 (global step 2) finishes. Average #tokens per batch is 111847. #End to end# execution time: *2405.716*s. Total time consumption: 4689.584s. 
(master_worker/0 pid=96390, ip=xxx.xxx.xxx.xxx) 20250222-12:27:25.084 master worker INFO: Epoch 1/1 step 3/19 (global step 3) finishes. Average #tokens per batch is 111847. #End to end# execution time: *2122.318*s. Total time consumption: 6811.949s. Estimated remaining time: 33957.093s. 
(master_worker/0 pid=96390, ip=xxx.xxx.xxx.xxx) 20250222-13:05:58.246 master worker INFO: Epoch 1/1 step 4/19 (global step 4) finishes. Average #tokens per batch is 111847. #End to end# execution time: *2313.134*s. Total time consumption: 9125.111s. Estimated remaining time: 33265.891s. 
(master_worker/0 pid=96390, ip=xxx.xxx.xxx.xxx) 20250222-13:44:14.349 master worker INFO: Epoch 1/1 step 5/19 (global step 5) finishes. Average #tokens per batch is 111847. #End to end# execution time: *2296.076*s. Total time consumption: 11421.214s. Estimated remaining time: 31413.800s. 
(master_worker/0 pid=96390, ip=xxx.xxx.xxx.xxx) 20250222-14:22:33.864 master worker INFO: Epoch 1/1 step 6/19 (global step 6) finishes. Average #tokens per batch is 111847. #End to end# execution time: *2299.448*s. Total time consumption: 13720.729s. Estimated remaining time: 29350.673s.
```

Six log entries are found. We explain the meaning of each field based on the last entry:
  - `Epoch 1/1`: Indicates that a total of 1 Epoch is required, and the first Epoch is currently being trained. This example only trains for 1 Epoch. Normally, training should run for 10 Epochs or more.
  - `step 6/19`: Indicates that the current Epoch has 19 Steps, and the 6th Step has just finished.
  - `global step 6`: Represents the step count across all Epochs.
  - `#End to end# execution time: *2299.448*s`: Indicates that the current Step took 2299.448 seconds to complete.
  - `Total time consumption: 13720.729s`: The total time elapsed since training started is 13720.729 seconds.
  - `Estimated remaining time: 29350.673s`: The estimated time remaining to complete training is 29350.673 seconds.


### Model Performance

Search for the keyword `task_reward` in the logs.


```bash
(master_worker/0 pid=96390, ip=xxx.xxx.xxx.xxx) 20250222-11:11:56.991 master worker INFO: RPC name actor_train returns {'ppo_approx_kl': -2.2640759198111482e-05, 'actor_loss': 1.1128166761409375e-06, 'actor_clip_ratio': 2.1122002635820536e-07, 'importance_weight': 1.0000014305114746, 'task_reward': -0.2996826171875, 'kl_reward': -2.27004832709099e-07, 'final_reward': -0.30145370960235596, 'advantage': 0.003593671601265669, 'avg_seq_len': 7907.8955078125, 'avg_prompt_len': 105.845703125, 'n_tokens': 127828786.0, 'n_valid_tokens': 127828786.0, 'n_seqs': 16384.0, 'no_eos_ratio': 0.122802734375, 'disable_value': 1.0, 'mask_no_eos_with_zero': 0.0}
(master_worker/0 pid=96390, ip=xxx.xxx.xxx.xxx) 20250222-11:52:02.712 master worker INFO: RPC name actor_train returns {'ppo_approx_kl': -2.493159263394773e-05, 'actor_loss': -3.846728588996484e-07, 'actor_clip_ratio': 3.16789424914532e-07, 'importance_weight': 0.9999996423721313, 'task_reward': -0.6793212890625, 'kl_reward': -2.536311853873485e-07, 'final_reward': -0.6813737154006958, 'advantage': 0.004844569601118565, 'avg_seq_len': 8203.9453125, 'avg_prompt_len': 111.892578125, 'n_tokens': 132580185.0, 'n_valid_tokens': 132580185.0, 'n_seqs': 16384.0, 'no_eos_ratio': 0.13812255859375, 'disable_value': 1.0, 'mask_no_eos_with_zero': 0.0}
(master_worker/0 pid=96390, ip=xxx.xxx.xxx.xxx) 20250222-12:27:25.077 master worker INFO: RPC name actor_train returns {'ppo_approx_kl': -2.572356243035756e-05, 'actor_loss': -5.036404786551429e-07, 'actor_clip_ratio': 1.8960582792715286e-07, 'importance_weight': 0.9999992251396179, 'task_reward': -0.6280517578125, 'kl_reward': -2.988609537624143e-07, 'final_reward': -0.6303607225418091, 'advantage': 0.004505862481892109, 'avg_seq_len': 7834.6328125, 'avg_prompt_len': 108.900390625, 'n_tokens': 126578395.0, 'n_valid_tokens': 126578395.0, 'n_seqs': 16384.0, 'no_eos_ratio': 0.11761474609375, 'disable_value': 1.0, 'mask_no_eos_with_zero': 0.0}
(master_worker/0 pid=96390, ip=xxx.xxx.xxx.xxx) 20250222-13:05:58.239 master worker INFO: RPC name actor_train returns {'ppo_approx_kl': -2.4861981728463434e-05, 'actor_loss': 1.3935685672095133e-07, 'actor_clip_ratio': 3.02603467616791e-07, 'importance_weight': 0.9999998807907104, 'task_reward': -0.78857421875, 'kl_reward': -3.672174671009998e-07, 'final_reward': -0.791388750076294, 'advantage': 0.005053278990089893, 'avg_seq_len': 7773.39404296875, 'avg_prompt_len': 108.7890625, 'n_tokens': 125576883.0, 'n_valid_tokens': 125576883.0, 'n_seqs': 16384.0, 'no_eos_ratio': 0.117919921875, 'disable_value': 1.0, 'mask_no_eos_with_zero': 0.0}
(master_worker/0 pid=96390, ip=xxx.xxx.xxx.xxx) 20250222-13:44:14.342 master worker INFO: RPC name actor_train returns {'ppo_approx_kl': -2.516058702894952e-05, 'actor_loss': -7.665488510610885e-07, 'actor_clip_ratio': 1.9505058901359007e-07, 'importance_weight': 0.9999997615814209, 'task_reward': -0.6158447265625, 'kl_reward': -4.6867208425283025e-07, 'final_reward': -0.6195111274719238, 'advantage': 0.004475570283830166, 'avg_seq_len': 7928.50830078125, 'avg_prompt_len': 105.517578125, 'n_tokens': 128171874.0, 'n_valid_tokens': 128171874.0, 'n_seqs': 16384.0, 'no_eos_ratio': 0.12353515625, 'disable_value': 1.0, 'mask_no_eos_with_zero': 0.0}
(master_worker/0 pid=96390, ip=xxx.xxx.xxx.xxx) 20250222-14:22:33.857 master worker INFO: RPC name actor_train returns {'ppo_approx_kl': -2.4821250917739235e-05, 'actor_loss': -3.922649227661168e-07, 'actor_clip_ratio': 3.323623900541861e-07, 'importance_weight': 1.0000001192092896, 'task_reward': -0.7025146484375, 'kl_reward': -5.863367960046162e-07, 'final_reward': -0.7071446776390076, 'advantage': 0.004277692176401615, 'avg_seq_len': 8002.4873046875, 'avg_prompt_len': 105.951171875, 'n_tokens': 129376851.0, 'n_valid_tokens': 129376851.0, 'n_seqs': 16384.0, 'no_eos_ratio': 0.12286376953125, 'disable_value': 1.0, 'mask_no_eos_with_zero': 0.0}
```

The last entry is used to explain the meaning of key fields:
  - `task_reward`: The average reward value of all sampled answers in this step. This value should steadily increase during training and eventually stabilize.
  - `importance_weight`: The average importance sampling ratio across all tokens in the PPO loss. This value is typically close to 1.
  - `actor_clip_ratio`: The ratio of tokens clipped in the PPO loss to the total number of tokens. This is usually less than 0.1.
  - `actor_loss`: The PPO loss. **It does not show a clear upward or downward trend during training** and should not be used as a reference for model performance.
  - `avg_seq_len`: The average length of all sequences (i.e., prompts with sampled answers) in this step. In a full multi-stage training process, this value will first decrease and then increase.
  - `no_eos_ratio`: The ratio of sampled answers truncated due to exceeding the maximum generation length. An increase in this value indicates that the average length of answers is increasing.

# Evaluation

## Evaluation Process

The evaluation code is located in the `evaluation` folder of the repository. As per the previous tutorial, the trained checkpoints will be saved under the path `/storage/ray/experiments/checkpoints/root/`, for example, `/storage/ray/experiments/checkpoints/root/ppo-zero-distill-7B-n16/1024x16-n16/actor/epoch1epochstep20globalstep20/`.

Start a new container to execute the evaluation script (note: evaluation requires updates to certain Python libraries; avoid using the training container for this task):
```
docker run -d --name r1-eval --privileged --gpus all --network host --shm-size 700g -v /storage:/storage ghcr.io/inclusionai/areal-runtime:v0.2.0 /bin/bash -c "tail -f /dev/null"
docker exec -it r1-eval bash
```

Run the following script inside the Docker container to evaluate:

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

+ `--model_path`: Path to the saved model parameters.
+ `--output_path`: Path to store the generated answers and log files during evaluation.
+ `--data_names`: Specify the dataset(s) to evaluate. Multiple datasets can be separated by commas. Default is `math_500, math, gsm8k, train_amc_aime, aime24, amc23`.
+ `--max_gen_tokens`: Maximum length of generated answers. Default is `32768`.

## Evaluation Results

The evaluation script will output a table in the terminal, for example:

```
+----------+---------------+---------------+---------------+------------+---------------+--------+---------+
| dataset  | num_questions | greedy_length | sample_length | greedy_acc | sample_pass@1 | pass@8 | pass@16 |
+----------+---------------+---------------+---------------+------------+---------------+--------+---------+
| math_500 |      500      |     6757.4    |     4139.5    |    84.4    |      92.7     |  97.3  |   97.7  |
|  aime24  |       30      |    19328.0    |    13663.5    |    50.0    |      50.4     |  77.3  |   80.0  |
|  amc23   |       40      |     8850.0    |     6526.2    |    80.0    |      90.5     |  96.8  |   98.8  |
+----------+---------------+---------------+---------------+------------+---------------+--------+---------+
```


+ `{greedy|sample}_length`: Average answer length under greedy or random sampling strategy.
+ `greedy_acc`: Average accuracy under greedy sampling.
+ `sample_pass@{k}`: Probability of generating a correct answer on average per `k` attempts under random sampling.

## Additional Notes

### Key Parameters

+ The evaluation script defaults to taking the average of 32 samples with temperature 0.6.
+ We observed that the `enforce_eager` parameter in vLLM significantly impacts evaluation performance. When `enforce_eager=True`, we can reproduce the model performance reported in previous work. Otherwise, the evaluation results may fall below the reported performance. Therefore, we enforce `enforce_eager` to be enabled during evaluation.

Due to the above reasons, the evaluation process typically takes a considerable amount of time.

### Runtime

The runtime of the evaluation depends on factors such as the maximum generation length, the number of questions in the dataset, and the model size. On a machine with 8x H100 GPUs, evaluating `aime` and `math_500` takes approximately 80 minutes and 160 minutes, respectively.

# Troubleshooting

If the following content does not address your issue, feel free to raise a GitHub Issue.


## Automatic Recover

When setting `recover_mode=auto` and the experiment config remains the same, AReaL will try to discover previous checkpoints and recover the experiment from it.

If the automatic recover fails, please check the following possibilities:

* The `experiment_name` and `trial_name` in the training script differ from the previous run.

* Changes in Batch Size (`dataset.train_bs_n_seqs` in the parameters), Group Size (`group_size` in the parameters), or the number of nodes (`n_nodes` in the parameters).

* No recover checkpoint was created in the previous run. By default, recover checkpoints are generated under two conditions:

	* After the completion of the second Step.

	* When a Step completes and more than 600 seconds have passed since the last recover checkpoint. This parameter is in the `./examples/configs/*/*.yaml`, named `exp_ctrl.ckpt_freq_secs=600`.

You can confirm if a recover checkpoint was generated by searching in the log:

```bash
(master_worker/0 pid=96390, ip=xxx.xxx.xxx.xxx) 20250222-11:52:02.760 master worker INFO: Dumped recover info to file.
(master_worker/0 pid=96390, ip=xxx.xxx.xxx.xxx) 20250222-12:27:25.105 master worker INFO: Dumped recover info to file.
(master_worker/0 pid=96390, ip=xxx.xxx.xxx.xxx) 20250222-13:05:58.264 master worker INFO: Dumped recover info to file.
(master_worker/0 pid=96390, ip=xxx.xxx.xxx.xxx) 20250222-13:44:14.411 master worker INFO: Dumped recover info to file.
(master_worker/0 pid=96390, ip=xxx.xxx.xxx.xxx) 20250222-14:22:33.883 master worker INFO: Dumped recover info to file.
(master_worker/0 pid=96390, ip=xxx.xxx.xxx.xxx) 20250222-14:59:44.925 master worker INFO: Dumped recover info to file.
```

## Series of OutOfMemory Errors

While our scripts are designed to minimize OOM (Out of Memory) errors, they can still occasionally occur, especially due to memory fragmentation and increasing sequence lengths. Although these issues are often resolved by automatic restarts, users may require the following targeted solutions.

### torch.cuda.CudaOutOfMemoryError

The key to resolving this issue is identifying the phase in which the error occurs.

- **If it occurs during initialization (before `actor_gen`):**
  - Check if there are any idle processes on the GPU. In distributed scenarios, restart the Ray cluster. In single-machine scenarios, use `pkill`.
- **This error typically does not occur during the `actor_gen` phase.**
- **If it occurs during `ref_inf` or `actor_train`:**
  - Adjust the microbatch size for the corresponding computation task. For example, set `actor_train.mb_spec.max_tokens_per_mb=20480`. This parameter limits the number of tokens per forward/backward pass and can be set as low as the maximum sequence length (including the prompt).
  - Modify the parallelism strategy (`allocation_mode`) for the 7B model. Try reducing data parallelism and increasing tensor or pipeline parallelism.

### CUDA error: out of memory

This issue may occur during vLLM's initialization of the CPU KV cache, indicating insufficient memory on the machine. To resolve this, reduce the value of `actor.vllm.swap_space`.

### RuntimeError: Aborted due to the lack of CPU swap space.

This issue arises when the sequence length and KV cache demand exceed GPU memory, and the CPU swap space is insufficient. It is closely related to [Preemption errors](https://docs.vllm.ai/en/latest/performance/optimization.html). To resolve this, increase `actor.vllm.swap_space`. If the error persists, reduce `actor.vllm.max_num_seqs` and refer to the [vLLM documentation](https://docs.vllm.ai/en/latest/performance/optimization.html).

### CUDA error: an illegal memory access was encountered

This error typically occurs during the vLLM generation phase and is another symptom of insufficient GPU memory. Solutions include:

- Reduce the training batch size or the number of answers generated per prompt. Note that this may lower sample efficiency and extend training time.
- [Switch vLLM's attention backend to xformers](https://github.com/vllm-project/vllm/issues/5376).



