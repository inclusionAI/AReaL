<h1 align="center">
<em>AReaL</em>: A fully open-sourced and inclusive RL project for large reasoning models
</h1>

AReaL (Ant Reasoning RL) is an open-source and efficient reinforcement learning system developed at **the RL Lab, Ant Research**. AReaL inherits and adapts the Open-Source Project [ReaLHF](https://github.com/openpsi-project/ReaLHF) for training Large Reasoning Models (LRMs) that everyone can reproduce and contribute to. AReaL is part of our efforts from Ant Research to develop tools and systems for a fully open and inclusive AGI world.

**AReaL Highlights**  
-  🛠️ **Open & Reproducible**: We will continuously release *all code, datasets, and training recipes* for training LRMs --- no hidden secrects or proprietary barriers.  
-  🚀 **Scalable Performance**: AReaL can seamlessly adapt to different computational resource settings, ranging from 1 single node to hundreds of GPUs.
-  🌍 **Community-Driven AGI**: With a fully open-source commitment, we hope our efforts can benefit the entire community to accelerate AGI research.

---

## News

[2025/02/24] Our initial release includes reproducible experiments based on our AReaL system for 1.5B and 7B LRMs, validated across diverse computational budgets. We will show that with AReaL, users can
1. **reliably train a 1.5B distilled model** with RL to **surpass o1-Preview on math reasoning within 40 hours**. 🚀 
2. **reliably perform an R1-Zero-style experiment with a 7B model**, i.e., by running RL training on Qwen2.5-7B, and **observe emergent thinking tokens and continuous model improvement on math reasoning**. 

---

## Training a 1.5B LRM from the Distilled Model

Our experiments are conducted on 16 nodes, each equipped with 8 H800 GPUs. The results, along with the associated training curves, are presented below.

![16nodes_reward_length.png](/assets/distill_1.5b_24k_curve.png) 

*Figure 1. The training rewards and response lengths during RL training. The base model is DeepSeek-R1-Distill-Qwen-1.5B. The curves are averaged with a window size of 25.*

We follow DeepScaleR to iteratively increase the output context length. The context length starts from 8K and is increased to 16K and 24K in the subsequent training process. The training reward continuously increases during RL training. We observe that the response length first **shrinks in the 8K training stage**, and then **increases in the 16K and 24K training stages**.

These three stages progressively enhanced the performance of the base model, as demonstrated below:


|          | MATH500  | AIME 2024 | AMC 2023 | Download |
| -------- | -------- | --------- | -------- | -------- |
| o1-Preview | 81.4 |	40.0 | - | - |
| DeepSeek-R1-Distill-Qwen-1.5B | 82.8 | 28.8 | 62.9 | - |
| DeepScaleR (Official) |  87.8 | 43.1 | 73.6 | - |
| AReaL Stage 1: 8K (Ours) |85.7 | 33.2 | 74.7 | [🤗 HuggingFace](https://huggingface.co/inclusionAI/AReaL-1.5B-Preview-Stage-1)   |
| AReaL Stage 2: 16K (Ours) | 87.4 | 34.2 | 79.6 | [🤗 HuggingFace](https://huggingface.co/inclusionAI/AReaL-1.5B-Preview-Stage-2)   |
| AReaL Stage 3: 24K (Ours) |88.0 | 40.2 | 81.2 | [🤗 HuggingFace](https://huggingface.co/inclusionAI/AReaL-1.5B-Preview-Stage-3)   |

*Table 1. Evaluation on a series of competition-level mathematics benchmarks, including AIME 2024, AMC 2023, and MATH-500. The results are reported using Pass@1 accuracy, which are averaged over 32 samples for each problem and evaluated with a temperature of 0.6.*

### Reproducing the Results 💯

AReaL is designed with accessibility in mind, enabling users to effortlessly reproduce results and extend their research.
Please refer to the [tutorial](/examples/README.md) under the `examples` directory for step-by-step guidelines.

- **Training:** The model checkpoints from different stages are available at HuggingFace (Please refer Table 1 for the link). With these intermediate checkpoints for all three stages, users can start from any stage to advance their own investigations. 

```bash
# Download the dataset
DATA_PATH=/storage/datasets/
cd $DATA_PATH
wget https://huggingface.co/datasets/inclusionAI/AReaL-RL-Data/resolve/main/data/prompts_for_r1_distilled.jsonl?download=true
wget https://huggingface.co/datasets/inclusionAI/AReaL-RL-Data/resolve/main/data/id2info.json?download=true

# Training in a Ray cluster with 16 nodes

# stage 1
MODEL_PATH=${path_to_DeepSeek-R1-Distill-Qwen-1.5B}
bash ./examples/train_1.5B_n16_on_ray.sh $MODEL_PATH $DATA_PATH/prompts_for_r1_distilled.jsonl $DATA_PATH/id2info.json 8192 

# stage 2
MODEL_PATH=${model_path_from_stage_1}
bash ./examples/train_1.5B_n16_on_ray.sh $MODEL_PATH $DATA_PATH/prompts_for_r1_distilled.jsonl $DATA_PATH/id2info.json 16384

# stage 3
MODEL_PATH=${model_path_from_stage_2}
bash ./examples/train_1.5B_n16_on_ray.sh $MODEL_PATH $DATA_PATH/prompts_for_r1_distilled.jsonl $DATA_PATH/id2info.json 24000

```

- **Evaluation:** We generate samples using vLLM with `enforce_eager=True`, which is critical to the final performance particularly when generating sequences longer than 16K.

```bash
# Evaluation
cd evaluation
python3 eval_and_aggregate.py --model_path $MODEL_PATH --max_gen_tokens 32768
```

### RL Training Details

Our RL training process draws on open-source experiences from [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1), [DeepScaleR](https://github.com/agentica-project/deepscaler), and [ReaLHF](https://github.com/openpsi-project/ReaLHF). Here we show our RL training setup:

- **Base Model**: We use [`DeepSeek-R1-Distill-Qwen-1.5B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) as the base model.
- **Dataset**: The RL training dataset consists of 40k high-quality mathematical reasoning tasks released by [DeepScaleR](https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset). We are also actively developing better datasets suitable for training stronger and larger models in future releases.
- **Reward Design**: We assign rewards of +5 for correct responses and -5 for incorrect responses. The correctness of responses is evaluated by comparing the responses with the answers using Sympy, following [the evaluation protocol of Qwen](https://github.com/QwenLM/Qwen2.5-Math/tree/main/evaluation).
- **RL Algorithm**: We adopt a simplified variant of PPO as the RL algorithm. We make a major change to PPO by eliminating the critic to reduce computation resources. We empirically observe that the training performance does not degenerate even without the critic. 
- **Key Hyper-parameters**: We set both the discount factor $\gamma$ and the GAE parameter $\lambda$ to 1. Such practices are also adopted by the [Open-Reasoner-Zero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero) project. We also set the coefficient for the KL rewards sufficiently small for effective training.
- **Iterative Context Lengthening**: We follow [DeepScaleR](https://github.com/agentica-project/deepscaler) to iteratively increase the context length, which results in a three-stage training process.
- **Large Batch Size**: We use a larger batch size for RL training than previous works. In each RL training step, we select 1024 mathematical questions and generate 8 rollouts for each question.

### AReaL with Various Computational Resources

AReaL can easily adapt to different computational resource settings. With more training computes, AReaL can significantly enhance the speed of RL training, leading to a notable acceleration in research progress.

We have listed detailed hardware requirements and the approach to setting up environments under different computation resources. Users can easily configure your cluster and launch your training trials [following our tutorials](/examples/README.md).

![hours.png](/assets/1.5b_time_n1n4n16.png) 

*Figure 2. Total time consumption for RL training under varying computational resource settings for 10 epochs.*

## Qwen2.5-7B-Zero RL Training

We start RL training from the base model Qwen2.5-7B with DeepSeek-R1-Zero style training. The initial training curves and results are presented below. We conducted experiments on the data released by [Open-Reasoner-Zero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero). As the training progresses, both the rewards and the response lengths gradually grow. A similar trend is also revealed [Open-Reasoner-Zero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero). **The simultaneous growth of average response lengths and rewards** can be considered as a critical sign of **the emergent deep thinking capabilities of LRM** to solve challenging reasoning problems.


![undefined](/assets/7b_zero_training_curve.png) 
*Figure 3. The training curve of Qwen2.5-7B-Zero during RL training*


We evaluate the performances of the intermediate checkpoints on the MATH500 and AIME24 datasets in the following figure. It appears that both the accuracy and the generated length continue to show an upward trend.

![undefined](/assets/7b_zero_eval_acc.png)
*Figure 4. The test accuracy and response length evaluated on the MATH500 and AIME24 datasets.*

We also conduct a experiment on the [DeepScalseR](https://github.com/agentica-project/deepscaler) dataset, which demonstrates a similar training trend. The evaluation results are presented in the following table.

|          | MATH-500  | AIME 2024 | AMC 2023 |
| -------- | -------- | --------- | -------- |
| Qwen2.5-7B | 34.0 | 2.0 | 17.8 |
| Open-Reasoner-Zero-7B, Step=150 | ~77 | ~15 | - |
| Qwen2.5-7B-Zero,    Step=150 Dataset=DeepScaleR | 75.9 | 13.9 | 56.1 |
| Qwen2.5-7B-Zero,    Step=150 Dataset=ORZ | 77.3 | 13.9 | 57.1 |
| Qwen2.5-7B-Zero,    Step=200 Dataset=ORZ | 78.0 | 14.5 | 58.7 |
Table 2. Evaluation on AIME 2024, AMC 2023, and MATH-500. The results are reported using Pass@1 accuracy, which are averaged over 32 samples for each problem and evaluated with a temperature of 1.0.

- **Training:** 

```bash
# Training in a Ray cluster with 16 nodes
DATA_PATH=/storage/datasets/
cd $DATA_PATH
wget https://huggingface.co/datasets/inclusionAI/AReaL-RL-Data/resolve/main/data/prompts_for_zero.jsonl?download=true
wget https://huggingface.co/datasets/inclusionAI/AReaL-RL-Data/resolve/main/data/id2info.json?download=true

MODEL_PATH=${path_to_Qwen2.5-7B}
bash ./examples/train_7B_zero_n16_on_ray.sh $MODEL_PATH $DATA_PATH/prompts_for_zero.jsonl $DATA_PATH/id2info.json 24000

```

- **Evaluation:** 

```bash
# Evaluation

cd evalutation
python3 eval_and_aggregate.py --model_path $MODEL_PATH --max_gen_tokens 32768 --prompt_type orz
```

## Future Plan

AReaL is under active development. We will have major releases in a weekly manner. We also highly appreciate efforts from the community as well. Here we highlight our future research and development plan. 

### System Development
In addition to our continuous efforts to make AReaL more stable, we also plan on the following major improvements. 
- [ ] Support for the latest vLLM, sglang, and megatron-core packages.
- [ ] Support RL training with Coding problems.
- [ ] Support RL training for larger MOE models.
- [ ] Optimizations for distributed training: expert parallel and zero-bubble pipelining.
- [ ] RL for vision-language models (VLM).
- [ ] Function calling and agent capabilities.

### Open Research Questions on Algorithm and Model

Based on AReaL, our team is also actively exploring the research frontier of LRMs. We highlight some of the major research topics we are working on.

- **Effective Training Datasets for LRMs**  How do we design training tasks that enable LRMs to progressively evolve sophisticated reasoning skills via RL? 
	- For more advanced distilled models, such as DeepSeek-R1-Distill-Qwen-7B and DeepSeek-R1-Distill-Qwen-32B, what kind of task distribution could drive further enhancements?
 	- How to properly curate the distillation dataset for cold-start to achieve the best RL results?
-  **Boosting Reasoning with Auxiliary Training Signals.** Can auxiliary signals — such as Process Reward Models (PRMs) and expert-curated labels — be adopted in RL training to refine the reasoning processes of LRMs?   
- **Adaptive Response Lengths: Quality Over Quantity.** How can RL training schemes teach LRMs to dynamically adjust response complexity based on task difficulty? The goal is to eliminate redundant "over-reasoning" for simpler inputs while preserving rigor for challenging problems.
- **Sample Efficient RL for LRMs** In addition to system-level acceleration, can we further speed up the LRM training process with a more sample-efficient advanced RL algorithm? Can we get the critic back to reduce gradient variance without compromising the computation efficiency? How to better encourage exploration?


## Acknowledgement

We would like to remark that major contributors are from the **Institute for Interdisciplinary Information Sciences, Tsinghua University**.

Our team has received invaluable assistance from the Super Computing Technology (SCT) team at Ant Group, particularly in the realm of large-scale cluster operations and maintenance. Their expertise and support have been instrumental in our efforts, and we are deeply grateful for their contributions.

We also appreciate all the pioneer works from the community, particularly the [ReaLHF](https://github.com/openpsi-project/ReaLHF) project from OpenPsi Inc. and those other projects, including but not limited to, [DeepScaleR](https://github.com/agentica-project/deepscaler), [Open-Reasoner-Zero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/tree/main), [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [VeRL](https://github.com/volcengine/verl).

## Citation

```
@article{mei2024realhf,
  title={ReaLHF: Optimized RLHF Training for Large Language Models through Parameter Reallocation},
  author={Mei, Zhiyu and Fu, Wei and Li, Kaiwei and Wang, Guangju and Zhang, Huanchen and Wu, Yi},
  journal={arXiv preprint arXiv:2406.14088},
  year={2024}
}
```

```
@misc{areal2025,
  author = {RL Lab, Ant Research},
  title = {AReaL: Ant Reasoning RL},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/inclusionAI/AReaL}},
}
```
