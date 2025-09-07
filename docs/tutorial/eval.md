# Evaluation

## Option 1: Using AReaL's Standalone Evaluation Package

The evaluation code is located in the `evaluation` folder of the repository. Following
the previous tutorial, trained checkpoints will be saved under
`${fileroot}/checkpoints/${USER}/${experiment_name}/${trial_name}/`.

### Setup Evaluation Environment

> **Note**: Evaluation requires updates to certain Python libraries like vLLM, so avoid
> using the training container or virtual environment for this task.

From the repository directory, create a new conda environment:

```bash
conda create -n areal-eval python=3.12
conda activate areal-eval
```

Install dependencies:

```bash
bash examples/env/setup-eval-pip-deps.sh
```

### Run Evaluation

Specify an `output_path` to save the test results. If not specified, the results will be
saved in `model_path`.

#### Math Evaluation

```bash
cd evaluation
nohup python eval_and_aggregate.py \
    --model_path /path/to/checkpoint \
    --output_path /path/to/outputs \
    --max_gen_tokens 32768 \
    --data_names math_500,aime24,amc23 \
    --prompt_type qwen3-think \
    --task math &> eval_and_aggregate_parallel.log &
```

#### Code Evaluation

**Obtaining Data:**

- Due to the size of code datasets (some test cases are relatively large), we have
  uploaded all our code datasets to [Hugging Face](https://huggingface.co/inclusionAI).
- Once you have downloaded the code dataset, place it under `./evaluation/data/`.

**Running Evaluation:**

```bash
cd evaluation
nohup python eval_and_aggregate.py \
    --model_path /path/to/checkpoint \
    --output_path /path/to/outputs \
    --max_gen_tokens 32768 \
    --data_names codeforces,lcb_v5 \
    --prompt_type qwen3-think-pure \
    --temperature 1.0 \
    --top_p 0.95 \
    --num_sample_nodes 8 \
    --samples_per_node 1 \
    --n_sampling $((num_sample_nodes * samples_per_node)) \
    --task code &> eval_and_aggregate_parallel.log &
```

#### Command Line Parameters

- **`--model_path`**: Path to the saved model parameters
- **`--output_path`**: Path to store generated answers and log files during evaluation
- **`--data_names`**: Dataset(s) to evaluate. Multiple datasets can be separated by
  commas. Available options:
  - Math: `math_500`, `aime24`, `aime25`, `amc23`
  - Code: `lcb_v5`, `lcb_v5_2410_2502`, `codeforces`, `code_contest_all`
- **`--max_gen_tokens`**: Maximum length of generated answers (default: 32768)
- **`--prompt_type`**: Specify the prompt template. For our latest model, we use
  `qwen3-think` for math datasets and `qwen3-think-pure` for code datasets.
- **`--num_sample_nodes`**: Number of multiple sampling seeds to ensure sampling
  diversity.
- **`--samples_per_node`**: Number of samples to generate per seed for each problem.

### Logs and Evaluation Results

Check `${output_path}/math_eval_${max_gen_tokens}/logs` to review the log of each
worker.

The evaluation script will output a results table in the terminal:

```
+----------+---------------+---------------+---------------+------------+---------------+--------+---------+
| dataset  | num_questions | greedy_length | sample_length | greedy_acc | sample_pass@1 | pass@8 | pass@16 |
+----------+---------------+---------------+---------------+------------+---------------+--------+---------+
| math_500 |      500      |     6757.4    |     4139.5    |    84.4    |      92.7     |  97.3  |   97.7  |
|  aime24  |       30      |    19328.0    |    13663.5    |    50.0    |      50.4     |  77.3  |   80.0  |
|  amc23   |       40      |     8850.0    |     6526.2    |    80.0    |      90.5     |  96.8  |   98.8  |
+----------+---------------+---------------+---------------+------------+---------------+--------+---------+
```

#### Metrics Explanation

- **`{greedy|sample}_length`**: Average answer length under greedy or random sampling
  strategy
- **`greedy_acc`**: Average accuracy under greedy sampling
- **`sample_pass@{k}`**: Probability of generating a correct answer within `k` attempts
  under random sampling

For the Codeforces dataset, we use the Elo ranking algorithm to evaluate model
performance, referring to [CodeElo](https://github.com/QwenLM/CodeElo) and
[rllm](https://github.com/agentica-project/rllm):

```
+------------+----------------+-----------+
|  Dataset   | Percentile (%) | CF Rating |
+------------+----------------+-----------+
| codeforces |      17.9      |    590    |
+------------+----------------+-----------+
```

- **`CF Rating`**: The overall Elo rank score of the model across 57 Codeforces
  contests.
- **`Percentile`**: The Elo ranking percentile of the model among all Codeforces users.

> **Note**: As the penalty mechanism may cause fluctuations in Elo rankings, we suggest
> performing multiple evaluations and taking the average score as the final result.

### Configuration Details

#### Sampling Parameters

- The evaluation script defaults to averaging 32 samples with temperature 1.0. For the
  code dataset, we set it to 8 samples.
- We observed that the `enforce_eager` parameter in vLLM significantly impacts
  evaluation performance. When `enforce_eager=True`, we can reproduce the model
  performance reported in previous work. Without this setting, evaluation results may
  fall below reported performance. Therefore, we enforce `enforce_eager=True` during
  evaluation.

#### Runtime Expectations

Due to the sampling requirements and `enforce_eager` setting, the evaluation process
typically takes considerable time.

Runtime depends on several factors:

- Maximum generation length
- Number of questions in the dataset
- Model size

**Performance benchmarks** (on 8x H100 GPUs):

- **AIME dataset**: ~80 minutes
- **MATH_500 dataset**: ~160 minutes

## Option 2: Using AReaL-lite's Integration for RL Rollout

When using AReaL-lite, you can leverage the existing workflow from your training script
for evaluation purposes. This approach allows you to reuse the rollout components you've
already implemented, making the transition from training to evaluation seamless.

### Setting Up Evaluation

To set up evaluation using this method, create a new script that contains only the
rollout portion of your original training script. This focused script should include the
data collection and inference logic without the training components.

Once you have prepared your evaluation script, launch the evaluation experiment using
the following command:

```bash
python3 -m areal.launcher.${local|slurm|ray} my_script.py --config my-config.yaml allocation_mode=sglang.d4p1t2+eval
```

### Key Differences from Training

The primary difference between training and evaluation commands lies in the
`allocation_mode` parameter. In the evaluation setup, we remove the allocation of
training processes and replace them with an `eval` annotation. This modified allocation
mode launches **CPU processes** that run your customized workflow and collect data from
the LLM server, optimizing resource usage for evaluation tasks.

> **Note**: Optimal configurations often differ significantly between training and
> evaluation phases. As a user, you are responsible for ensuring that your evaluation
> experiment configuration and evaluation metrics are appropriate for your specific use
> case. Pay particular attention to settings such as `GenerationHyperparameters` and
> reward function to ensure they align with your evaluation objectives rather than your
> training setup.
