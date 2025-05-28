# Evaluation

The evaluation code is located in the `evaluation` folder of the repository. Following the previous tutorial, trained checkpoints will be saved under `/storage/ray/experiments/checkpoints/root/`.

## Setup Evaluation Environment

Start a new container to execute the evaluation script. **Note**: Evaluation requires updates to certain Python libraries, so avoid using the training container for this task.

```bash
docker run -d --name areal-eval --privileged --gpus all --network host --shm-size 700g -v /storage:/storage ghcr.io/inclusionai/areal-runtime:v0.3.0 /bin/bash -c "tail -f /dev/null"
docker exec -it areal-eval bash
```

## Install Dependencies and Run Evaluation

Execute the following commands inside the Docker container:

```bash
cd /storage/codes/AReaL/evaluation
cd latex2sympy
pip install -e .
cd ..
pip install -r requirements.txt 
pip install vllm==0.8.5 --no-build-isolation
pip install transformers==4.51.1
pip install prettytable timeout_decorator
mkdir -p /storage/ray/eval_output/
nohup python eval_and_aggregate.py \
    --model_path /storage/ray/experiments/checkpoints/root/my-exp/my-trial/epoch1epochstep20globalstep20/ \
    --output_path /storage/ray/eval_output/ \
    --data_names "math_500,aime24,amc23" \
    --max_gen_tokens 32768 &> /storage/ray/eval_output/eval_and_aggregate_parallel.log &
```

### Command Line Parameters

- **`--model_path`**: Path to the saved model parameters
- **`--output_path`**: Path to store generated answers and log files during evaluation
- **`--data_names`**: Dataset(s) to evaluate. Multiple datasets can be separated by commas. Available options: `math_500`, `math`, `gsm8k`, `train_amc_aime`, `aime24`, `amc23`
- **`--max_gen_tokens`**: Maximum length of generated answers (default: 32768)

## Evaluation Results

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

### Metrics Explanation

- **`{greedy|sample}_length`**: Average answer length under greedy or random sampling strategy
- **`greedy_acc`**: Average accuracy under greedy sampling
- **`sample_pass@{k}`**: Probability of generating a correct answer within `k` attempts under random sampling

## Configuration Details

### Sampling Parameters

- The evaluation script defaults to averaging 32 samples with temperature 0.6
- We observed that the `enforce_eager` parameter in vLLM significantly impacts evaluation performance
- When `enforce_eager=True`, we can reproduce the model performance reported in previous work
- Without this setting, evaluation results may fall below reported performance
- Therefore, we enforce `enforce_eager=True` during evaluation

### Runtime Expectations

Due to the sampling requirements and `enforce_eager` setting, the evaluation process typically takes considerable time.

Runtime depends on several factors:
- Maximum generation length
- Number of questions in the dataset  
- Model size

**Performance benchmarks** (on 8x H100 GPUs):
- **AIME dataset**: ~80 minutes
- **MATH_500 dataset**: ~160 minutes