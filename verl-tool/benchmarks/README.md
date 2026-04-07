# Benchmark 
All benchmarks have been added as submodules. Make sure you have run the following command to clone all of them:

```bash
git submodule update --init --recursive
```

Specifically, to manually clone them, please run the following commands:

```bash
# Adapted math-evaluation-harness
git clone https://github.com/Zhuofeng-Li/math-evaluation-harness/tree/main

# Adapted BigCodeBench
git clone https://github.com/jdf-prog/bigcodebench.git

# Adapted evalplus
git clone https://github.com/jdf-prog/evalplus.git

# LiveCodeBench
git clone https://github.com/jdf-prog/LiveCodeBench
```

## Math Benchmarks
We provide a unified math benchmark that includes the following datasets: `GSM8K`, `MATH 500`, `Minerva Math`, `Olympiad Bench`, `AIME24`, and `AMC23`. Please see [math-evaluation-harness](https://github.com/Zhuofeng-Li/math-evaluation-harness/tree/9271e69bece4d14b33340df050c469996f1d6ab1) for more details.


## Coding Benchmarks
We provide three coding benchmarks: `BigCodeBench`, `evalplus`, and `LiveCodeBench`.  These benchmarks have been adapted to support the `verl-tool` tool-calling API for model evaluation.

**Before running evaluations, make sure to start the [eval_service](../eval_service) using the `verl_tool` environment** to launch an OpenAI-compatible server serving the tool-calling model and **provide the `vt_base_url`**, which is set in the script (default: `http://0.0.0.0:5000`).


```bash
bash eval_service/scripts/start_api_service.sh
```
Then you can run the eval script in different envs for each benchmark. For temperature and prompt settings, refer to the following instructions.

## BigCodeBench

### Environment Configuration
```bash
cd bigcodebench
git checkout verltool
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e .
uv pip install -r https://raw.githubusercontent.com/bigcode-project/bigcodebench/main/Requirements/requirements-eval.txt
uv pip install protobuf==3.20
```
### Evaluation
```bash
export BIGCODEBENCH_TIMEOUT_PER_TASK=30 # originally 240
split=complete # instruct or complete
subset=hard    # hard or full
bigcodebench.evaluate \
  --model "<model_name>" \
  --execution local \
  --split $split \
  --subset $subset \
  --backend openai \
  --bs 2048 \
  --base_url http://0.0.0.0:5000 \
  --temperature 0.0
```

- Note: you may want to modify system prompt in `bigcodebench/gen/util/openai_request.py`.

## evalplus (`humaneval` and `mbpp`)

### Environment Configuration
```bash
cd evalplus
git checkout verltool
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e .
uv pip install -r requirements.txt
```

### Evaluation
```bash
# humaneval split
export OPENAI_API_KEY="{KEY}" # https://platform.deepseek.com/api_keys
evalplus.evaluate --model "<model_name>"  \
                  --dataset humaneval     \
                  --base-url http://0.0.0.0:5000  \
                  --backend openai --greedy \
                  --temperature 0.0 

# mbpp split
export OPENAI_API_KEY="{KEY}" # https://platform.deepseek.com/api_keys
evalplus.evaluate --model "<model_name>"             \
                  --dataset mbpp           \
                  --base-url http://0.0.0.0:5000  \
                  --backend openai --greedy \
                  --temperature 0.0
```

- Note: you may want to modify system prompt in `evalplus/gen/util/openai_request.py`

## LiveCodeBench
### Environment Configuration
```bash
cd LiveCodeBench
git checkout verltool
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e .
```

### Evaluation
```bash
export OPENAI_API_KEY="{KEY}" # random key
export OPENAI_BASE_URL="http://0.0.0.0:5000" 
# set start or end time for custom evaluation
python -m lcb_runner.runner.main --model "<model_name>"  --scenario codegeneration --evaluate --start_date 2023-09-01 --end_date --multiprocess 64 --n 1  --temperature 0 --max_tokens 4096 --top_p 0.95 --num_process_evaluate 32

# test lcb_v4
python -m lcb_runner.runner.main --model "<model_name>"  --scenario codegeneration --evaluate  --release_version release_v4 --multiprocess 64 --n 1  --temperature 0 --max_tokens 4096 --top_p 0.95 --num_process_evaluate 32

# test lcb_v5
python -m lcb_runner.runner.main --model "<model_name>"  --scenario codegeneration --evaluate  --release_version release_v5 --multiprocess 64 --n 1  --temperature 0 --max_tokens 4096 --top_p 0.95 --num_process_evaluate 32
```

- Note: you may want to modify system prompt in `lcb_runner/runner/oai_runner.py`
