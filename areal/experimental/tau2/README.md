# TAU2 Agentic RL with AREAL-Asys

## Agentic Framework

docs: https://yuque.antfin.com/xrl-team/asystem-user-guide/ovrftp3f0kx2sh97

## Directory Structure

```bash
areal/experimental/tau2/
├── example_user.json   # chat history example of user for a task
├── example_agent.json  # chat history example of agent for a task
├── convert_dataset.py  # convert TAU2 data to huggingface dataset
├── tau2_env.py         # tau2 environment
└── agent.py            # agent, demonstrates how to interact with the TAU2 environment
```

## Quickstart

### Prepare Data

```bash
## 1. Download tau-bench data
export TAU2_BENCH_DIR=/path/to/tau2-bench
git clone https://github.com/sierra-research/tau2-bench.git $TAU2_BENCH_DIR
export TAU2_DATA_DIR=$TAU2_BENCH_DIR/data
export TAU2_OUTPUT_DIR=/path/to/output/dataset

## 2. Convert data
python areal/experimental/tau2/convert_dataset.py --data_dir $TAU2_DATA_DIR --output_dir $TAU2_OUTPUT_DIR --split train

## 3. start to train
export REAL_PACKAGE_PATH=/storage/openpsi/codes/puzhen.pz/AReaL
export PYTHONPATH=/usr/local/lib/python3.12/dist-packages:/storage/openpsi/codes/puzhen.pz/AReaL

python -u /storage/openpsi/codes/puzhen.pz/AReaL/areal/examples/grpo_proxy_trainer.py \
--config /storage/openpsi/codes/puzhen.pz/AReaL/areal/experimental/tau2/on_policy_agent_tau2.yaml \
scheduler.endpoint="http://asystem-scheduler.asystem-cluster-prod-1.svc:8081" \
experiment_name=tau2-agentic-rl \
trial_name=20251124-asystem-1 \
agent_module_path="areal.experimental.tau2.agent" \
train_dataset.path=/storage/openpsi/codes/puzhen.pz/aireline
```

## See Example Agent and User

| File               | Description                      |
| ------------------ | -------------------------------- |
| example_agent.json | Example agent (DeepSeek-R1-0528) |
| example_user.json  | Example user (Kimi-K2-Instruct)  |
