# Tau-Agent

Adapt the verl-agent framework to work with the [tau2-bench](https://github.com/sierra-research/tau2-bench-private) environment.

Note: The gym feature is currently only available in the private version of tau2-bench, on `feature/gym` branch.

## Setup

Running the tau-agent requires a GPU machine with at least 4 A100 GPUs (80GB memory).
To obtain a GPU machine on Lambda Lab, please refer to the [internal document](https://docs.google.com/document/d/17FlYp3Yoc6GvdfQDmSiRWw0uevDvjfH3LTAiKKC1kZc/edit?usp=sharing) to setup the machine.

Then we assume you have anaconda installed on the machine, and the CUDA version is 12.8.

```bash
conda create python==3.12 -n tau-agent -y
conda activate tau-agent
```

Then install the dependencies.
```bash
# Install vllm
pip install vllm==0.8.5 --extra-index-url https://download.pytorch.org/whl/cu128

# Install flash-attention
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
```

Then install tau2-bench and tau-agent repositories.
```bash
# Install tau2-bench
# First clone the private version of tau2-bench, switch to the correct branch.
cd /path/to/tau2-bench-private
pip install -e .

# Install tau-agent
# First clone the tau-agent repository.
cd /path/to/tau-agent
pip install -e .
```

Then you need to manually fix some dependecy (also needed if you reinstall tau2-bench or tau-agent):
```bash
pip install opentelemetry-exporter-prometheus==0.47b0
pip install opentelemetry-exporter-otlp==1.26.0
```

## Run the tau-agent

The following command will run the tau-agent on the telecom small dataset (The ones containing only one atomic sub-task).

```bash
bash examples/grpo_trainer/run_tau2_telecom_small.sh
```
