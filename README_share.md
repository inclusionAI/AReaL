#  This is the code for sharing SFT and RL on Qwen3 4B

## SFT
Install the normal llamafactory env and activate it

Change the output and input directory
```
cd LLaMA-Factory-direct-conversation

llamafactory-cli train train_config.yaml
```

## RL

Install the AReal env

My env is 
```
cd AReal
python3 -m areal.launcher.slurm examples/math/parallel_grpo.py     --config examples/math/openr1_rl.yaml     experiment_name=    trial_name=     allocation_mode=sglang:d24p1t1+d2p1t8     cluster.n_nodes=5     cluster.n_gpus_per_node=8 
```

## Eval

```
cd Test_latest
./test8times_shell_input.sh -m <your model path>
```
It will run 8 times on 8 GPUs
