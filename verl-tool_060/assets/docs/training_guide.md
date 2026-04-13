## Training

### Data Preprocessing

Prepare the data for training using the provided scripts. More examples can be found in [examples/data_preprocess](/examples/data_preprocess/).

```bash
python examples/data_preprocess/deepmath.py --data_source zwhe99/DeepMath-103K --local_dir data/deepmath_torl --sys_prompt_style torl
# use simple_rl style for non-tool system prompt
```

### Single Node Training

We provide an integrated training script for each example. To train ToRL, run the following command:

```bash
bash examples/train/math_tir/train_1.5b_grpo.sh
```

See more examples in [examples/train](/examples/train), where each folder is a separate project with its own training script. You can also submit a PR to add your own training script to the project.

**Training Tips:**

1. **For low VRAM GPUs**: Set `do_offload=True`, `enforce_eager=True`, `tensor_parallel_size=1`, `use_dynamic_bsz=False`, and use a small `ppo_micro_batch_size_per_gpu`.
2. **For high VRAM GPUs**: Set `do_offload=False` and `use_dynamic_bsz=True` to speed up training.
3. **If VLLM generation gets stuck**: Try lowering `workers_per_tool` and reducing `gpu_memory_utilization` in the script.
4. **If you encounter CPU OOM issues during VLLM rollout generation**: Try setting `do_offload=False` and lowering `gpu_memory_utilization`.
5. See [verl performance tuning](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html) for more details.
6. **If your model is too large and cause GPU OOM on the rank-0 device during loading**: Try change `strategy="fsdp2"` to `strategy="fsdp"` in the training script.

### Multi Node Training (Experimental)

1. On slurm
```bash
sbatch --account ${your_account} --nodes 2 examples/train/math_tir/train_7b_grpo_multi_node_slurm.sh ${your_container_path} ${your_mount_path} ${your_workdir_inside_container}
```

Other cluster's examples are to be added.

### Training Logs

During training, the generated responses, rewards, etc., of each step are recorded in the `verl_step_records` directory. The corresponding code logic is written in the `verl_tool/worker/reward_manager/{reward_manager_name}.py` file. This helps you debug the training process and understand how the model interacts with the tool server.

If it's multi-node training, the step records may be saved in the checkpoint directory instead, e.g., `{checkpoint_dir}/step_records`.
