# MathTIR Training Guide

## Dataset Preparation

```bash
# simple rl system prompt (no tool)
python examples/data_preprocess/deepmath.py --data_source zwhe99/DeepMath-103K --local_dir data/deepmath_simple_rl --sys_prompt_style simple_rl
# torl system prompt (with code interpreter tool)
python examples/data_preprocess/deepmath.py --data_source zwhe99/DeepMath-103K --local_dir data/deepmath_torl --sys_prompt_style torl
```

## Training example
- grpo
```bash
bash examples/train/math_tir/train_1.5b_grpo.sh
```
- dapo
```bash
bash examples/train/math_tir/train_1.5b_dapo.sh
```

## Tips
Tips:
- if output shared memory, try lower the `data.dataloader_num_workers`
- if out of cuda memory during vllm rollout, try set `actor_rollout_ref.rollout.enforce_eager=True`, might be slower.
- if out of cuda memory during training, try lower the `use_dynamic_bs=False`.
