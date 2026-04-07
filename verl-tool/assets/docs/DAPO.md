# DAPO support

we add DAPO support for training. 

- To enable dynamic batch, add following config in your training script:

```bash
    +algorithm.filter_groups.enable=True \
    +algorithm.filter_groups.metric='seq_final_reward' \
    +algorithm.filter_groups.max_num_gen_batches=0 \
```

- to mask the overlong trajectory (avoid training on it), add following config in your training script:

```bash
    actor_rollout_ref.agent.mask_overlong_loss=True \
```


- to clip higher reward, add following config in your training script:

```bash
    actor_rollout_ref.actor.clip_ratio_high=0.3 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
```

- to do token-level loss aggregation, add following config in your training script:

```bash
    actor_rollout_ref.actor.loss_agg_mode='token-mean' \
```

## Example Training

### Data Preprocessing

Prepare the data for training using the provided scripts. More examples can be found in [examples/data_preprocess](examples/data_preprocess).

```bash
python examples/data_preprocess/deepmath.py --data_source zwhe99/DeepMath-103K --local_dir data/deepmath_torl --sys_prompt_style torl
# use simple_rl style for non-tool system prompt
```

### Single Node Training

DAPO training example script for Qwen-1.5B on DeepMath dataset:

```bash
bash examples/train/torl/train_qwen_1.5B_math_deep_math_dapo.sh
```

We put example wandb training curve [here](https://wandb.ai/tiger_ai_lab/torl/reports/Example-Qwen-1-5b-Math-DAPO-training-on-DeepMath-data--VmlldzoxMzU4ODg4Mw)
