# Training a Multi-Turn GSM8K Math Agent in AReaL

Files in this folder presents an example that train a multi-turn GSM8K math agent from
Qwen/Qwen2-1.5B-Instruct, using `ArealOpenAI` APIs and its `individual` mode to organize
training data and discount reward. Note that `sglang.disable_radix_cache` is set to true
to stablize training.

# To run the example

```bash
python3 -m areal.launcher.ray examples/math/multi-turn/train.py \
    --config examples/math/multi-turn/config.yaml \
    experiment_name=gsm8k-math-multiturn trial_name=trial0
```

## Reward Curve

<img align="center" alt="reward curve" src="reward_curve.png" width="100%">
