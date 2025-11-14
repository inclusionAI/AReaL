# Diagnosing RL Performance

This guide helps you diagnose and resolve common performance issues in reinforcement
learning training. Use the strategies below to identify bottlenecks, tune
hyperparameters, and optimize your RL workflows.

## Using Synchronous RL Instead of Asynchronous Training

If you suspect asynchronous RL training impacts learning performance, or if you want to
debug a new agentic application, you can switch to standard synchronous RL training with
the following configuration:

```yaml
rollout:
  max_head_offpolicyness: 0  # 0 implies synchronous training
actor:
  recompute_logprob: false  # use logprobs returned by inference backend
  use_decoupled_loss: false  # reverts to the original PPO loss
```

For detailed information about these configurations, see our
[asynchronous RL guide](../algorithms/async.md) and
[CLI reference](../cli_reference.md).

## Training Rewards Not Increasing

This is a common issue that may be due to multiple reasons. We recommend the following
diagnostic steps:

1. **Establish a baseline:** Run evaluation on the test set to measure baseline
   performance before training. AReaL allows zero-code changes between training and
   evaluation, so you can reuse your training code for evaluation.
1. **Test on simpler data:** Run RL training on the test set instead of the training set
   to verify whether rewards increase.
1. **If rewards don't increase on the test set:** Tune your hyperparameters (e.g.,
   increase batch size or learning rate) or switch to a different base model. Consider
   applying SFT first, as this indicates the task may be too difficult for your current
   model.
1. **If rewards increase on test set but not training set:** Inspect the quality and
   difficulty of your training data. Ensure the distributions match and the difficulty
   is appropriate for your base model. You can enable dynamic filtering (similar to
   DAPO) by passing a `should_accept_fn` parameter to `prepare_batch` to ensure task
   difficulty remains appropriate during runtime. See our
   [detailed code walk-through](../lite/gsm8k_grpo.md) for more information.
