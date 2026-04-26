# AT-GIGPO Known Issues

## 1. epoch_count inflated by rollout repeat factor `n`

`ATGiGPOSampler.update()` receives the batch AFTER `repeat(n=4)`, so each
unique prompt is counted 4 times in `n_samples`.  This inflates:

- `epoch_count` metric by `n`x (reports ~3 epochs when actual is ~0.75)
- `epoch_decay` in `_recompute_probs()` triggers `n`x earlier than intended
- UCB exploration term `sqrt(2*log(N)/(n_k+1))` deflates faster

**Fix (not yet applied):** divide by `n` when computing epoch in
`_recompute_probs()` and `get_metrics()`, or deduplicate in `update()`.

## 2. Epoch decay coefficient may be too weak

Current formula:
```
decay = max(0.1, 1.0 - 0.3 * max(0.0, epoch - epoch_decay_start))
```

With `epoch_decay_start=2.0`:
| Real epoch | decay |
|------------|-------|
| ≤2         | 1.0   |
| 3          | 0.7   |
| 4          | 0.4   |
| 5+         | 0.1   |

Floor is 0.1, so even heavily oversampled tasks retain 10% weight.
The 0.3/epoch slope means 3 full epochs after start to reach floor.
Consider increasing slope or lowering floor if overfitting is observed.

## 3. Turn-bucket reweighting ineffective — REMOVE

`_at_gigpo_turn_bucket` in `RayPPOTrainer.fit()` reweights advantages by
turn-count buckets.  Empirical findings (step 1–133, mixed-at-gigpo-4node):

- **133 steps total, only 72 steps produced >1 bucket** (54%).  The remaining
  46% had all 256 samples in a single bucket → reweight scale = 1.0 (no-op).
- Root cause: most batches have narrow turn distributions, so automatic
  bucketing collapses to a single bin.
- When 2 buckets do appear, high-turn bucket has ~4.7× larger |adv| than
  low-turn, but reweight *flattens* this signal (equalizes bucket weights)
  instead of amplifying it.

**Decision:** remove turn-bucket reweighting.  The reweight logic lives in
`verl/verl/trainer/ppo/ray_trainer.py` lines 1302–1333 (the `at_gigpo`
block inside `fit()`).  Keep the per-bucket metrics logging for diagnostics
but skip the `advantages[mask] *= scale` step and `sort_by_turns`.

## 4. Sampling probabilities near-uniform despite UCB

At step 128 all 6 tasks have `sampling_prob` in [0.133, 0.197] (uniform =
0.167).  Root causes:

- **L_hat values too similar** (0.531–0.602): all tasks yield comparable
  episode-level learning signal at this training stage.
- **tau = 1.0 too high**: softmax temperature flattens score differences.
- **Epoch inflation (issue #1)**: decay triggers on all tasks, further
  compressing the distribution.

Raw episode |advantage| per task differs by only ~7% (0.36–0.39).  Using
total advantages (episode + step) would not materially change this because
the episode component dominates and score distributions are genuinely
similar across tasks.

UCB-based adaptive sampling will only differentiate once some tasks
saturate (accuracy → 1) and their |A| drops.  This is expected behavior
when tasks have similar difficulty for the current model.
