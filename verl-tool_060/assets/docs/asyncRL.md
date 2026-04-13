## Trajectory-level Asynchronous Reinforcement Learning

VerlTool now officially supports Trajectory-Level asynchronous rollout by setting `actor_rollout_ref.rollout.mode='async'`, which speeds up the rollout generation with tool calling by at least 2x! 
- `actor_rollout_ref.rollout.mode='sync'`: The default mode, where the rollout will only call the tool servers after the first turn generations of all the examples in the batch are completed.
- `actor_rollout_ref.rollout.mode='async'`: The new mode, where each trajectory is independently processed, allowing for tool calls to be made as soon as the first turn generation is available. This significantly reduces the waiting time for tool calls, especially in large batches.

### Comparison
For a simple math TIR RL training where there are 2048 trajectories in a batch, training 1.5B Qwen-Math on 4 H100 GPUs, the rollout time (excluding other RL operations) is as follows:

| Max Turns | Async (s) | Sync (s) | Speedup (times) |
|-----------|-----------|----------|-----------------|
|         1 |        **75** |      127 |    **1.69**         |
|         3 |        **94** |      194 |    **2.06**         |

Apparently, the async rollout is much faster than the sync one, and the speedup increases with the number of turns.

### Note

- We don't recommend to use ray based tool server for now as there are performance issues for ray to deal with multiple concurrent requests. Set `use_ray=False` (default value) when you are using asynchronous rollout.