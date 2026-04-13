### AgentActorConfig Parameters

| Parameter | Type | Default Value | Description |
|-----------|------|---------------|-------------|
| `enable_agent` | bool | `True` | Whether to enable the agent. If False, behaves the same as original verl |
| `max_turns` | int | `0` | Maximum number of interaction turns |
| `max_start_length` | int | `None` | Maximum token length of the initial prompt (before any turns) |
| `max_prompt_length` | int | `None` | Maximum total token length of the prompt (including turns) |
| `max_response_length` | int | `None` | Maximum token length of the response (e.g., LLM response + observation) |
| `max_obs_length` | int | `None` | Maximum token length of the observation from environment for each interaction turn |
| `max_action_length` | int | `None` | Maximum token length of the action (e.g., LLM response) for each interaction turn |
| `tool_server_url` | str | `None` | URL of the verl-tool server to call tools or APIs |
| `n` | int | `1` | Number of response samples |
| `truncate_obs_side` | str | `'left'` | Truncation direction for observations if they exceed length limit ('left' or 'right') |
| `truncate_response_side` | str | `'left'` | Truncation direction for responses if they exceed length limit ('left' or 'right') |
| `rolling_with_prompt` | bool | `False` | If True, keeps the system prompt when truncation occurs |
| `call_tool_first` | bool | `False` | Whether to call tool before generating the response |
| `min_turns` | int | `0` | Minimum number of actions required before allowing the agent to finish |
| `action_stop_tokens` | list | `None` | List of stop tokens that indicate the end of an action |
| `additional_eos_token_ids` | list | `None` | List of additional tokens treated as end-of-sequence |
| `mask_observations` | bool | `True` | Whether to mask observations in the attention mask and train on them |
| `force_finish_for_last_turn` | bool | `False` | Force the agent to end after the last turn without tool interaction |
| `enable_mtrl` | bool | `False` | Whether to enable multi-turn chat format, meaning the observation will be given in a new chat turn for reinforcement learning. If enabled, uses the same format as VLLM Chat Scheduler |
| `mtrl_role` | str | `user` | If `enable_mtrl` is enabled, this determines the role of the observation chat turn |
| `mtrl_sep` | str | `None` | In mtrl mode, this defines a special token that if present in the model's action, indicates it wants to interact with the tool server |
| `turn_end_token` | str | `"<\|im_end\|>"` | Token used to mark the end of each turn |
| `max_concurrent_trajectories` | int | `None` | Maximum number of concurrent trajectories for async rollout to avoid crash if too high concurrency. If None, no limit is applied. |



### Configuration Examples

#### 1. ToRL-style Training
Critical parameters configuration:
```bash
enable_agent=True
max_turns=1 # 1 time of code execution
min_turns=0 # no minimum turns required
action_stop_tokens='```output' # if the model outputs this token, we consider it wants to interact with the tool server
enable_mtrl=False # no multi-turn chat format
```

Trajectory format:
````
To solve this problem...
```python
...
```
```output
...
```
So the answer is ...
````

**When does a trajectory stop?** Since `min_turns=0`, the model will finish the trajectory either by generating an EOS token without any action stop tokens or by reaching `max_turns=1`.

#### 2. Multi-turn RL Training
Trajectory format:
````
To solve the problem ...
```python
...
```
...
<|im_end|>
<|im_start|>user
```output
...
```
<|im_end|>
<|im_start|>assistant
So the answer is ...
````

Critical parameters configuration:
```bash
enable_agent=True
max_turns=3 # 3 turns
min_turns=3 # force the model to call tools at least 3 times
action_stop_tokens='' # since always forcing, the action tokens are not needed
enable_mtrl=True # multi-turn chat format
mtrl_role=user # the observation will be given in a new chat turn
# add --done_if_invalid True to the verl-tool server command to ensure the trajectory stops when no tool matches
```

**When does a trajectory stop?** Since `min_turns=3` and `max_turns=3`, the model always tries to call tools by sending each turn's response. This response goes through each active tool to check for valid matches determined by the `parse_action` method. If any tool matches, the trajectory continues. If no tool matches, the trajectory stops and returns the final response.