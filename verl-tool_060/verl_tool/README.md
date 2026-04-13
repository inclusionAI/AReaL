# Main codebase for verl-tool

This is the core of verl-tool. 

`agent_workers` implement reward managers responsible for rule-based reward computation for each rollout. `fsdp_workers` implement the actual roll-out logic during model training.

`llm_agent` handles the interception of LLM's intermediate output and perform dynamic tool-calling during model training. 

`servers` involves the implementation of all possible tools that the LLM can call during training and evaluation.

`trainer` derives from `verl`'s ppo trainer. The `main_dapo.py` corresponds to the migrated DAPO training implementation.