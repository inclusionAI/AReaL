## Updating verl version
To update the verl submodule to the latest version, run the following commands in the verl-tool root directory:
```bash
cd verl && git pull origin main && cd ..
cp -r verl/verl/trainer/config/* ./verl_tool/trainer/config/
uv pip install -e verl
```
Then copy following to the proper place in `verl_tool/trainer/config/ppo_trainer.yaml`:
```yaml
defaults:
  # VerlTool Agent
  - verltool@actor_rollout_ref.agent: agent.yaml
```
Note there might be some small parameters needed to remove the '+' prefix in the training script because default values may be added to the new config files.


## Updating to verl 0.6.0
```bash
cd verl && git pull origin main && git checkout -b v0.6.0 && cd ..
cp -r verl/verl/trainer/config/* ./verl_tool/trainer/config/
uv pip install -e verl
```
And search in config to replace `default_agent_loop: single_turn_agent` to `default_agent_loop: verltool_agent` to save additional config for verltool agent loop.