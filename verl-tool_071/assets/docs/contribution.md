## Contributing

### Adding New Tools

Go to the [./verl_tool/servers/tools](./verl_tool/servers/tools) directory. Each tool has a name (e.g., `base`, `python_code`). The tool name is exactly the name of the Python file you should create in the directory. See [./verl_tool/servers/tools/python_code.py](./verl_tool/servers/tools/python_code.py) for an example.

### Creating New Reward Managers

Go to the [`./verl_tool/worker/reward_manager`](./verl_tool/worker/reward_manager) directory and add your new reward manager. Then, make sure to update the `verl_tool/trainer/main_ppo.py` file to include your new reward manager.
