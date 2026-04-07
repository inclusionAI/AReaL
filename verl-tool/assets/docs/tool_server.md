
### Tool Server Design

Verl-Tool provides a unified tool server interface that allows you to easily add new tools and manage the tool calling process. The tool server is designed to handle multiple tools and can be extended to support new tools by simply adding a new Python file in the `verl_tool/servers/tools` directory.

The overall execution pipeline of a tool server is as follows:

1. **Request Reception**: The tool server receives a request from the model in the following format:
```json
{
    "trajectory_ids": ["traj_id_1", "traj_id_2", ...],
    "actions": ["action_1", "action_2", ..."],
    "finish": [false, true, ...], // whether a trajectory is finished and should not perform action
    "is_last_step": [false, false, ...], // whether this is the last step of the trajectory
    ... // other parameters
}
```

2. **Action Processing**: For each action in the request, the tool server tries to parse the action using all active tools (in the `identify_tool_for_action` method). If any tool's `parse_action` returns a valid sign, the action is sent to that tool's `get_observations` (or `conduct_action`) method to get the observation. If no tool matches, the observation will be an empty string, the valid sign will be False, and whether the trajectory is finished is determined by the `done_if_invalid` parameter when starting the tool server.

3. **Finish Handling**: The special `finish` field indicates whether the trajectory is determined to be finished on the verl side, meaning the model doesn't want to call any tool and the trajectory is finished. If this is set to True, the tool server directly sends the action to the special `finish` tool to clean up the corresponding trajectory's environment state.

4. **Response**: The tool server returns observations in the following format:
```json
{
    "observations": ["observation_1", "observation_2", ...],
    "dones": [false, true, ...], // whether the trajectory is finished
    "valids": [true, false, ...] // whether the action is valid (i.e., whether the action was parsed by any tool)
}
```

### Tool Server Usage

We provide a tool server starting command to start any tool server supported by verl-tool (see the full list in [verl_tool/servers/tools](verl_tool/servers/tools)). To start the tool server, use the following command:

```bash
# Start the tool server
host=localhost
port=5000
tool_type=python_code # separate by comma if you want to start multiple tool servers
workers_per_tool=512 # number of workers for the tool server, meaning how many threads will be used to handle a single tool request with multiple trajectories
python -m verl_tool.servers.serve --host $host --port $port --tool_type $tool_type --workers_per_tool $workers_per_tool --use_ray=True --max_concurrent_requests=8192 --router_workers=64 --log_level debug > tool_server.log 2>&1 &
```
for production use with multiple workers, please increase the `uvi_workers` parameter to utilize multiple CPU cores, e.g., `--uvi_workers=16`.

After running, you should see the following output. Tools marked with 🟢 are active, while those marked with ⚪ are inactive. The `finish` tool is always added to manage the end of each trajectory (e.g., delete environment state):

```
2025-06-05 14:28:24,029 - __main__ - INFO - Initializing tools: ('python_code',)
2025-06-05 14:28:24,037 - __main__ - INFO - Initialized tool: python_code
2025-06-05 14:28:24,037 - __main__ - INFO - Available Tools:
2025-06-05 14:28:24,037 - __main__ - INFO -   - base: inactive ⚪
2025-06-05 14:28:24,037 - __main__ - INFO -   - text_browser: inactive ⚪
2025-06-05 14:28:24,037 - __main__ - INFO -   - finish: active 🟢
2025-06-05 14:28:24,037 - __main__ - INFO -   - piston: inactive ⚪
2025-06-05 14:28:24,037 - __main__ - INFO -   - ipython_code: inactive ⚪
2025-06-05 14:28:24,037 - __main__ - INFO -   - python_code: active 🟢
2025-06-05 14:28:24,037 - __main__ - INFO -   - sandbox_fusion: inactive ⚪
2025-06-05 14:28:24,037 - __main__ - INFO -   - python_oj: inactive ⚪
2025-06-05 14:28:24,038 - __main__ - INFO - Starting async server on localhost:5500
2025-06-05 14:28:24,038 - __main__ - INFO - Server configured for up to 128 concurrent requests
INFO:     Started server process [2897325]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:5500 (Press CTRL+C to quit)
```

To test the tool server, we provide corresponding test scripts in the `verl_tool/servers/tests` directory. For example, to test the `python_code` tool server:

```bash
# Test the python_code tool server
python -m verl_tool.servers.tests.test_python_code_tool python --url=http://localhost:$port/get_observation
# python -m verl_tool.servers.tests.test_bash_terminal_tool bash --url=http://localhost:$port/get_observation
python verl_tool/servers/tests/test_ipython_efficiency.py --url=http://localhost:5000/get_observation --requests=200 --concurrency=512
python verl_tool/servers/tests/test_ipython_efficiency.py --url=http://localhost:5000/get_observation --requests=512 --concurrency=512 

for i in {1..8}; do
    python verl_tool/servers/tests/test_ipython_efficiency.py --url=http://localhost:5000/get_observation --requests=512 --concurrency=512 >> ipython_efficiency_$i.log 2>&1 &
done
```

- request
```json
payload = {
    "trajectory_ids": ["traj_1"],
    "actions": ["""```<python>\nprint('Hello from Python!')</python> ... <python>print('Hello again!')</python>``` ..."""],
    "extra_fields": [{}]
}
```
- response
```json
{
    "observations": [
        "\n<result>\nHello from Python!\nHello again!\n</result>\n"
    ],
    "dones": [
        false
    ],
    "valids": [
        true
    ],
    "processing_time_ms": 65.95945358276367
}
```


Notes:
    - If you set `use_ray=True`, please adjust the `workers_per_tool` parameter according to your system resources, as this parameter will be strictly enforced **globally** when using Ray actors. Try larger values if you find that your training is blocked waiting for tool server responses. This gives better control on how many concurrent requests can be handled by each tool.