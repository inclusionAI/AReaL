

## Evaluation

Evaluating verl-tool trained models is naturally challenging due to the tool calling nature, where we need to maintain not only the model's inference engine (e.g., VLLM or SGLang) but also the tool server that allows the models to interact. Therefore, to better facilitate the evaluation service, we wrap the entire interaction process into an OpenAI-like API service, where you can simply send messages **in the OpenAI chat format** and the rest of the multi-turn interaction between the inference engine and the tool server will be handled internally by the service, returning the final result in the OpenAI response format.

Specifically, the service can be accessed via OpenAI's `client.chat.completions.create` or `client.completions.create` API. When accessing, please ensure the model name corresponds to the one being set in the script. The server is managed by [`app.py`](eval_service/app.py), while the main tool-calling logic is implemented in [`model_service.py`](eval_service/model_service.py). [`config.py`](eval_service/config.py)'s default parameters are overridden in [`scripts/start_api_service.sh`](eval_service/scripts/start_api_service.sh).

### Evaluation Service Setup

We use our published checkpoint 🤗[VerlTool/torl-deep_math-fsdp_agent-qwen2.5-math-1.5b-grpo-n16-b128-t1.0-lr1e-6-320-step](VerlTool/torl-deep_math-fsdp_agent-qwen2.5-math-1.5b-grpo-n16-b128-t1.0-lr1e-6-320-step) as an evaluation example.

Content of `eval_service/scripts/start_api_service.sh`:
```bash
#!/bin/bash
set -x
# 1. Start Tool Server
host=0.0.0.0
port=$(shuf -i 30000-31000 -n 1)
tool_server_url=http://$host:$port/get_observation
python -m verl_tool.servers.serve --host $host --port $port --tool_type "python_code" --workers_per_tool 32 --done_if_invalid True --silent True &
server_pid=$!
echo "Server (pid=$server_pid) started at $tool_server_url"

# 2. Start API service
model_path=VerlTool/torl-deep_math-fsdp_agent-qwen2.5-math-1.5b-grpo-n16-b128-t1.0-lr1e-6-320-step
max_turns=4 # defines the maximum number of interaction turns between the model and the tool server, if set to 0, it will not limit the number of turns
min_turns=0 # defines the minimum number of action turns between the model and the tool server, will force the model to call tools at least this many times, if set to 0, it will not force the model to call tools
api_host="0.0.0.0"
api_port=5000
action_stop_tokens='```output' # stop at this token, then send the output to the tool server, this is a special token that we use to indicate the end of the action, you can change it to any other token that your model will produce when it is asking for a tool calling round
tensor_parallel_size=1
num_models=1 # number of vllm instances; num_models * tensor_parallel_size should be equal to the number of GPUs
enable_mtrl=False # whether to evaluate in multi-chat-turn setting (taking each observation as a new chat turn)
# temp file for action tokens as verl cannot pass special strings as params
action_stop_tokens_file=$(mktemp)
echo "$action_stop_tokens" > $action_stop_tokens_file
echo "action_stop_tokens_file=$action_stop_tokens_file"

python eval_service/app.py \
    --host $api_host \
    --port $api_port \
    --tool-server-url $tool_server_url \
    --model $model_path \
    --max_turns $max_turns \
    --min_turns $min_turns \
    --action_stop_tokens $action_stop_tokens_file \
    --tensor-parallel-size $tensor_parallel_size \
    --num-models $num_models \
    --enable_mtrl $enable_mtrl 

api_server_pid=$!
echo "API started at $api_host:$api_port"

# 3. Kill all servers
pkill -9 -P $server_pid
kill -9 $server_pid
pkill -9 -P $api_server_pid
kill -9 $api_server_pid
```

**Steps:**

1. **Start the Python code execution tool server and API Service**: [eval_service/scripts/start_api_service.sh](eval_service/scripts/start_api_service.sh)
```bash
bash eval_service/scripts/start_api_service.sh &
```

This starts both the tool server and the API service, where the tool server handles tool calling requests from the model, and the API service handles OpenAI-like API requests. When you see the following output, the API service is running successfully:

```
INFO:     Started server process [671818]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
vLLM instance model-0 status: SyncPage[Model](data=[Model(id='VerlTool/torl-deep_math-fsdp_agent-qwen2.5-math-1.5b-grpo-n16-b128-t1.0-lr1e-6-320-step', created=1749194165, object='model', owned_by='vllm', root='VerlTool/torl-deep_math-fsdp_agent-qwen2.5-math-1.5b-grpo-n16-b128-t1.0-lr1e-6-320-step', parent=None, max_model_len=4096, permission=[{'id': 'modelperm-459bedc083f4492b8d908b521c2447c7', 'object': 'model_permission', 'created': 1749194165, 'allow_create_engine': False, 'allow_sampling': True, 'allow_logprobs': True, 'allow_search_indices': False, 'allow_view': True, 'allow_fine_tuning': False, 'organization': '*', 'group': None, 'is_blocking': False}])], object='list')
✅ vLLM service started successfully with model: VerlTool/torl-deep_math-fsdp_agent-qwen2.5-math-1.5b-grpo-n16-b128-t1.0-lr1e-6-320-step
```

### Testing the API Service

We provide test scripts in [eval_service/test/test_api.py](eval_service/test/test_api.py) to test the API service. You can run the following command:

```bash
model_name=VerlTool/torl-deep_math-fsdp_agent-qwen2.5-math-1.5b-grpo-n16-b128-t1.0-lr1e-6-320-step
test_task=math # or code
test_type=chat_completion # or completion
base_url=http://localhost:5000 # replace with your local server address
python eval_service/test/test_api.py --model_name $model_name --test_task $test_task --test_type $test_type --base_url $base_url
```

Example output:
````
Testing math task...
Testing math with chat_completion on model VerlTool/torl-deep_math-fsdp_agent-qwen2.5-math-1.5b-grpo-n16-b128-t1.0-lr1e-6-320-step at http://localhost:5000
To convert the point $(0,3)$ from rectangular coordinates to polar coordinates, we need to find the radius $r$ and the angle $\theta$.

The radius $r$ is the distance from the origin to the point and can be calculated using the formula:
\[ r = \sqrt{x^2 + y^2} \]
where $x$ and $y$ are the rectangular coordinates. For the point $(0,3)$, we have $x = 0$ and $y = 3$.

The angle $\theta$ is the angle formed with the positive x-axis and can be calculated using the formula:
\[ \theta = \arctan\left(\frac{y}{x}\right) \]
However, since $x = 0$, we need to consider the special case where the point lies on the y-axis. In this case, $\theta = \frac{\pi}{2}$ if $y > 0$ and $\theta = \frac{3\pi}{2}$ if $y < 0$. Since $y = 3 > 0$, we have $\theta = \frac{\pi}{2}$.

Let's calculate this using Python to ensure accuracy.
```python
import math

# Rectangular coordinates
x = 0
y = 3

# Calculate the radius r
r = math.sqrt(x**2 + y**2)

# Calculate the angle theta
# Since x = 0, we need to handle the special case
if x == 0:
    if y > 0:
        theta = math.pi / 2
    elif y < 0:
        theta = 3 * math.pi / 2
    else:
        theta = 0  # or 2*pi, but we'll use 0 for simplicity
else:
    theta = math.atan2(y, x)

print(((r, theta)))
```
```output
(3.0, 1.5707963267948966)
```
The polar coordinates for the point $(0,3)$ are $(3.0, \frac{\pi}{2})$. Therefore, the final answer is:

\[
\boxed{(3, \frac{\pi}{2})}
\]
````