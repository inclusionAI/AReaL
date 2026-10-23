# Hermes 数学任务八卡推理评测

在现有 8×H20 上，用 Qwen3-4B-Instruct-2507 驱动 Hermes 解答固定的 200 道 DAPO
数学题并评分。仅运行推理与工具服务，不启动训练或保存优化器状态。

起始配置：SGLang **TP1×DP8**，每卡一个完整模型副本；Hermes **1 个 pair**；客户端 **32 个并发 episode**。4B
与1.7B在原8轮预算下的首轮200题分别正确150题与48题。现恢复 Hermes 原生默认轮数（0.19为90），新结果使用独立目录；切换到 Qwen3-1.7B
见第9节。

## 1 在 Mac 同步代码和固定题目

以下命令在 Mac 执行。同步工作区代码与已冻结的题目文件；公司端也可按[2×2操作文档](RUN_2X2_8GPU.md)拉取分支，题单仍需另行复制。

```bash
cd /Users/tournesol/Desktop/HKUST/Project/Ant/AutoPilot-v2

ssh -o ProxyCommand="ssh -p 22001 wens@203.160.89.178 -W %h:%p" \
  wens@192.168.1.148 'mkdir -p /mnt/data/wens/hermes-eval'

rsync -az --exclude=.git --exclude=.venv --exclude=__pycache__ \
  --exclude=.pytest_cache \
  -e 'ssh -o ProxyCommand="ssh -p 22001 wens@203.160.89.178 -W %h:%p"' \
  AReaL artifacts/hermes-math-implementation-20261022/tasks \
  wens@192.168.1.148:/mnt/data/wens/hermes-eval/
```

服务器上的目录分别是：

| 路径                                 | 内容                                                  |
| ------------------------------------ | ----------------------------------------------------- |
| `/mnt/data/wens/hermes-eval/AReaL`   | 项目代码与本操作文档                                  |
| `/mnt/data/wens/hermes-eval/tasks`   | `eval.jsonl` 200 题，以及 pilot、train 清单和版本摘要 |
| `/mnt/data/wens/hermes-eval/results` | 后续生成的评测结果、工具记录和服务日志                |

## 2 在服务器准备容器

先登录服务器：

```bash
ssh -o ProxyCommand="ssh -p 22001 wens@203.160.89.178 -W %h:%p" \
  wens@192.168.1.148
```

看到 `wens@1stgk54` 提示符后，再单独粘贴下一段；后续宿主机命令均在这个服务器 shell 中执行。

```bash
export AREAL_EVAL_ROOT=/mnt/data/wens/hermes-eval
export AREAL_MODEL_SOURCE=/mnt/data/wens/areal-autopilot-elastic/elastic/native-acceptance-20261022-1792600135/assets/Qwen3-4B-Instruct-2507

df -h /mnt/data
nvidia-smi
test -r "$AREAL_MODEL_SOURCE/config.json"
docker image inspect ghcr.io/areal-project/areal-runtime:v2.1.0-sglang python:3.12 --format '{{.RepoTags}} {{.Id}}'
mkdir -p "$AREAL_EVAL_ROOT/results"
```

确认八卡可用、清理后的空间足够，且模型与两个本地镜像存在。下面使用新容器名 `wens-hermes-eval`；若同名容器已经存在，先检查并处理命名冲突。只暴露 GPU 0
的旧容器需要按新参数重新创建。

```bash
docker run -dit --name wens-hermes-eval --gpus all \
  --network host --shm-size 16g \
  -v "$AREAL_EVAL_ROOT:/workspace" \
  -v "$AREAL_MODEL_SOURCE:/model:ro" \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$(command -v docker):/usr/local/bin/docker:ro" \
  -w /workspace/AReaL \
  -e PYTHONPATH=/workspace/AReaL \
  -e AREAL_MODEL_PATH=/model \
  -e AREAL_INFERENCE_URL=http://127.0.0.1:30000 \
  -e AREAL_INFERENCE_ADMIN_KEY="$(openssl rand -hex 24)" \
  -e AREAL_AGENT_ADMIN_KEY="$(openssl rand -hex 24)" \
  -e AREAL_TASK_PYTHON_IMAGE=python:3.12 \
  -e AREAL_TASK_TOOL_ROOT=/workspace/results/tools \
  -e AREAL_TASK_MAX_TOKENS=4096 \
  -e AREAL_TASK_TEMPERATURE=1.0 \
  -e AREAL_TASK_TOP_P=1.0 \
  ghcr.io/areal-project/areal-runtime:v2.1.0-sglang sleep infinity

docker exec -it wens-hermes-eval bash
```

此时进入容器。API 密钥已作为容器环境变量生成，后续 `docker exec` 自动继承。挂载宿主机 Docker CLI 和 socket，供每题的 Python
工具创建隔离容器。

## 3 在容器安装工作流并预检

```bash
uv pip install --python /opt/.venv/bin/python --no-deps -e .
uv pip install --python /opt/.venv/bin/python 'hermes-agent==0.19.0' 'math-verify==0.8.0'
uv pip freeze --python /opt/.venv/bin/python > /workspace/results/packages.txt

nvidia-smi -L
docker version
sha256sum /workspace/tasks/eval.jsonl
python -m examples.task_workflows.environments.python
```

应看到八张 GPU，Docker client/server 均可连接。`eval.jsonl` 的 SHA256 应为：

```text
e899346883efa7bd1f4cf3abb8319684bbbcd2deb2e397b3e3f8ca54640f1c49
```

工具预检应返回 `status: completed`，输出为 `42`。预检通过后退出到宿主机：

```bash
exit
```

## 4 在宿主机启动八卡 SGLang

下面在后台启动服务，日志写入挂载目录。`TP1×DP8` 在同一个端口后面启动八个单卡副本，并分发模型请求；这是 SGLang 的原生 DP
模式，见[固定版本启动源码](https://github.com/sgl-project/sglang/blob/v0.5.10.post1/python/sglang/srt/entrypoints/engine.py)。

```bash
docker exec -d wens-hermes-eval bash -c '
  exec python -u -m sglang.launch_server \
    --model-path "$AREAL_MODEL_PATH" \
    --served-model-name hermes-math \
    --host 127.0.0.1 --port 30000 \
    --tp-size 1 --dp-size 8 \
    --load-balance-method round_robin \
    --dtype bfloat16 --context-length 65536 \
    --mem-fraction-static 0.5 \
    --tool-call-parser qwen \
    --api-key "$AREAL_INFERENCE_ADMIN_KEY" \
    > /workspace/results/sglang.log 2>&1
'

docker exec wens-hermes-eval tail -n 50 -f /workspace/results/sglang.log
```

`docker exec -d` 成功提交后台命令时不会打印启动进度；首次导入时日志也可能暂时为空。上面的 `tail -f` 持续跟踪日志，服务就绪后按 Ctrl+C
退出日志查看，SGLang 继续运行，再验证模型接口。

若日志持续为空，在另一宿主机终端查看进程；存在 Python 进程仅说明还在运行，接口检查通过才表示可用：

```bash
docker top wens-hermes-eval -eo pid,ppid,stat,etime,comm
```

若仍在加载，稍后重试接口检查。

```bash
docker exec wens-hermes-eval bash -c '
  curl -fsS -H "Authorization: Bearer $AREAL_INFERENCE_ADMIN_KEY" \
    "$AREAL_INFERENCE_URL/v1/models"
'
```

应返回模型名 `hermes-math`。八卡副本共用磁盘上的同一份只读模型目录，每卡各自占用模型显存和 KV cache。

## 5 在容器启动 Hermes Agent Service

先在宿主机执行：

```bash
docker exec -it wens-hermes-eval bash
```

看到 `root@1stgk54:/workspace/AReaL#` 容器提示符后，再单独粘贴下面这段。不要将进入容器与进入后执行的命令一次性粘贴。

```bash
env -u HERMES_MAX_TURNS CUDA_VISIBLE_DEVICES="" areal agent run --service hermes-math-eval \
  --agent examples.task_workflows.service.TaskAgent \
  --num-pairs 1 --admin-api-key "$AREAL_AGENT_ADMIN_KEY"

areal agent status --service hermes-math-eval
export AREAL_AGENT_URL="$(areal agent status --service hermes-math-eval --json \
  | python -c 'import json, sys; print(json.load(sys.stdin)["gateway_url"])')"
```

`env -u` 清除旧容器可能保留的8轮覆盖，使用 Hermes 原生默认。Agent Service 在后台运行。它的 gateway 端口由 CLI
分配，上面从状态中读取，无须手填。下面两轮评测继续在这个容器 shell 中执行；若重新进入容器，重新执行 `export AREAL_AGENT_URL=...`。

## 6 先跑四题检查闭环

新版客户端每写入一条任务结果就刷新进度条，显示已处理数、用时和预计剩余时间；全部结束后打印 JSON 汇总。进度条计数包含成功、失败及未提交记录，成功数以汇总为准。

若服务器仍是旧代码，先在 Mac 同步整个共享工作流包，再按第5节重启 Agent，之后使用新客户端：

```bash
rsync -az --exclude=__pycache__ \
  -e 'ssh -o ProxyCommand="ssh -p 22001 wens@203.160.89.178 -W %h:%p"' \
  /Users/tournesol/Desktop/HKUST/Project/Ant/AutoPilot-v2/AReaL/examples/task_workflows/ \
  wens@192.168.1.148:/mnt/data/wens/hermes-eval/AReaL/examples/task_workflows/
```

下面在容器中执行；`cat` 要等 Python 命令结束才会执行。

```bash
python -m examples.task_workflows.client --benchmark dapo_math \
  --tasks /workspace/tasks/eval.jsonl \
  --output /workspace/results/smoke-4 \
  --agent-url "$AREAL_AGENT_URL" \
  --inference-url "$AREAL_INFERENCE_URL" \
  --model hermes-math --mode eval --limit 4 --concurrency 32

cat /workspace/results/smoke-4/summary.json
```

正常完成应为 `completed: 4`、`infrastructure_errors: 0`、`not_submitted: 0`。答案错误得 0
分，属于有效评测结果。工具由 Hermes 按需调用，不要求每题都使用工具。四题不足以检验八卡吞吐。

## 7 跑完整的固定二百题评测

```bash
python -m examples.task_workflows.client --benchmark dapo_math \
  --tasks /workspace/tasks/eval.jsonl \
  --output /workspace/results/eval-200-run1 \
  --agent-url "$AREAL_AGENT_URL" \
  --inference-url "$AREAL_INFERENCE_URL" \
  --model hermes-math --mode eval --limit 200 --concurrency 32

cat /workspace/results/eval-200-run1/summary.json
exit
```

保持客户端所在的 SSH 会话直到命令结束，最后的 `exit` 回到宿主机。再次运行时将输出目录换成 `eval-200-run2`
等新名字；客户端拒绝覆盖已有目录。复用相同 `eval.jsonl`，不要重新抽题。

`--concurrency 32` 是八卡合计的在途 episode 上限，包含模型生成、工具执行、评分和清理。`--num-pairs 1` 可并发运行多个
episode；当前单 Worker 的默认线程池在 CPU 充足时最多 32 个线程。是否能充分利用八卡，需要结合 GPU 利用率与队列实测后调整。

题目及清单顺序固定，`run.json` 保存所选题目 ID；并发请求的实际到达时序、完成顺序与各题内部模型调用的交错不保证一致。`episodes.jsonl`
按完成顺序写入，比较时按 `task_id` 对齐。

## 8 查看结果和停止服务

结果在宿主机 `/mnt/data/wens/hermes-eval/results/eval-200-run1/`，对应容器
`/workspace/results/eval-200-run1/`：

| 文件                  | 用途                                             |
| --------------------- | ------------------------------------------------ |
| `summary.json`        | 完成数、正确数、基础设施错误数、模型与工具调用数 |
| `episodes.jsonl`      | 每题答案、评分、时间、错误与工具记录             |
| `run.json`            | 模型名、并发数及有序题目 ID                      |
| `tasks-manifest.json` | 数据版本与题目文件摘要                           |

`episodes.jsonl` 的 `timing` 保存每题总耗时 `total_seconds`、累计推理调用耗时
`inference_seconds`、累计工具调用耗时
`tool_seconds`（秒）。总耗时从获得客户端并发名额算到评分与清理结束；推理包含服务排队、网络和完整流式回复，工具包含启动与等待。每次调用的
`started`／`finished` 时间戳和 `seconds` 耗时分别在 `metadata.inference_calls`、
`metadata.environment.timing_calls`；并行调用的累计耗时可能重叠。旧结果不会补出缺失的推理计时。

正常完成 200 题后，准确率为 `correct / 200`。若有 `infrastructure_errors` 或 `not_submitted`，先检查对应
episode 和服务日志，再判断是否重跑。

评测期间可在另一个宿主机终端运行：

```bash
watch -n 2 'wc -l /mnt/data/wens/hermes-eval/results/eval-200-run1/episodes.jsonl'
```

每两秒刷新已写入的记录数；第一题结束前文件尚未创建。行数包括失败及未提交记录，最终成功数以 `summary.json` 为准。Ctrl+C 退出查看，不影响评测。

服务状态和日志：

```bash
nvidia-smi
docker exec wens-hermes-eval areal agent status --service hermes-math-eval
docker exec wens-hermes-eval areal agent logs --service hermes-math-eval --component worker-0 --lines 80
docker exec wens-hermes-eval tail -n 50 /workspace/results/sglang.log
```

评测结束后，在宿主机停止本次服务；同时把结果目录归还当前用户，便于后续整理和删除：

```bash
docker exec wens-hermes-eval areal agent stop --service hermes-math-eval
docker exec wens-hermes-eval chown -R "$(id -u):$(id -g)" /workspace/results
docker stop -t 30 wens-hermes-eval
```

结果保留在宿主机挂载目录中。

## 9 切换到 Qwen3 1.7B

复用现有容器及已安装依赖，继续使用原来的 `eval.jsonl`、TP1×DP8、1 pair、32并发及采样／工具预算。新模型使用独立服务模型名与结果目录，保留4B结果。

Qwen3-1.7B 默认开启 thinking，本次关闭，与4B的非 thinking 模式保持一致。Hermes 0.19 要求至少64000上下文；32768会在
Agent 初始化时失败。本次按
[Qwen 官方配置](https://qwen.readthedocs.io/en/latest/deployment/sglang.html#context-length)
使用2倍 YaRN 扩展到65536。YaRN 也可能影响短文本表现，结果应注明「1.7B／非 thinking／64K YaRN」。

### 在宿主机准备模型并重启容器

之前的空间检查中已有下列1.7B目录；先确认它仍在。将其复制到已挂载的工作目录，容器便可访问，额外空间约一份模型权重，约4GB。

```bash
export AREAL_EVAL_ROOT=/mnt/data/wens/hermes-eval
export AREAL_MODEL_17_SOURCE=/mnt/data/wens/autopilot-h20/models/Qwen3-1.7B

test -r "$AREAL_MODEL_17_SOURCE/config.json"
test -r "$AREAL_MODEL_17_SOURCE/tokenizer_config.json"
df -h /mnt/data
```

确认模型文件和余量后执行；`-L` 将源目录中的软链接转成实际文件，避免容器内无法解析链接。

```bash
mkdir -p "$AREAL_EVAL_ROOT/models/Qwen3-1.7B"
rsync -aL --info=progress2 "$AREAL_MODEL_17_SOURCE/" \
  "$AREAL_EVAL_ROOT/models/Qwen3-1.7B/"
```

确认旧评测已结束。若容器仍在运行，先执行下面两条；若已按第8节停止，直接从 `docker start` 继续。

```bash
docker exec wens-hermes-eval areal agent stop --service hermes-math-eval
docker stop -t 30 wens-hermes-eval
```

```bash
docker start wens-hermes-eval
docker exec wens-hermes-eval areal agent stop --service hermes-math-eval
```

重启保留已安装的 Python 包，停止命令同时清理旧 Agent 状态。容器原来的 `/model` 仍挂载4B；下面显式使用新的1.7B路径。

### 在宿主机生成模板并启动推理

从1.7B自己的聊天模板派生配置，只关闭 thinking，保留工具格式。文件使用 `.jinja` 扩展名，供固定 SGLang 版本加载。

```bash
docker exec -i wens-hermes-eval python - <<'PY'
import json
from pathlib import Path

config = json.loads(Path("/workspace/models/Qwen3-1.7B/tokenizer_config.json").read_text())
template = config["chat_template"]
assert isinstance(template, str), "Expected the Qwen3 Jinja chat template"
output = Path("/workspace/results/qwen3-17b/nonthinking.jinja")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text("{% set enable_thinking = false %}\n" + template)
PY

docker exec -d \
  -e 'AREAL_QWEN3_ROPE_OVERRIDES={"max_position_embeddings":65536,"rope_scaling":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768,"rope_theta":1000000}}' \
  wens-hermes-eval bash -c '
  exec python -u -m sglang.launch_server \
    --model-path /workspace/models/Qwen3-1.7B \
    --served-model-name hermes-math-17b \
    --host 127.0.0.1 --port 30000 \
    --tp-size 1 --dp-size 8 \
    --load-balance-method round_robin \
    --dtype bfloat16 --context-length 65536 \
    --json-model-override-args "$AREAL_QWEN3_ROPE_OVERRIDES" \
    --mem-fraction-static 0.5 \
    --chat-template /workspace/results/qwen3-17b/nonthinking.jinja \
    --tool-call-parser qwen \
    --api-key "$AREAL_INFERENCE_ADMIN_KEY" \
    > /workspace/results/qwen3-17b/sglang-64k.log 2>&1
'

docker exec wens-hermes-eval tail -n 50 -f /workspace/results/qwen3-17b/sglang-64k.log
```

服务就绪后按 Ctrl+C 退出日志查看，再按第4节检查 `/v1/models`，应返回 `hermes-math-17b` 且 `max_model_len`
为65536。此配置已在服务器的 SGLang 0.5.10.post1／Transformers 5.3.0
在原8轮预算下通过四题检查：全部完成、正确1题、基础设施错误0，模型调用8次、工具执行5次。恢复原生轮数后的评测另存；此配置直接输出最终回复，不添加
`--reasoning-parser`。

### 在容器启动 Agent 并评测

按第5节进入容器并启动 Agent；若服务已运行，直接查看状态。每次重启后端口可能变化，先刷新 URL：

```bash
areal agent status --service hermes-math-eval
export AREAL_AGENT_URL="$(areal agent status --service hermes-math-eval --json \
  | python -c 'import json, sys; print(json.load(sys.stdin)["gateway_url"])')"
```

Agent 四组件均为ok后，执行：

```bash
python -m examples.task_workflows.client --benchmark dapo_math \
  --tasks /workspace/tasks/eval.jsonl \
  --output /workspace/results/qwen3-17b/smoke-4-default \
  --agent-url "$AREAL_AGENT_URL" \
  --inference-url "$AREAL_INFERENCE_URL" \
  --model hermes-math-17b --mode eval --limit 4 --concurrency 32
```

四题全部完成、基础设施错误与未提交均为0后，继续：

```bash
python -m examples.task_workflows.client --benchmark dapo_math \
  --tasks /workspace/tasks/eval.jsonl \
  --output /workspace/results/qwen3-17b/eval-200-default-run1 \
  --agent-url "$AREAL_AGENT_URL" \
  --inference-url "$AREAL_INFERENCE_URL" \
  --model hermes-math-17b --mode eval --limit 200 --concurrency 32
```

汇总在宿主机
`/mnt/data/wens/hermes-eval/results/qwen3-17b/eval-200-default-run1/summary.json`。旧版客户端运行中可在另一个宿主机终端查看：

```bash
watch -n 2 'wc -l /mnt/data/wens/hermes-eval/results/qwen3-17b/eval-200-default-run1/episodes.jsonl'
```

结束后按第8节停止本次服务。原4B的150／200和1.7B的48／200使用8轮预算；同预算比较需将4B也按原生默认重跑。
