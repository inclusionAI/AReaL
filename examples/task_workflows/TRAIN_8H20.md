# Qwen3-1.7B：八卡常驻 PPO 训练

使用固定 DAPO 题库训练 Hermes；算法沿用现有 online PPO。16题／4次更新已验收，正式训练使用同一1,000题重复5轮。本文不启用 on_demand。

| 项目         | 首轮设置                                                       |
| ------------ | -------------------------------------------------------------- |
| GPU          | FSDP4 Actor＋4个TP1 Rollout，合计8张H20                        |
| 模型         | `/workspace/models/Qwen3-1.7B-hermes-64k`，Actor／Rollout共用  |
| 上下文／模板 | 65,536 token、YaRN 2×、非thinking；Hermes原生90轮              |
| 每次生成     | 最多4,096 token，temperature=1、top_p=1                        |
| 优化         | AdamW（配置`type: adam`）、学习率`1e-6`；reward／adv归一化关闭 |
| 正式训练     | 固定1,000题×5轮＝5,000个episode，batch 40、125次更新、并发32   |

学习率是保守起点，未经调优；保持现有奖励尺度。batch计数题目episode；同题多次模型调用会产生多条训练序列，不增加题目数。具体配置在[configs/dapo_17b_resident.yaml](configs/dapo_17b_resident.yaml)，继承现有数学配方。

## 当前准备与验证

本组5轮运行于香港时间2026-10-23 03:25第4次更新CUDA
OOM失败，仅完成3次更新，无checkpoint；失败进程已清理。998条正常题目记录计时完整，另2题服务错误。微批拆分与分配器提案仅通过配置检查，不能保证64K长序列显存安全；重跑前须验证失败规模与64K完整前后向，见[PROGRESS.md](../../PROGRESS.md)。本页保留原运行步骤；验证后从原初始模型重跑，使用新目录与`trial_name`。

共享模型目录已准备：复用原权重软链接，模型配置与tokenizer均使用上述上下文和模板。真实八卡常驻训练已通过：16题、4次更新、65次模型调用、47次工具执行，基础设施错误0；HF权重已导出且确认发生变化，Trainer和客户端退出码均为0，八卡显存已释放。逐题计时完整，见[验收记录](../../../artifacts/hermes-qwen3-17b-20261023/train-resident-smoke-16.json)。

这轮仅验证闭环，不能据此判断学习收益。正式比较前的同预算200题基线使用`EVAL_8H20.md`第9节的新目录，评测结束后按第8节停止该评测容器，再开始下文。原24%结果采用8轮预算，不能直接用于90轮训练前后比较。

## 1 进入现有容器

以下命令在服务器宿主机执行；三个终端分别进入同一个容器，后续命令都在容器内运行：

```bash
docker start wens-hermes-eval
docker exec -it wens-hermes-eval bash
```

已停止本轮占满八卡的评测SGLang。训练启动前用`nvidia-smi`核对GPU；不要同时启动八副本评测服务。

## 2 终端一：启动训练

每次换新目录和`trial_name`，不要覆盖旧运行。容器已有推理和Agent密钥，Actor控制服务复用本实验的推理密钥。

```bash
cd /workspace/AReaL
export AREAL_TASK_RUN_ROOT=/workspace/results/qwen3-17b/train-1000x5-run1
export AREAL_MODEL_PATH=/workspace/models/Qwen3-1.7B-hermes-64k
export AREAL_ACTOR_ADMIN_KEY="$AREAL_INFERENCE_ADMIN_KEY"
mkdir "$AREAL_TASK_RUN_ROOT" || exit 1
set -o pipefail

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TOKENIZERS_PARALLELISM=false \
python -m examples.task_workflows.train \
  --config examples/task_workflows/configs/dapo_17b_resident.yaml \
  2>&1 | tee "$AREAL_TASK_RUN_ROOT/trainer.log"
```

等待`RolloutControllerV2 ... Gateway: http://...`出现后再提交题目。该终端持续运行；先等题目到达属于正常状态。

## 3 终端二：启动训练专用Agent

新实验前停止旧的同名训练Agent。`HERMES_HOME`只声明实际64K上下文，不设置轮数；训练网关不提供评测SGLang的上下文探测字段。

```bash
cd /workspace/AReaL
export AREAL_TASK_RUN_ROOT=/workspace/results/qwen3-17b/train-1000x5-run1
export HERMES_HOME="$AREAL_TASK_RUN_ROOT/hermes-home"
export AREAL_TASK_TOOL_ROOT="$AREAL_TASK_RUN_ROOT/tools"
mkdir -p "$HERMES_HOME"
printf 'model:\n  context_length: 65536\n' > "$HERMES_HOME/config.yaml"

areal agent stop --service hermes-math-train --grace-period 2
env -u HERMES_MAX_TURNS CUDA_VISIBLE_DEVICES="" \
  areal agent run --service hermes-math-train \
  --agent examples.task_workflows.service.TaskAgent \
  --num-pairs 1 --admin-api-key "$AREAL_AGENT_ADMIN_KEY"

areal agent status --service hermes-math-train
```

## 4 终端三：同一1,000题按相同顺序运行5轮

下段从本轮日志获取Rollout网关，避免误用旧评测端口或Actor网关。若网关尚未启动，先查看终端一，待就绪后重试。

```bash
cd /workspace/AReaL
export AREAL_TASK_RUN_ROOT=/workspace/results/qwen3-17b/train-1000x5-run1
export AREAL_MODEL_PATH=/workspace/models/Qwen3-1.7B-hermes-64k
export AREAL_AGENT_URL="$(areal agent status --service hermes-math-train --json \
  | python -c 'import json,sys; print(json.load(sys.stdin)["gateway_url"])')"
export AREAL_INFERENCE_URL="$(python - <<'PY'
import os, pathlib, re
text = pathlib.Path(os.environ["AREAL_TASK_RUN_ROOT"], "trainer.log").read_text()
text = re.sub(r"\x1b\[[0-9;]*m", "", text)
urls = re.findall(r"RolloutControllerV2[^\n]*Gateway: (http://\S+)", text)
if not urls:
    raise SystemExit("Rollout gateway is not ready; inspect trainer.log")
print(urls[-1])
PY
)"
test -n "$AREAL_AGENT_URL" && test -n "$AREAL_INFERENCE_URL" || exit 1
for epoch in $(seq 1 5); do
  printf 'Epoch %s/5\n' "$epoch"
  python -m examples.task_workflows.client --benchmark dapo_math \
    --tasks /workspace/tasks/train.jsonl \
    --output "$AREAL_TASK_RUN_ROOT/client-epoch-$epoch" \
    --agent-url "$AREAL_AGENT_URL" \
    --inference-url "$AREAL_INFERENCE_URL" \
    --model "$AREAL_MODEL_PATH" --mode resident \
    --limit 1000 --concurrency 32 || break
done
```

每轮按同一冻结清单的顺序提交题目，重新采样并创建独立session；前一轮客户端完成后再提交下一轮，Trainer持续运行。异步完成顺序不保证一致，客户端轮次与每25次更新的保存边界可有少量交错。训练服务收集奖励轨迹后执行更新并发布权重，见[原生online训练说明](https://github.com/areal-project/AReaL/blob/main/docs/en/tutorial/online_proxy.md)。

## Rollout batch 采集

在终端一启动 Trainer 前设置，输出目录使用本次运行的新目录：

```bash
export AREAL_ROLLOUT_TRACE_DIR="$AREAL_TASK_RUN_ROOT/rollout-batches"
```

推理进程会追加 `batches-<host>-<pid>.jsonl`，分别保存 prefill／decode 请求数、完整上下文 token 总量及本次处理 token
数；mixed batch 拆分计数。GPU 时间来自 CUDA event，保留与墙钟对齐的估计范围。 字段、普通 TP／DP
支持范围和公司端通用配置见[采集说明](../../docs/zh/best_practices/perf_profiling.md#sglang-batch-%E8%AF%B7%E6%B1%82%E6%95%B0%E4%B8%8E-token-%E6%97%B6%E5%BA%8F)。这项采集不代表
Actor OOM 修正已通过验证。

## 完成标准与输出

- 每轮`client-epoch-N/summary.json`均为1,000题完成、基础设施错误0、未提交0；总计5,000个episode。每题计时在对应目录的`episodes.jsonl`。
- Trainer：125次更新全部完成，最终HF权重导出后正常退出。客户端完成不等于训练完成。
- 每25次更新保存一次，共5份HF权重，可选第1、3、5份作阶段评测。最终权重位于`$AREAL_TASK_RUN_ROOT/checkpoints/root/hermes-math/qwen3-17b-train-1000x5-run1/default/epoch4epochstep24globalstep124/`，对应更新后的版本125。
- 本机实测每份HF权重约3.8GiB，5份约19GiB，另留工具和日志空间。
- `recover.mode=disabled`：导出仅含HF模型，不保存可精确续训的优化器／RNG状态；resident仍在内存维护AdamW状态。
- 客户端发生基础设施错误或轨迹被丢弃时，最后一批可能收不满。检查日志并在终端一Ctrl+C停止该次Trainer，保留现场，不能将客户端退出算成成功。

训练后用导出目录替换评测模型，保持同一200题、90轮、采样和工具预算，再比较正确率与耗时。正式质量结论还需要未用于挑选模型的保留题集。
