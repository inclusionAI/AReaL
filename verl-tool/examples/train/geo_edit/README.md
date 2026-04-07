# ChartQA RL Training with GeoEdit Tools

## 目录结构

```
geo_edit/
├── README.md                  # 本文件
├── PLAN.md                    # 完整训练计划（含所有已知问题和修复记录）
├── setup_env.sh               # 一键环境配置（pip 依赖 + .pth + model config 修复）
├── start_tool_server.sh       # 从 head 节点通过 Ray 在 worker 上启动 tool server
├── run_smoke_test.sh          # Smoke test（2 training steps，验证 pipeline）
├── run_instruct_rl.sh         # Instruct 模型正式 RL 训练（3 epochs）
├── run_thinking_rl.sh         # Thinking 模型正式 RL 训练（3 epochs）
├── preprocess_chartqa_rl.py   # 从 split_info.json 生成 RL 训练用 parquet
├── run.sh                     # verl-tool 原有的通用 geo_edit 训练模板
```

## 集群架构

```
HEAD  node:  192 CPU, 8 GPU (L20X 143GB), 0 tool_agent  → 训练
WORKER node: 192 CPU, 8 GPU (L20X 143GB), 8 tool_agent  → Tool Agents
```

训练在 HEAD 的 8 GPU 上运行，6 个 tool agents 占用 WORKER 的 6 GPU。

## 当前进展

### 已完成

- [x] RL 数据准备（train 1294 条，val 144 条）
- [x] pip 依赖安装（vllm、fire、colorlog、iopath、ftfy、pyzmq 降级、paddlepaddle-gpu、paddlex）
- [x] .pth 文件创建（两节点，让 Ray workers 能 import verl_tool/geo_edit）
- [x] tokenizer_config.json 修复（extra_special_tokens list→dict）
- [x] config.json rope_scaling 修复（从官方 Qwen3-VL-8B 复制）
- [x] actor.py setup_gpu() 添加唯一端口分配（解决 vLLM V1 多实例端口冲突）
- [x] Tool Server 5/6 agents 成功启动（chartr1 x3 + grounding_dino + sam3）

### 未完成（阻塞训练）

- [ ] **PaddleOCR agent 加载失败** — PaddleOCR-VL-1.5 需要 paddlex 的 vLLM 插件
  `register_paddlex_genai_models` 来注册自定义模型架构。插件安装后仍报
  `EngineCore initialization failed`，根因待查。
- [ ] Smoke test（因 paddleocr 阻塞）
- [ ] Instruct RL 训练
- [ ] Thinking RL 训练

## 复现步骤

### Step 0: 环境配置（两个节点都执行）

```bash
bash setup_env.sh
```

执行完后在两个节点杀掉旧 Ray workers 让 .pth 生效：

```bash
ps aux | grep 'ray::IDLE' | grep -v grep | awk '{print $2}' | xargs kill -9
```

### Step 1: 启动 Tool Server（在 HEAD 节点执行）

```bash
bash start_tool_server.sh
```

验证：
```bash
# 检查 health
WORKER_IP=$(python3 -c "
import ray; ray.init(address='auto',ignore_reinit_error=True)
for n in ray.nodes():
    if n['Resources'].get('tool_agent',0)>0 and n['Alive']:
        print(n['NodeManagerAddress']); break
")
curl http://$WORKER_IP:30888/health

# 检查 GPU 占用（期望 6/16）
python3 -c "
import ray; ray.init(address='auto',ignore_reinit_error=True)
a=ray.available_resources(); t=ray.cluster_resources()
print(f'GPU={t[\"GPU\"]-a.get(\"GPU\",0):.0f}/{t[\"GPU\"]:.0f}')
print(f'tool_agent={t[\"tool_agent\"]-a.get(\"tool_agent\",0):.0f}/{t[\"tool_agent\"]:.0f}')
"

# 检查 worker GPU 显存（期望 chartr1 x3 ~113GB, grounding_dino ~1.5GB, sam3 ~4GB）
python3 -c "
import ray; ray.init(address='auto',ignore_reinit_error=True)
@ray.remote(resources={'tool_agent':0.001})
def g():
    import subprocess
    return subprocess.run('nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits',shell=True,capture_output=True,text=True).stdout
print(ray.get(g.remote()))
"

# 检查 backend log 有无 agent 失败
python3 -c "
import ray; ray.init(address='auto',ignore_reinit_error=True)
@ray.remote(resources={'tool_agent':0.001})
def e():
    with open('/storage/openpsi/users/lichangye.lcy/antoinegg1/AReaL/tool-server-logs/tool_server_backend_0.log') as f:
        lines=f.readlines()
    return [l.strip()[:120] for l in lines if 'failed' in l.lower() and 'tool agent' in l.lower()][-5:]
print(ray.get(e.remote()))
"
```

### Step 2: Smoke Test（在 HEAD 节点执行）

```bash
bash run_smoke_test.sh
```

期望：2 个 training steps 无报错完成。

### Step 3: 正式训练

```bash
# Instruct 模型（3 epochs）
bash run_instruct_rl.sh

# Thinking 模型（3 epochs）
bash run_thinking_rl.sh
```

## 当前阻塞问题：PaddleOCR Agent

PaddleOCR-VL-1.5 模型有自定义架构 `PaddleOCRVLForConditionalGeneration`，其中包含
`mlp_AR` 等非标准模块。vLLM 0.11.0 需要 paddlex 的插件来注册该模型。

已尝试的修复（均未解决）：
1. `pip install paddlepaddle-gpu paddlex` → 插件注册但 pycocotools numpy 不兼容
2. `pip install --upgrade pycocotools` → numpy 兼容了但 EngineCore 仍 init 失败
3. `VLLM_USE_V1=0` → V0 engine 不支持 VLM
4. `setup_gpu()` 添加唯一 MASTER_PORT/VLLM_PORT → 解决了 chartr1 但 paddleocr 仍失败

可能的下一步尝试：
- 升级 vllm 到 0.11.1/0.11.2 看是否修复
- 检查 PaddleOCR-VL-1.5 的 modeling 代码是否需要适配 vllm 0.11.0
- 修改 paddleocr_tool.py 改用 transformers 直接加载（不走 vLLM）

## 关键路径

```
数据 → 环境 → tool server (6 agents) → smoke test → instruct RL → thinking RL
                             ↑
                        当前卡在这里 (5/6)
```
