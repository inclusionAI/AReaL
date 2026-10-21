# r10 SWE 配置与 8×H200 交接

2026-10-19；已验证环境为单机 8×H20，目标为公司端 8×H200 的相同配置基线。 本目录归档配置和运行契约；**未包含原本地 `trial.py`
与桥接模块，不能直接交给官方 SWE 入口启动**。 公司端先按下述契约接入自己的运行入口，H200 的可运行性与性能另行验收。

## 配置入口

| 文件                                           | 内容                                                          |
| ---------------------------------------------- | ------------------------------------------------------------- |
| [trial.yaml](trial.yaml)                       | 实际 r10 YAML，仅将五处机器路径改为环境变量                   |
| [agent-config.yaml](agent-config.yaml)         | 原 agent prompt、工具、24 轮／4096 tokens 限制                |
| [dataset-manifest.json](dataset-manifest.json) | 四题顺序、数据 revision、grading hash；fixture 路径相对化     |
| [run-contract.json](run-contract.json)         | 版本、隐式默认值、进程环境、运行契约、H20 结果与原始文件 hash |

环境变量：`R10_MODEL_PATH` 为模型目录，`R10_OUTPUT_ROOT` 为本次独立输出目录， `R10_FIXTURE_ROOT` 为四题 fixture
目录，`TRIAL_ADMIN_KEY` 为本次运行生成的管理凭证。 配置中的 `trial_name` 为历史名称，新实验应覆盖为唯一名称；机器路径、凭证、SSH
配置和权重均不入库。

## 固定基线

| 项目          | r10 实际设置                                                                                               |
| ------------- | ---------------------------------------------------------------------------------------------------------- |
| 模型／算法    | Qwen3-30B-A3B，GRPO，seed=42；model revision 见契约                                                        |
| Actor         | 4 GPU，`fsdp:d4`，BF16 参数／梯度／Adam 状态，Kahan 更新，gradient checkpointing                           |
| Actor 内存    | `fsdp.offload_params=true`、`memory_efficient_load=false`；顶层 `enable_offload=false`                     |
| Actor packing | `n_mbs=1`、`max_tokens_per_mb=16384`、`pad_to_maximum=false`；`n_mbs` 不是实际 MB 数，仍受 token 上限拆分  |
| Rollout       | 4 GPU，`sglang:d2t2p1`：两个独立 TP2 副本；内部 DP attention／EP 关闭                                      |
| 并发          | 4 在途 groups×4 samples；每副本 `max_running_requests=4`，合计 8 请求槽                                    |
| 推理          | context=16384，max prefill=16384，chunk=4096，memory fraction=.75，CUDA graph max batch=4                  |
| 训练语义      | 16 updates；每步 4 accepted groups×4 samples；一个 PPO minibatch；staleness=2                              |
| 采样／轨迹    | temperature=top_p=1，单次 completion≤4096，agent≤24轮；concat template／export，无 `/no_think`             |
| 优化器        | lr=3e-6，weight decay=.01，warmup=.001，constant schedule，clip grad=1                                     |
| GRPO loss     | group reward mean/std；KL=0；clip=.2/.28；decoupled loss；token ratio>5 mask；`prox_logp_method=recompute` |
| 保存          | 第16步同步导出 HF；无 optimizer recovery，无 evaluator；保存单独计时                                       |

`recompute_logprob=false` 与 `prox_logp_method=recompute` 同时存在，实际仍有 `recompute_logp`
阶段。完整字段以 YAML 和基准 commit 默认值为准。

## 数据、agent 与评分

- 四题按 manifest 顺序重复16次，构造64条逻辑 records。原 `trial.py` 将构造的 Dataset 直接传给 `PPOTrainer`， 因而
  YAML 中的 pytest 单题路径不是实际训练数据来源。`dynamic_bs=false`，被拒绝的组自动补采样； record ID
  可在循环后复用，训练组按唯一执行 ID／update 关联。
- 数据来自 **SWE-bench Verified TEST split**，仅作成本／训练链路 trial；不能报告为泛化评测。 gold patch
  只用于环境验收，不能进入 agent 输入。四题各自验证空补丁=0、gold=1与独立清理。 fixture
  构造及基础设施见[现有 SWE 说明](../elastic/swe/README.md)。镜像 tag 不是 digest，迁移时另存实际 digest。
- 除 agent YAML 外，r10 有 context admission guard：最近一次请求 usage≥10000，或其与新增消息 UTF-8 JSON
  bytes 加4608预留超过16384时，停止新请求并评分当前 patch。这里使用单次 usage，不是累计 usage。
- 评分前重新提取 live diff 并与最终 patch 比较；对已经修改的 worktree 调用 `get_reward(patch='')`， 仍执行原
  grading script，避免再次 apply patch。SDK／基础设施错误应拒绝，不能转为 reward=0。
- 缺完整 grading marker 时，仅允许有严格证据的候选 SyntaxError／IndentationError／TabError：同一 testbed 解释器下
  HEAD 可编译、错误文件／行／类别匹配、评测前后源码 hash 不变、测试实际开始结束、SDK failures=0。 其他缺 marker 的错误仍拒绝。sample
  评分返回后独立核验 Pod 删除，四个 sample 完成后组才能交付。

## r10 专用兼容处理

原入口使用 `PPOConfig` 与 `TrialFSDPPPOActor`，不是 `examples/swe/train_swe_rl.py` 的原始 SWE
workflow。 下列模块属于试验外部依赖，原文件 hash 已归档；现有基础设施脚本不包含这些全部修复。

| 原模块                                      | 公司端须保留或在目标版本验证已解决的行为                                                                                        |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `trial_engine.py`                           | Transformers 5 packed experts 按专家轴 FSDP ownership 拆成逐专家 AWEX views；每次从最新 CPU shards staging 到 GPU，不跨版本缓存 |
| `trial_awex_meta.py`                        | 合并同名 TP metadata、去重，保留独立 NO_SHARDING replicas；验收8 rank计划、16对peer顺序及完整不重叠覆盖；原实现限定 DP2×TP2     |
| `trial_awex_runtime.py`／`sitecustomize.py` | `AWEX_WU_USE_GROUP=0` 手动 P2P；通信前保证 contiguous copy 完成，清理已消费梯度和空闲缓存，记录8 rank实际传输完成               |
| `trial_prefix.py`                           | 仅规范 assistant 空 `tool_calls` 的 missing／None／\[\]；保持每组恰好4条 concat 轨迹，不能挑叶子凑数                            |
| `agent_worker.py`                           | context guard、live-worktree 评分及严格失败判断；agent 与 sandbox SDK 分进程解决依赖冲突                                        |

基准 AReaL commit 为 `8aaafd1e`，运行镜像 tag 为 `v2.1.0-sglang`；不是公司历史 Tau2 的 AReaL v1。 已确认
PyTorch 2.9.1、Transformers 5 packed experts；完整环境 lock／镜像 digest 未随此档案保存， 公司端记录实际
PyTorch／Transformers／SGLang／AWEX／CUDA 版本并核对兼容性。

## 公司端执行与观测要求

1. 首轮保留上述训练语义和 CPU offload，仅替换硬件与部署路径。若后续关闭 offload、改变保存或数据，另记配置。
   先通过四题评分／清理及真实30B四卡训练→四卡同步检查；H20 验收不能代替 H200 验收。
1. 保存 resolved config、硬件／互联／CPU RAM、版本、任务／镜像与脚本 hash；逐步保存 reward、advantage、 有限梯度／成功更新和
   AWEX version。16步应有64份 rank optimizer 记录与256条已消费轨迹。
1. 为 Expert 补观测：关联 group→sample/session→request→实际 replica 与 consumed update；记录请求到达／排队／
   prefill／decode／响应、工具、评分、清理、group交付边界，以及 Actor 每阶段／rank 的实际 packing 与耗时。 缺失时标未知；r10
   只有合并 RPC，真实 replica 映射未知，不能从它推造独立阶段标签。
1. 启动、warmup、保存与稳态耗时分开报告。硬件改变后的 token 数、轨迹和 batch 可能不同， Expert 先以各自实际 workload 做条件预测，H20
   的时间校准参数不直接当作 H200 参数。

H20 参考：16步主循环 **1809.49s**，其中 train **892.86s**、logp **63.15s**、advantage **16.59s**、 等
batch **535.25s**、权重发布 **141.84s**、保存 **157.35s**；Rollout 与 Actor 重叠。 64个训练组中20组 reward
非恒定，13步非零梯度，消费样本 mean reward **0.21094**，前后8步均值相同。 全部300个有效评分含预取和拒绝组
siblings，不能当作训练样本；HF 仅做 header／索引审计，未重载评测。

时间线与绘图标准见
[Expert 进度](../../../areal/autopilot/elastic/policy/expert_designed/PROGRESS.md)。 本地后续
action space 仍在讨论，本次提交只交接 r10 已运行配置。
