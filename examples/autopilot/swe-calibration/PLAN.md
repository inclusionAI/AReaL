# 8×H20 SWE 校准计划

2026-10-19：用户确认 A–G、I、J 共9个候选 trial，取消 H 重复实验；J 具体选择
`fsdp:d2c2`（DP2×CP2／Ulysses）。配置／维度检查通过，真实训练与同步尚未验证，当前未启动训练。 本计划替代此前本地准备脚本中的7项草案；旧
`campaign.json`／固定4＋4的脚本不能直接执行此矩阵。

目标：补齐 Expert 的跨配置模拟与独立验证，为同一8卡环境的 offline revision 提供实测数据。 合法候选由后端、硬件和训练语义约束决定；Expert
当前未覆盖的布局属于待实现项，不因此排除。

## 固定条件

- 单机8×H20，Actor／Rollout 分卡；Qwen3-30B-A3B、模型 revision、四题及 agent／评分规则沿用
  [r10 档案](../swe-r10/README.md)。四题来自 Verified TEST split，仅作成本／训练 trial。
- 每个 trial 24 updates，共216 updates；每步4 accepted groups×4 samples，staleness=2，seed=42。
  每次从同一初始模型开始，逻辑 batch、sample reuse 和 loss normalization 保持一致。
- Actor 保留 BF16 Adam/Kahan、CPU offload、16k microbatch token 上限、`pad_to_maximum=false`；
  其余学习、采样与 agent 参数保持 r10 设置。`n_mbs` 是拆分约束，不代表实际 microbatch 数。
- 保存策略与观测配置在采集前统一确定；启动、warmup、训练主循环、保存和退出分别计时。 公司端8×H200按独立硬件档案记录，不能直接复用 H20 时间校准参数。

## 输入任务一致性

- 跨 trial 固定任务内容与逻辑输入序列：按 manifest 中 pytest→Django→Astropy→sklearn 的顺序构建96条记录，读完按同序循环。
  每次从序列起点开始，不恢复上一 trial 的 dataloader 状态。
- 固定
  `train_dataset.shuffle=false`、`num_workers=0`、`batch_size=4`、`drop_last=true`、`dynamic_bs=false`、
  `seed=42`、`gconfig.n_samples=4`。当前 V2 控制器统一读取数据，不随 Actor DP／CP 布局改变数据分片。
- 启动前校验有序任务清单、题目／fixture 内容、仓库初始 commit、环境镜像 digest 与 grading 脚本 hash；不依赖可变镜像 tag。 原
  trial 直接构建四题 Dataset，仅设置 YAML 的单题 `path` 不足以复现输入。
- 记录全局输入序号、cycle、record／instance ID 与内容 hash，核对各 trial 已读取的共同前缀一致；此审计仍待接入。 固定24个
  accepted-batch 更新时，拒绝补采样与预取会使读取总量不同；异步完成及 Actor 消费顺序不要求相同。
- 本轮以各 trial 的实际轨迹评估 Expert 误差，不要求生成输出或训练结果逐位一致，也不将输入任务一致称为实际 workload 完全一致。

## 候选矩阵

请求上限对应每个 SGLang 副本的 `max_running_requests`；“全局槽数”仅为各副本上限之和，不是共享池。

| Trial | Actor            | Rollout      | 并发 groups | 每副本请求上限 | 全局槽数 | `n_mbs` | 用途             |
| ----- | ---------------- | ------------ | ----------: | -------------: | -------: | ------: | ---------------- |
| A     | 4卡，`fsdp:d4`   | 4卡，DP2×TP2 |           4 |              4 |        8 |       1 | r10 配置基线     |
| B     | 4卡，`fsdp:d4`   | 4卡，DP1×TP4 |           4 |              8 |        8 |       1 | Rollout 布局     |
| C     | 4卡，`fsdp:d4`   | 4卡，DP2×TP2 |           8 |              4 |        8 |       1 | group 并发       |
| D     | 4卡，`fsdp:d4`   | 4卡，DP2×TP2 |           4 |              8 |       16 |       1 | 请求容量         |
| E     | 4卡，`fsdp:d4`   | 4卡，DP2×TP2 |           4 |              4 |        8 |       4 | microbatch 拆分  |
| F     | 4卡，`fsdp:d4`   | 4卡，DP1×TP4 |           8 |             16 |       16 |       1 | Rollout 组合留出 |
| G     | 4卡，`fsdp:d4`   | 4卡，DP1×TP4 |           8 |             16 |       16 |       4 | 两侧组合留出     |
| I     | 2卡，`fsdp:d2`   | 6卡，DP3×TP2 |           4 |              4 |       12 |       1 | 两侧 GPU 配比    |
| J     | 4卡，`fsdp:d2c2` | 4卡，DP2×TP2 |           4 |              4 |        8 |       1 | Actor 内部布局   |

A–E 用于校准，F–G 保持组合留出；I／J 的数据划分在采集前固定。 F–G 在模型冻结后评分，不能根据其误差反复调整模型并继续称为留出验证。 I
保持每副本4槽，增加一个副本；它同时改变两侧资源分配与总服务容量，不是固定全局槽数的单因素对照。

## 待完成与验收

1. 验证 J 的 DP2×CP2：每个 DP 单元接收2组／8条样本，两张卡执行 Ulysses 序列切分；逻辑 batch、loss 归一化不变。 32个 query
   heads／4个 KV heads 均可被 CP2 整除。后端按 `dp_sp` 的4卡 mesh 做 FSDP，仍保留四卡参数分片； 普通 packing 先按原16k
   token 上限组包，再切分序列，不将预算自动翻倍。需验证 padding／logp／梯度和 AWEX 接收值， 并补齐 Expert 的 CP 通信及 packing
   模型；真实30B训练与同步尚未验证，不预设 J 更快。
1. 将原试验适配从固定4训练卡／4推理卡／DP2×TP2扩展到候选拓扑；逐参数核验权重切片覆盖、传输与接收值。 小张量检查不能替代真实模型更新／发布验证；CPU
   offload 的显存与传输峰值分别记录。
1. 明确 CUDA graph batch 范围等非搜索项是否固定，避免请求槽实验混入未记录的配置变化；落实安全启动与自有资源清理。
1. 记录 group→sample/session→request→实际 replica→consumed update 关联，分开服务、工具、评分、清理、交付； Actor
   保存逐阶段／rank packing、有效 tokens、梯度与发布版本。训练组不得按跨 cycle 复用的 record ID 聚合。
1. 先按实际 workload 比较预测误差与覆盖率，再评估跨配置排序；保留失败／截断，基础设施错误不得算正常 reward=0。
   每次24步应覆盖96个已消费组、384条轨迹及所有 Actor rank 的成功更新记录。

当前状态统一维护在
[Expert 进度](../../../areal/autopilot/elastic/policy/expert_designed/PROGRESS.md)。
