# AutoPilot Harness

当前实现 Monitor 的单次采集：整理训练说明、当前配置、运行观测、执行历史，保存为一个上下文快照。支持
W&B、Prometheus，以及经过相同解析流程的本地合成数据。无需启动 AReaL；没有 GPU 也能运行和测试。

Validator 提供候选配置静态检查：覆盖修改权限、参数取值、资源、布局、显存及状态切换约束。检查上下文由 harness 提供，Planner
只提交候选配置；Validator 不执行运行时修改。

Planner 当前提供占位 `plan()`：接收上下文，返回当前配置的副本，标记 `changed=False`、`search_status="not_run"`。

开发基线为 AReaL **v2.1.0**（`ecc8b0e4dfb4e3965f67121a5c87866a07c390ae`）。Harness 已实现同配置重启与经过
Validator 的 Config Apply，已接入 FSDP 的 DP/TP/CP 布局映射及 Megatron 的 portable checkpoint
映射。环境采用该版本对应的依赖锁文件，GPU 训练与恢复验证待有设备后开展。

## 工程位置与安装

本工程位于 `AReaL/autopilot-harness/`，与上层训练运行时共享 Git 仓库，保留独立的 Python 包、依赖、配置和测试。从 AReaL
仓库根目录可单独安装 Harness：

```bash
python -m pip install -e ./autopilot-harness
cd autopilot-harness
```

下文命令均在 `autopilot-harness/` 中执行。使用 `PYTHONPATH=src` 的离线样例无需安装包；CPU 样例与测试从上层仓库读取 AReaL
源码，不要求 GPU 训练依赖。`runs/` 保存本地产物，不提交到 Git。

## 先运行离线样例

在 `autopilot-harness/` 下执行（Python 3.11+，标准库即可）：

```bash
PYTHONPATH=src python3 -m autopilot_harness monitor \
  --config examples/monitor/config.json \
  --output runs/monitor-demo
```

这个样例提供两种来源的数据，包含稀疏的 W&B history 和两个 Prometheus instance。数据全部是合成的，输出标记为
`fixture`，不代表真实训练测量。再次运行时使用新的输出目录。

输出目录：

```text
runs/monitor-demo/
├── manifest.json           # 采集输入、harness 版本、来源状态
├── events.jsonl            # 已有历史的副本 + 本次采集事件
└── snapshots/monitor.json  # 下列四类信息
```

| 快照字段                 | 内容                                                      |
| ------------------------ | --------------------------------------------------------- |
| `training_specification` | 代码与版本、工作流、模型、资源；缺项列在 `missing_fields` |
| `current_configuration`  | 配置、来源、标识及生效确认状态                            |
| `runtime_observations`   | 指标样本、时间、单位、实体标签、覆盖情况、来源状态        |
| `execution_history`      | 当前训练 run 已记录的上下文、决策、应用、结果、修订等事件 |

## 连接真实服务

安装为 Python 包，启用可选的 W&B SDK：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[wandb]'
```

复制 `configs/monitor.example.json`，填入 W&B run、Prometheus 时间范围、实际指标和标签。训练说明中未知的信息可以保留
`null`。配置中的相对文件路径以该 JSON 文件所在目录为基准；资源 URI 只作为引用保存，不会自动读取文件或执行代码。

凭据通过环境变量提供，JSON 不需要包含密钥：

| 信息             | 配置／环境变量                                                                                  |
| ---------------- | ----------------------------------------------------------------------------------------------- |
| W&B run          | `wandb.run`，支持 `entity/project/run_id` 或 `/entity/project/runs/run_id` 网页链接             |
| W&B API 地址     | `wandb.base_url` → `WANDB_BASE_URL` → `ALAB_WANDB_INTERNAL_BASE_URL`；未配置时使用 W&B 公有 API |
| W&B API key      | `WANDB_API_KEY` → `ALAB_WANDB_API_KEY`；可用 `wandb.api_key_env` 指定其他变量                   |
| Prometheus 地址  | `prometheus.base_url` → `PROMETHEUS_URL` → `ALAB_PROMETHEUS_URL`                                |
| Prometheus token | 可选 `PROMETHEUS_BEARER_TOKEN`；可用 `prometheus.bearer_token_env` 指定其他变量                 |

ALab 的 W&B 客户端还会读取它自己的 `STATE_ROOT/wandb-api-key`，并将公网展示地址映射到内部 API 地址；Prometheus
客户端则直接请求 `/api/v1/query_range`。这里兼容它使用的环境变量，私有 W&B 请显式填写 API 地址。ALab 中保存的密钥不会自动复制到
harness。地址配置要以运行 Monitor 的机器实际可访问的服务为准。

```bash
autopilot-harness monitor --config configs/my-monitor.json --output runs/capture-001
```

可以只配置其中一个来源。保留的 W&B run 历史和 Prometheus 时序数据可以在训练结束后查询，但仍需要服务可访问且数据未过期。PromQL
表达式显式指定聚合方式和实验／实例／节点标签；示例中的标签不是所有部署通用的。节点级 GPU 指标保持节点粒度，不自动归属到某个训练角色。

## 输出的边界

- W&B 使用 `scan_history()`，保留稀疏行和原始 `_step`；不会把 `_step` 当作训练步。若数据中有独立训练步字段，在
  `wandb.train_step_key` 中指定，否则 `train_step` 为 `null`。
- Prometheus 保留每条 series 的标签及查询时间。这个时间是 PromQL 求值时间，不等同于原始 scrape 时间；两种来源不会自动按训练步拼接。
- `observed_at` 是来源时间，`available_at`
  是本次采集完成时间。读取旧数据不能证明这些数据在过去某个决策边界已经可用，当前版本不做这类在线回放判定。
- 每个观测有
  `value/unit/entity/window/coverage/basis/source_ref`。缺失、非有限数不会补零；空结果、部分缺失、截断和错误都会在来源报告中出现。返回
  series 之外的缺失实例数量未知。
- W&B 配置默认仅为 `recorded`。可通过 `current_configuration` 显式提供 `values` 和 `source_ref`；若标为
  `confirmed`，必须同时提供 `confirmation` 中的
  `event_id`、`segment_id`、`applied_at`。这是上游执行模块提供的生效证据声明，Monitor 当前不独立验证远端应用状态。
- `history_path` 可以指向上一次输出的 `events.jsonl`。仅载入同一 `run_id` 且在采集开始前已记录的事件，保留各个
  segment；不修改原文件。`monitor/outputs.py` 中的 `HistoryStore` 支持追加事件，修订也写成新事件。当前按单写入进程使用。
- 一个来源出错时仍保存另一来源的结果。退出码 `0` 表示所有配置来源返回完整数据，`1` 表示空／部分／失败且快照已保存，`2`
  表示输入或文件错误。训练说明和配置是否完整须另看各自状态。

Monitor 当前实现单次采集、标准化与历史保存。配置中的常见凭据字段保存时会脱敏；不要把密钥放进自由文本、指标标签或查询表达式中。

## 检查候选配置

在工程目录运行 CPU 样例：

```bash
PYTHONPATH=src python3 -m autopilot_harness validate \
  --baseline examples/validator/baseline.json \
  --candidate examples/validator/candidate.json \
  --policy configs/validator.json \
  --context examples/validator/context.json \
  --runtime-revision 7599750a260aac07281d2c01694e70e045bf8ec8 \
  --output runs/validator-demo/result.json
```

示例是缩减的合成配置，`context.json` 中的 GPU、模型、显存上界和切换支持声明也全部为合成数据，只用于 CPU 接口测试，不能作为真实运行的许可或直接启动
AReaL。实际使用时，基准和候选都必须是已解析默认值与插值的完整配置；Monitor 快照中的 `current_configuration.values`
可作为配置输入。可编辑安装后，也可使用 `autopilot-harness validate`。

`configs/validator.json` 与当前核对的 AReaL commit 绑定。基准、规则和 `runtime_revision` 由 harness
提供；Planner 只提交候选，不能从候选中覆盖规则。规则包含草稿列出的 14 个原生路径，未列出的所有字段都固定，包括训练
batch、采样、奖励、staleness、优化器和角色放置。配置结构必须保留，可调字段也不能新增或删除；可空字段使用显式 `null`。

静态检查包括：

- 递归比较完整配置，保留类型差异；数组作为整体检查。报告使用路径片段数组，避免把带点的键名误当嵌套字段。
- 按字段检查整数、数值、布尔值、字符串、可空性、上下界和可选值。默认规则给出基本范围，具体任务可由 harness 收紧，不推测硬件容量上限。
- 只允许调整当前推理后端对应的 SGLang 或 vLLM 参数；backend 字段只允许提出布局调整，不能替换实现。
- 检查 padding 所需的显式 token 上限，以及当前 Megatron 路径所需的非空正整数
  `n_mbs`。这些依据在规则和结果中保留源码引用；不套用固定长度训练的 batch 公式去猜测 AReaL 的动态序列打包关系。
- 解析原生 DP/TP/PP/CP/EP 和 attention/FFN 折叠布局。GPU 数按 `DP×TP×PP×CP` 计算，EP 不重复计卡；检查角色 GPU
  池、实际 rank 分配、共置／分离关系、后端支持和节点跨度。
- 检查 attention/KV heads、层数和 experts 的分割条件，区分支持 KV 复制的推理后端与 FSDP，允许有明确依据的非均匀 PP 分层；检查
  SGLang 实例内 DP/EP 与 TP 的关系。训练 batch 按完整样本组分发，可按后端能力允许 DP 间不均匀。
- 对每张预留 GPU 汇总所有驻留角色的参数、梯度、optimizer、activation、KV、buffer
  和其他显存上界，加上切换额外峰值；同时检查推理显存比例、host memory 和 checkpoint disk
  预算。缺项不当作零，观测到的峰值或模型预测不自动当作上界。
- 检查一次完整修改所需的状态处理支持，包括参数、optimizer、scheduler、梯度、数据游标、RNG、已完成与在途 rollout、policy version
  和设备驻留。布局改变不能直接保留旧 optimizer 分片；缺少支持、恢复依据或遗漏同时修改的字段均不能通过。

`--context` 使用独立 JSON 对象，各部分见示例：

| 字段                                                          | harness 提供的依据                                                                                                                        |
| ------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `runtime_revision`、`baseline_id`、`candidate_id`、`state_id` | 精确绑定代码版本、配置对和所检查的状态；配置 ID 可用 `validator.configuration_id(config)` 生成                                            |
| `model`                                                       | 模型维度与来源；复杂模型使用该后端实际参与切分的维度                                                                                      |
| `resources`                                                   | 固定 GPU 清单、每卡容量及不可用显存、各角色设备池、候选 rank 顺序、共置组，以及 CPU 内存／磁盘预留                                        |
| `backends`                                                    | 针对模型和后端版本确认的布局列表、KV 复制规则、PP 层分配及每个实例的最大节点跨度                                                          |
| `dispatch`                                                    | 工作流实际送入训练的样本数、不可拆分的样本组大小，以及 DP 是否允许不均匀分发                                                              |
| `memory`                                                      | 针对候选和固定工作负载的保守上界；每卡 `roles` 必须覆盖全部驻留角色，空闲卡为 `{}`；`transition_extra` 包含旧／新状态重叠、暂存及恢复峰值 |
| `transition`                                                  | 对整个配置对的切换支持声明、适用边界、各项状态处理及恢复依据；无配置变化时不需要                                                          |
| `auxiliary.teacher`                                           | 共享 SGLang／vLLM 参数发生变化时，teacher 自己的 `model`、`backends.teacher` 支持描述和 `inference_memory_fraction`                       |

显存各项及主机／磁盘容量单位都是 bytes；每个可为零的项仍必须明确填写。`inference_memory_fraction` 是后端实际解析的显存比例，原配置为
`null` 时也必须提供。`resources.rank_devices` 中 rollout 按独立实例依次排列，每实例占 `TP×PP`
张卡；其他固定角色也要列入，避免遗漏共置模型的显存。`transition.state_mapping` 中每项带 `method` 和
`source_ref`：一般保留状态用 `preserve`；actor 布局改变时参数／optimizer 用 `reshard`、RNG 用
`remap`；梯度可声明边界处已清空 `none_at_boundary`；在途请求允许有依据的 `preserve`、`drain` 或
`migrate`，不接受直接丢弃。

以上 `transition` 声明用于离线检查，不能直接授权在线 Apply。在线调用另传当前实例的 `execution` 绑定，由固定的 adapter
检查实际支持范围并生成映射，不能用 `supported=true` 绕过。其他来源引用只保留溯源，不会自动抓取或验证引用文件：harness
必须从可信的资源分配、模型信息及后端实现／验证记录构建上下文，不能使用 Planner 自报的支持或显存预测。缺少实际依据时结果为 `undetermined`。

| `status`       | 含义                                                       |
| -------------- | ---------------------------------------------------------- |
| `passed`       | 对绑定的配置对与可信上下文，所有必需检查通过               |
| `rejected`     | 发现越权、非法值、资源超限、不兼容布局或不受支持的状态处理 |
| `undetermined` | 缺少依据、显存上界未知，或依据与本次配置／版本不匹配       |

所有结果带有 `scope=configuration_and_transition`。`runtime_applicability` 为
`supported`、`not_required`（无变化）、`unsupported` 或 `undetermined`。没有 `--context`
时仍可发现字段错误，但不再给出只检查取值的 `passed`。通过结果绑定 `context_id` 和
`state_id`；配置或相关状态变化后需要重新验证，不能复用旧结果。通过也不表示配置已经生效。

结果记录配置与规则的内容标识、变更字段、错误原因和检查规则，不输出配置值，也不修改基准或候选。指定的结果文件必须不存在；省略 `--output` 时输出 JSON
到终端。退出码：`0` 为静态检查通过，`1` 为拒绝或无法判定，`2` 为输入文件、规则格式或输出错误。结果可以作为后续执行历史事件的 payload，当前命令不自动追加
Monitor 的历史文件。

## 调用占位 Planner

`contracts/planner.py` 定义 `PlannerInput` 和 `PlannerResult`，`planner/runtime.py` 提供
`plan()`。已有 Monitor 快照可直接接入：

```python
from autopilot_harness.contracts.planner import PlannerInput
from autopilot_harness.planner import plan

# snapshot 是 monitor.collector.collect(...) 返回的 MonitorSnapshot。
inputs = PlannerInput.from_snapshot(snapshot)
result = plan(inputs)
# result.config：snapshot.current_configuration.values 的深拷贝
# result.changed：False
# result.search_status："not_run"
```

输入保留训练说明、当前配置、观测和历史的接口，当前只使用配置值。返回值保留完整配置的字段和类型，修改返回值不会影响 Monitor 快照。这个占位入口不初始化
Planner，不调用 Model，不执行搜索或版本切换，也不把配置标为已生效。

调用方仍需将 `result.config` 交给 Validator。原样返回只表示没有提出配置调整，不代表配置通过检查。`result.to_dict()`
提供统一结果的字典表示；历史记录仍由调用方组织，本接口不自动写文件或应用配置。

## 同配置重启

下面的 CPU 样例和可选接口集成测试读取上层 AReaL 仓库的源码，使用 `lws/feat/autopilot` 集成分支即可。

在工程目录运行不需要 GPU 的完整编排样例：

```bash
PYTHONPATH="$PWD/src" python3 -m autopilot_harness restart \
  --launch examples/execution/launch.json \
  --output runs/restart-demo \
  --completion-timeout 30
```

输出目录必须是新目录。样例使用 CPU 计数器、子进程与上层 `areal/` 中的真实快照模块，展示保存、退出、重新启动和续步；它不启动 GPU 训练。CLI
成功时先打印恢复确认，再等待新实例完成训练与资源清理，最后以 `0` 退出。

程序接口允许在需要切换时才发起重启，也允许多次重启仍在运行的实例：

```python
from pathlib import Path
from autopilot_harness.execution import LaunchSpec, RestartCoordinator
from autopilot_harness.validator import configuration_id

spec = LaunchSpec(
    argv=("python3", "train_managed.py", "--config", "train.yaml"),
    cwd=Path("/path/to/training-project"),
    run_id="experiment-001",
    runtime_revision="the-recorded-AReaL-revision",
    configuration_id=configuration_id(resolved_configuration),
)
with RestartCoordinator(spec, Path("runs/experiment-001")) as runtime:
    runtime.start()
    # 在所需时机调用；返回前已确认旧进程组退出和新实例恢复。
    result = runtime.restart()
    runtime.wait(timeout=86400)
```

`resolved_configuration` 是包含默认值与插值结果的完整 AReaL 配置字典。`restart()` 保持当前已生效配置；配置变化使用下面的
`apply(plan)`。`configuration_id` 只是身份绑定，不能代替 Validator 的合法性检查。`argv`、工作目录和 `environment`
来自可信训练入口；CLI 的 launch JSON 使用相同字段，`cwd` 相对于该 JSON 文件，可选 `timeouts` 指定
startup／snapshot／exit／restore／poll 秒数。

接真实 AReaL 时，训练脚本照常解析配置和准备数据集，把原先的 `with PPOTrainer(...): trainer.train(...)` 替换为：

```python
from autopilot_harness.execution.runtime import run_areal

run_areal(config, train_dataset=train_dataset, valid_dataset=valid_dataset,
          workflow=workflow, workflow_kwargs=workflow_kwargs)
```

该脚本需要通过上述 Harness 启动，运行环境需同时能导入 harness 和带有本地保存／恢复扩展的 AReaL；上游原版 v2.1.0
尚无这些扩展。适配器核对解析后的配置标识，并计算实际 AReaL Python 源码摘要，要求重启前后一致；`runtime_revision`
是调用方记录的版本标识，依赖环境仍需由实验环境固定。

调用顺序为：

1. Harness 启动独立训练进程组，等待初始化回执。
1. Harness 发保存并停止请求；训练进程在步边界调用 AReaL 快照接口。
1. `train()` 返回 `SnapshotStopResult`，离开 Trainer 上下文完成清理，发布保存／清理回执。
1. Harness 核对快照，等待主进程退出码为 `0` 且整个进程组消失，再拉起新实例。
1. 新实例显式恢复该快照；恢复流程重新开放 rollout 后发出回执。Harness 核对来源、配置和代码身份后记录
   `restart.confirmed`。若快照已在最后一步，则确认正常完成与清理。

`segments/<id>/` 保存独立的 `launch.json`、请求／回执与 `runtime.log`；`snapshots/` 保存 AReaL
快照；`events.jsonl` 通过 Monitor 的 `HistoryStore` 记录过程。该路径可作为 Monitor 的
`history_path`，收集时使用同一 `run_id`。历史中的 `restart.confirmed` 表示同配置恢复已确认，不表示新配置 Apply
或训练性能已验证。

失败时 CLI 返回 `1`，记录失败阶段。保存、清理或旧进程组退出失败时不启动替代实例；上下文退出保留源进程，CLI 显示其
PID／日志位置，源进程可能仍在训练、保存或清理。需要主动取消时显式调用 `runtime.close()`。新实例恢复失败则清理该实例并保留源快照；当前没有自动回滚或
Harness 崩溃后的自动接管。正常上下文退出会清理仍由它持有的进程组。

当前进程退出后端针对 POSIX 上的标准 v1 LocalScheduler，不支持
Ray、Slurm、v2、在线外部会话或自行脱离进程组的自定义启动命令。进程组检查不能覆盖逃离该进程组的第三方子进程；实际后端及其依赖的退出行为仍需 GPU
环境验收。此时不应把本地 CPU 编排测试作为集群资源已经释放的证据。

## 应用候选配置

在 `runtime.start()` 后，取得 Planner 的候选与 harness 提供的完整静态依据，再按顺序调用：

```python
from autopilot_harness.execution import build_apply_plan
from autopilot_harness.validator import validate_configuration

execution = runtime.execution_context()
checked = validate_configuration(
    baseline, candidate, policy,
    runtime_revision=policy.runtime_revision,
    context=context, execution=execution,
)
if checked.passed:
    plan = build_apply_plan(baseline, candidate, checked, execution=execution)
    result = runtime.apply(plan)
    baseline = plan.to_dict()["candidate"]
```

**先验证，后暂停。** `execution` 来自运行实例，绑定 run、segment、nonce、当前配置、运行时 revision、实际 AReaL 源码摘要和
adapter。验证结果及完整配置对冻结在 `ApplyPlan`
中；后续修改候选不会改变计划。旧实例发生重启后必须重新验证，不能复用旧计划。`context.state_id` 标识验证依据，实际保存边界使用另一个 snapshot
state ID；当前映射对所有成功 drain 的步末边界成立，无需先暂停才能做合法性判断。

配置不变返回 `unchanged`，不发送暂停请求。配置改变时，`plans/<plan_id>/`
保存私有计划与目标配置，随后请求保存、确认整组退出，再由相同入口启动新实例。`run_areal()` 在创建 Trainer 前加载目标配置，并用 AReaL
的结构化配置解析器确认其与被验证内容一致；自定义入口须先调用 `load_managed_configuration()` 再使用可调参数构造组件，才能声明
`supports_config_apply=True`。固定任务的数据／workflow 构造仍由可信入口负责。

原生 `ConfigurationTransition` 保留训练参数、optimizer、scheduler、数据游标、队列、计数、task IDs 和 policy
version；重新创建推理实例并同步保存的权重。同布局恢复保留各 rank 的随机状态；改变 FSDP 布局时，由原生 DCP 重分片参数与
optimizer，恢复共同的调度器进度和 LR，并在最后安装新随机流。并发变化只映射 `staleness.limits[0]`，不清空积压队列，也不重算 accepted
计数。恢复回执同时匹配 plan、目标配置、来源快照和源码摘要后记录 `apply.confirmed`。此确认表示恢复完成并开放训练入口；它不代表已完成新的
optimizer step 或 GPU 正确性验收。`ApplyResult.elapsed_seconds` 是总墙钟时间；尚无单独观测的 exposed delay 保持
`None`。

FSDP 布局切换的随机策略为 `fsdp-mesh-reseed-v1`，随源／目标布局写入 `plans/<plan_id>/plan.json`。种子由计划、来源快照
ID 和清单摘要、目标配置、角色及逻辑坐标经 SHA-256 派生；来源快照保存在目标实例的 `segments/<segment_id>/launch.json`
中。同一计划和快照生成相同的初始随机流，不要求与未切换运行相同，也不承诺 GPU 运算逐位复现。TP 组内共享 tensor seed，不同 DP/CP
坐标使用独立流；Python、NumPy 和 Torch CPU 使用各逻辑 rank 的 host seed。后续同布局重启保留这些流已经推进的位置及 DTensor
tracker 初始化状态。目标 VLM+TP 的 AdamW `fused` 设置也会在 DCP 后恢复，避免被源参数组覆盖。

14 个可调路径均纳入 adapter 判断。microbatch／padding、rollout 并发、推理布局、SGLang／vLLM 参数和 FSDP 的
DP/TP/CP 变化可走上述映射，并继续受完整资源、模型、后端支持和内存依据约束。原生 FSDP 不提供 PP／EP 执行。Megatron 跨布局使用新增的
`areal-megatron-fully-reshardable-v1` 格式，并要求下面的全局参数与 optimizer 语义依据。Megatron LoRA 的原生
DCP 保存限制、XCCL 的 pipeline 兼容条件、AWEX GPU 关系及 LocalScheduler worker 共置约束也在 Validator 中检查。共享
SGLang／vLLM 配置影响 teacher 时，还检查 `auxiliary.teacher` 中的独立模型、后端布局与服务器显存比例；缺少依据为
`undetermined`，不兼容则拒绝。Teacher 设备和内存也必须计入 `resources` 与 `memory`。

Megatron 的新格式使用锁定的 MCore 0.17.0 `fully_reshardable`：模型及 optimizer 张量保留全局参数键，移除 optimizer
chain 索引前缀；保存时通过 Gloo 聚合至各 DP 组的 rank 0，内存依据须覆盖这部分 host staging
和目标加载峰值。公共状态另存为全局参数组表，目标按自身组和 chain 恢复超参数、optimizer step 与 scaler。新随机策略
`megatron-model-parallel-reseed-v1` 从同一组计划／快照标识派生 base seed，按 PP rank 区分，TP／EP／ETP 流由
MCore 原生 seeding 建立；原生 checkpoint 的旧 RNG 不参与新格式加载，最后由 supplemental state 恢复或重建。普通
Recovery 的 `dcp` 行为保持不变；旧快照只支持同布局恢复，下一次 Reconfiguration 保存使用新格式。

Megatron 布局变化时，可信 harness 上下文还需提供 `training_checkpoint`：

- `source_ref`：模型 adapter／checkpoint 兼容依据。
- `format`：`areal-megatron-fully-reshardable-v1`。
- `model_schema.source/target`：全局参数 key、shape、dtype、实际 tied/alias 关系及 extra-state 的语义标识。
- `optimizer_schema.source/target`：optimizer 类型、master/moment dtype、参数组超参数与 scaler
  的语义标识。

两端语义标识必须一致，排除允许变化的 rank 所有权、局部 shard、chain 索引及 expert
分组位置。依据由可信模型／后端适配资料提供，不能为了通过检查填写任意相同字符串；当前不自动生成这些资料。缺失返回 `undetermined`，不兼容则拒绝。PP／EP／ETP
可随 `actor.backend` 变化，但仍须符合模型支持配置；VPP 设置自身不在可调白名单内。无 bridge 的 fallback 不放行 PP，Megatron
custom FSDP 需要另一种 checkpoint 格式。CPU 测试验证格式与状态映射接口，以及切换后的再次重启；尚未验证真实 Megatron 张量重分片。

本轮接口依据为 AReaL 提交 `7599750a260aac07281d2c01694e70e045bf8ec8`，其中已包含
`ConfigurationTransition` 和布局映射。运行这些样例需使用包含该提交的源码；Harness
迁入后从同一仓库读取训练运行时。重启前后还必须保持实际源码摘要一致。失败沿用同配置重启的保存／清理处理，并记录
`apply.failed`、阶段、配置标识和耗时；不把失败目标登记为当前配置，不自动回滚。

## 验证

Harness 测试验证模块接口与外部编排，在本目录运行：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

测试不需要 GPU、外部服务或 W&B SDK，覆盖 Monitor 的解析与落盘、Validator 的权限与约束、Monitor → 占位 Planner →
Validator 的接口衔接，以及实际本地子进程的保存／退出／恢复编排和失败路径。

可选集成测试复用上层 AReaL 仓库 `tests/` 中的 CPU fixtures，执行生产 `PPOTrainer` 的训练／清理方法和恢复接口，核对
actor、critic 恢复后续步，以及新配置生效后再次重启的配置与进度连续性。测试环境有 `pytest` 时自动运行，否则跳过这些测试；`pytest` 不是
Harness 的运行依赖。

AReaL 原生保存、恢复、状态映射与 checkpoint 格式测试统一放在上层 `tests/`，可从本目录运行：

```bash
python3 -m pytest ../tests/test_reconfiguration*.py
```

其中 [test_reconfiguration_transition.py](../tests/test_reconfiguration_transition.py)
覆盖配置与训练布局的状态映射，[test_reconfiguration_megatron.py](../tests/test_reconfiguration_megatron.py)
覆盖 Megatron portable optimizer、格式路由和随机流恢复。Harness 的 `tests/test_apply.py`
保留配置切换后再次重启等进程级测试。 以上均不验证真实 GPU 后端、DCP 张量重分片或 CUDA collectives。

接口依据：[W&B Run API 源码](https://github.com/wandb/wandb/blob/main/wandb/apis/public/runs.py)、[Prometheus HTTP API](https://prometheus.io/docs/prometheus/latest/querying/api/)。
