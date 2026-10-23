# Slurm＋Singularity：单机八卡任务工作流

这是与[本地Docker方式](RUN_2X2_8GPU.md)并列的执行方式。默认仍为Docker；显式设置 `AREAL_TASK_RUNTIME=singularity`
才使用本页后端。覆盖Hermes／CC × DAPO／SWE-rebench的评测与常驻PPO，算法、题单与评分规则沿用现有实现。

## 当前验证与启动边界

公司端用户已验证：宿主`singularity`为Apptainer1.3.2-1.el7，`srun --mpi=none`能启动
`/storage/openpsi/images/areal-dev-sglang-1014.sif`
并运行Python3.12.3。仅为CPU验证；训练镜像内部没有容器命令。

```text
同一Slurm作业
├─ 宿主Python工具服务 → singularity → 每题独立SIF＋可写overlay
└─ 训练SIF：Trainer／Agent／客户端 → 私有Unix socket → 宿主工具服务
```

宿主服务仅依赖Python≥3.10的标准库；不导入torch／Hermes／CC，也不需要推理密钥。Agent／客户端内部不执行容器命令。
SWE评分仍由训练镜像内的固定上游evaluator解析、判分，测试命令在宿主服务创建的全新评分overlay中运行；评分补丁不进入Agent环境。

## 1．准备训练与任务镜像

- 训练SIF或挂载的Python虚拟环境须装齐[原2×2文档](RUN_2X2_8GPU.md)中的依赖，含Hermes0.19.0、CC
  SDK0.1.71／CLI2.1.123与固定SWE evaluator。现有公司SIF仅证明能运行Python，依赖需另查；不要在只读SIF内直接安装。
- 代码、模型、冻结题单、结果、虚拟环境和evaluator置于同一共享父目录，按原文档准备同一64K／非thinking模型与数据；本方式不重新抽题。下面路径均使用实际绝对路径，不含空格、逗号或冒号。
- 宿主`singularity`必须支持`overlay create --fakeroot`、`exec --fakeroot --containall --net --network none`及禁用默认bind。这些尚未在公司验证；不支持时探针报错，无宿主执行或开放网络回退。
- 每题使用默认4GiB的sparse
  overlay，工具与新评分环境各自独立，结束后删除overlay／scratch，保留日志和patch。overlay容量不是RAM限额；可用`TASK_OVERLAY_MB`调整，见[Apptainer1.3说明](https://apptainer.org/docs/user/1.3/persistent_overlays.html)。
- 网络关闭、私有PID／IPC、最小`/dev`、干净环境且不传`--nv`；只读评分patch是唯一额外bind。共享存储／home／socket不挂进任务容器。实际挂载与fakeroot隔离仍需公司验收。
- CPU／RAM总额依赖公司Slurm的作业cgroup配置，需确认实际生效；`TASK_TOOL_WORKERS`限制同时执行的工具／评分数量。当前不提供Docker式每题CPU／RAM
  cgroup；每次调用有时间／输出上限，结束时清理自身进程组。

在允许准备镜像的机器先把Python镜像与冻结SWE题单的每个`image_name`转换为SIF，例如：

```bash
singularity pull /absolute/images/python.sif docker://python:3.12
singularity pull /absolute/images/swe-task.sif docker://registry/repository@sha256:REAL_DIGEST
```

每个别名必须对应实际输入中的精确镜像名。生成`images.json`；正式实验保存原registry
digest和转换后SIF的SHA256，在H20／H200两端复用同一份SIF：

```bash
python3 - <<'PY'
import hashlib, json
from pathlib import Path
mapping = {
    "python:3.12": "/absolute/images/python.sif",
    "registry/repository@sha256:REAL_DIGEST": "/absolute/images/swe-task.sif",
}
manifest = {}
for alias, value in mapping.items():
    path = Path(value).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    manifest[alias] = {"sif": str(path), "sha256": digest.hexdigest()}
Path("images.json").write_text(json.dumps(manifest, indent=2))
PY
```

服务启动时校验SIF摘要；`image_id`记录SIF SHA256，而不是伪装成Docker本地镜像ID。Agent和评分使用同一已解析身份。

## 2．先验收CPU工具路径

从公司可提交作业的终端配置以下变量。脚本在`TASK_RUN_ROOT`创建新目录，不覆盖已有运行：

```bash
export TASK_CODE_ROOT=/absolute/path/to/AReaL
export TASK_MOUNT_ROOT=/storage
export TASK_TRAIN_SIF=/storage/openpsi/images/areal-dev-sglang-1014.sif
export TASK_IMAGE_MANIFEST=/absolute/path/to/images.json
export TASK_HOST_PYTHON=python3  # 须为宿主Python≥3.10，也可指定绝对路径
export TASK_PYTHON=python3  # CPU工具探针只需镜像中的Python
export TASK_RUN_ROOT=/absolute/results/sif-python-probe-1
export AREAL_TASK_PYTHON_IMAGE=python:3.12
sbatch --gres=none --cpus-per-task=8 --mem=16G --time=00:10:00 \
  --export=ALL,TASK_MODE=probe "$TASK_CODE_ROOT/examples/task_workflows/slurm_singularity.sbatch"
```

`python-tool-probe.json`应为completed、输出42；`host-tools.log`无容器启动异常。该探针同时检查overlay／fakeroot／网络namespace能否启动，比之前单纯运行Python的探针多了一层工具隔离，但不验证GPU或每题资源限额。

## 3．选择组合并评测／训练

每次一个新作业、新目录，沿用上面的镜像／路径配置。设置私有训练密钥环境变量，勿在命令行打印环境或启用`set -x`。保存的`launch.json`只包含任务配置，不包含密钥。

```bash
export AREAL_INFERENCE_ADMIN_KEY=YOUR_PRIVATE_KEY
export AREAL_AGENT_ADMIN_KEY=YOUR_PRIVATE_KEY
export TASK_HW=h200  # H20用h20；标签不自动调整训练参数
export TASK_PYTHON=/absolute/shared/venv/bin/python  # 已装齐训练／Agent依赖
export TASK_MODEL_PATH=/absolute/models/qwen3-17b-64k
export AREAL_SWE_EVALUATOR_ROOT=/absolute/pinned/swe-rebench-v2
export TASK_CASE=hermes-dapo  # 或cc-dapo、hermes-swe、cc-swe
export TASK_TRAIN_TASKS=/absolute/frozen/dapo/train.jsonl
export TASK_EVAL_TASKS=/absolute/frozen/dapo/eval.jsonl
export TASK_N=16 TASK_EPOCHS=1 TASK_BATCH=4 TASK_CONCURRENCY=4
export TASK_RUN_ROOT=/absolute/results/hermes-dapo-sif-eval-1
sbatch --export=ALL,TASK_MODE=eval "$TASK_CODE_ROOT/examples/task_workflows/slurm_singularity.sbatch"
```

SWE组合换成相应的SWE训练／评测题单，确认manifest覆盖全部镜像。正式评测前换新目录，提交CPU评分验收作业（题单宜先冻结为少量任务）：

```bash
export TASK_RUN_ROOT=/absolute/results/swe-sif-reference-1
sbatch --gres=none --cpus-per-task=8 --mem=16G \
  --export=ALL,TASK_MODE=reference "$TASK_CODE_ROOT/examples/task_workflows/slurm_singularity.sbatch"
```

检查`reference-grades.log`中每题参考patch的reward为1；任何基础设施异常均使作业失败。

评测使用8个TP1 Rollout，训练使用4卡Actor＋4个TP1 Rollout。评测通过后换新目录提交训练冒烟：

```bash
export TASK_RUN_ROOT=/absolute/results/hermes-dapo-sif-train-smoke-1
sbatch --export=ALL,TASK_MODE=train "$TASK_CODE_ROOT/examples/task_workflows/slurm_singularity.sbatch"
```

四种组合均先验收16题／4更新；检查每轮summary、逐调用计时、实际Train step数、HF导出、客户端／Trainer退出码。不能只看Slurm退出0或客户端完成。
本适配不构成原1.7B长序列OOM的修复，H20／H200均须先验证失败规模和64K完整前后向，再安排正式训练。

已有固定1,000题的5轮设置仍可显式选择；以下仅在显存验收后使用：

```bash
export TASK_N=1000 TASK_EPOCHS=5 TASK_BATCH=40 TASK_CONCURRENCY=32 TASK_SAVE_EVERY=25
export TASK_RUN_ROOT=/absolute/results/hermes-dapo-sif-train-1000x5-1
sbatch --export=ALL,TASK_MODE=train "$TASK_CODE_ROOT/examples/task_workflows/slurm_singularity.sbatch"
```

默认SBATCH资源只是模板：partition／account／GPU类型、CPU／RAM和时限须按公司现成配置覆盖。`srun --mpi=none`只影响Slurm进程启动，不关闭PyTorch的NCCL通信。
Agent、推理、客户端和宿主工具服务都属于同一作业；脚本只停止本次服务名和记录PID，作业结束由Slurm清理其余作业进程。

## 记录与验证边界

沿用现有episodes／simulation_trace、Rollout batch与Trainer
trace。工具`queued`在调用入口，`started/seconds`采用宿主实际执行时间，排除RPC和工具槽位等待；日志中的宿主pool容量不是现有Expert逐题pool模型，模拟全局CPU并发仍需额外组装。

本地33项正常路径通过，覆盖Docker回归和新增Singularity的Hermes／CC × 数学／SWE，真实Unix
RPC／git／pytest／上游评分器参与，Apptainer CLI和模型响应为替身。
真实SIF、隔离、GPU／NCCL及训练均待公司端验收；详见[证据](../../../artifacts/task-workflows-singularity-20261023/offline-validation.json)。
