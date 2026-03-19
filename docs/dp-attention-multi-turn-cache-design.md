# 开源 AReaL DP Attention + Multi-Turn Cache 设计方案（v6）

> v6 更新：rid 格式重新加入 `global_step`，改为 `{qid}-{step}-{dup_cnt}`。
>
> - `fill_rid_base` 新增 `global_step` 参数（默认 0）
> - `dup_cnt` 的 counter key 改为 `f"{qid}@{global_step}"`，每个 step 独立计数
> - `RolloutController` 传入 `global_step=self.get_version()`
> - 加入 step 的好处：(1) 明确标识 rid 属于哪个训练版本 (2) dup_cnt 按 step 分区，避免跨 epoch 累积
>
> v5 更新：实现完成，更新代码示例以匹配实际实现。关键变更：
>
> - `compute_dp_rank` 改用 `hashlib.sha256` 替代 `hash()` 以保证跨进程确定性
> - `_qid_counter` 添加 `threading.Lock` 保护线程安全
> - `rid_queue` 改用 `collections.deque` (O(1) popleft)
> - 重复的 rid 构建代码提取为 `_build_rid_and_dp_rank()` 共享函数
> - `sglang_dp_size` 配置放在 `OpenAIProxyConfig` 中
> - 方法名 `parse_rid_base` 实现为 `parse_routing_key`，`get_data_qid` 实现为 `get_rid_base`
>
> v4 更新：简化 rid 格式，去掉 global_step，改用 per-qid 全局计数器。 v3 更新：修正 async training abort
> 机制理解，补充监控设计。 v2 更新：基于 SGLang 代码分析，发现 SGLang 已原生支持 `data_parallel_rank` 字段用于 DP rank
> 指定路由。 **无需修改 SGLang 代码**，仅需 AReaL 侧改动。

______________________________________________________________________

## 1. 目标

在开源 AReaL 中实现：

1. **DP Attention 下多轮对话 prefix cache 命中** — 同一 episode 的所有 turn 路由到相同 DP rank
1. **Async Training 支持** — SWE RL agent workflow 在 async 模式下正确运行
1. **零 SGLang 侧修改** — 完全利用 SGLang 已有的 `data_parallel_rank` API

______________________________________________________________________

## 2. SGLang 已有能力（关键发现）

### 2.1 `data_parallel_rank` 字段

SGLang **所有 API** 都已支持客户端指定 DP rank：

```python
# SGLang data_parallel_controller.py:506-511
def maybe_external_dp_rank_routing(self, req: Req):
    if req.data_parallel_rank is not None:
        self.workers[req.data_parallel_rank].send_pyobj(req)
        return True
    return False
```

每个 DP scheduling method（round_robin、total_requests、total_tokens）都**优先检查**此字段。只要请求中传入
`data_parallel_rank=N`，SGLang 就会直接路由到 DP rank N，跳过所有 load-balance 策略。

**支持的 API 端点**：

- `/generate`（AReaL 当前使用的端点）
- `/v1/chat/completions`
- `/v1/completions`
- Engine API（`async_generate()`）

### 2.2 `rid` 字段

SGLang 的 `/generate` 端点也接受 `rid` 字段（`BaseReq.rid`），但 `rid` 仅用于请求追踪（response 中返回），**不影响
DP 路由**。AReaL 可以同时传递 `rid`（用于追踪）和 `data_parallel_rank`（用于路由）。

### 2.3 DP Attention 架构

SGLang DP Attention 模式下：

- 单一 NCCL group，TP 维度被拆分为 `dp_size × attn_tp_size`
- 每个 DP rank 有**独立的 radix cache**（互相隔离）
- MLP/MoE 层通过 all-gather/all-reduce 跨 DP rank 协同
- 单一进程入口 → `DataParallelController` 分发到各 DP rank

### 2.4 两层 DP 独立

| 概念                    | 含义                             | 控制范围                        |
| ----------------------- | -------------------------------- | ------------------------------- |
| AReaL allocation `d`    | 独立 SGLang server 进程数        | `sglang:d4t4` → 4 个独立 server |
| SGLang `sglang.dp_size` | 单个 server 内部 DP attention 数 | 每个 server 内部的 DP rank 数   |

两者可以组合：`d2t8` + `sglang.dp_size=4` = 2 个 server，每个 server 内部 4 个 DP rank。

______________________________________________________________________

## 3. 问题分析

两层路由，两层亲和性：

```
┌────────────────────────────────────┐
│     层级 1: AReaL → SGLang Server  │  多 server 场景
│     rid_to_address (已有机制)       │  需要: rid_base 作为 key
├────────────────────────────────────┤
│     层级 2: SGLang → DP Rank       │  单 server 内部 DP attention
│     data_parallel_rank (SGLang 原生)│  需要: AReaL 计算并传入 rank
└────────────────────────────────────┘
```

**当前现状**：

- 层级 1：`rid_to_address` 已有，但 rid 是随机 UUID → 多轮无法命中 → **需要确定性 rid**
- 层级 2：SGLang 已支持 `data_parallel_rank`，但 AReaL 从未传递 → **需要传入**

______________________________________________________________________

## 4. 总体架构

```
┌──────────────────────────────────────────────────────────────────┐
│                      RolloutController                            │
│  1. fill_rid_base(item) → rid_base                               │
│  2. _choose_worker() → worker (round-robin，不变)                 │
│  3. dispatcher → submit to worker                                │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│            OpenAIProxyWorkflow.arun_episode()                      │
│  → 提取 rid_base，传给 session                                     │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│     AsyncCompletionsWithReward.create()                            │
│  → rid = build_rid(rid_base, sample_idx, round_idx)               │
│  → dp_rank = sha256(parse_routing_key(rid)) % sglang_dp_size     │
│  → ModelRequest(rid=rid, data_parallel_rank=dp_rank)              │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│            RemoteInfEngine.agenerate()                             │
│  → rid_base 查 rid_to_address → 选 SGLang server (层级 1)         │
│  → SGLangBackend.build_generation_request():                      │
│    payload["data_parallel_rank"] = req.data_parallel_rank         │
│    payload["rid"] = req.rid                                       │
│  → POST /generate                                                │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│               SGLang Server (DP Attention 模式)                    │
│  → DataParallelController.maybe_external_dp_rank_routing()        │
│    req.data_parallel_rank != None → 直接路由到指定 DP rank          │
│  → 该 DP rank 的 radix cache 命中 prefix ✅                       │
│  ⚠️ 零修改：利用 SGLang 已有的 data_parallel_rank API              │
└──────────────────────────────────────────────────────────────────┘
```

______________________________________________________________________

## 5. 详细设计

### 5.1 组件一：RolloutIdBuilder

**新增文件**：`areal/utils/rollout_id.py`

从内源移植，纯工具类，无外部依赖。

```python
import hashlib
import threading
from collections import defaultdict
from typing import ClassVar

class RolloutIdBuilder:
    """构造确定性 rid，确保多轮对话路由到相同 DP rank。

    rid 格式: {qid}-{step}-{dup_cnt}-r-{round_idx}_{sample_idx}
    rid_base: {qid}-{step}-{dup_cnt}
    routing_key (parse_routing_key): {qid}-{step}-{dup_cnt}_{sample_idx}
    """

    EXT_QID_FIELD = "_ext_query_id"
    EXT_QID_IDX_FIELD = "_ext_query_id_idx"

    # Per-(qid, step) dup counter. Resets implicitly when a new step appears.
    _qid_counter: ClassVar[defaultdict[str, int]] = defaultdict(int)
    _counter_lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def fill_rid_base(cls, data_item: dict, global_step: int = 0) -> None:
        """为数据项分配确定性 rid_base。"""
        qid = None
        for key in ("query_id", "qid", "id", "instance_id"):
            if key in data_item:
                qid = str(data_item[key])
                break
        if qid is None:
            content = str(data_item.get("prompt", data_item.get("question", "")))
            qid = hashlib.sha256(content.encode()).hexdigest()[:16]

        # Counter key includes step so dup_cnt resets per step.
        counter_key = f"{qid}@{global_step}"
        with cls._counter_lock:
            dup_cnt = cls._qid_counter[counter_key]
            cls._qid_counter[counter_key] += 1
        data_item[cls.EXT_QID_FIELD] = [f"{qid}-{global_step}-{dup_cnt}"]

    @classmethod
    def get_rid_base(cls, data_item: dict) -> str:
        """获取 data_item 的 rid_base，无则 fallback 到 UUID。"""
        ext = data_item.get(cls.EXT_QID_FIELD)
        if ext and len(ext) > 0:
            return ext[0]
        return str(uuid.uuid4())

    @classmethod
    def build_rid(cls, rid_base: str, sample_idx: int,
                  round_idx: int | None = None) -> str:
        if round_idx is not None:
            return f"{rid_base}-r-{round_idx}_{sample_idx}"
        return f"{rid_base}_{sample_idx}"

    @classmethod
    def infer_round_idx(cls, messages: list[dict]) -> int:
        return sum(1 for m in messages if m.get("role") == "assistant")

    @classmethod
    def parse_routing_key(cls, rid: str) -> str:
        """提取 routing key（去掉 round 部分，保留 sample_idx）。

        "django-123-5-0-r-2_0" → "django-123-5-0_0"
        "django-123-5-0_0"     → "django-123-5-0_0"（无 round，原样返回）
        rsplit("-r-", 1) 从右侧分割，正确处理 qid 本身含 "-r-" 的情况。
        """
        parts = rid.rsplit("-r-", 1)
        if len(parts) == 2:
            round_and_rest = parts[1]
            idx = round_and_rest.find("_")
            if idx >= 0:
                return f"{parts[0]}{round_and_rest[idx:]}"
            return parts[0]
        return rid

    @classmethod
    def compute_dp_rank(cls, rid: str, dp_size: int) -> int:
        """从 rid 计算确定性 DP rank。

        使用 hashlib.sha256（非 hash()）保证跨进程确定性。
        """
        if dp_size <= 1:
            return 0
        routing_key = cls.parse_routing_key(rid)
        h = int(hashlib.sha256(routing_key.encode()).hexdigest(), 16)
        return h % dp_size
```

### 5.2 组件二：ModelRequest 扩展

**修改文件**：`areal/api/io_struct.py`

```python
@dataclass
class ModelRequest:
    rid: str = field(default_factory=lambda: str(uuid.uuid4()))
    input_ids: list[int] = field(default_factory=list)
    gconfig: GenerationHyperparameters = field(default_factory=GenerationHyperparameters)
    # 新增：指定 SGLang 内部 DP rank
    data_parallel_rank: int | None = None
    ...
```

### 5.3 组件三：SGLangBackend 传递 DP rank

**修改文件**：`areal/engine/sglang_remote.py`

```python
class SGLangBackend:
    def build_generation_request(self, req: ModelRequest, ...) -> HttpRequest:
        payload = {
            "input_ids": req.input_ids.copy(),
            "sampling_params": sample_params,
            "return_logprob": True,
            "stream": False,
            # 新增：传递 rid 和 dp_rank 到 SGLang
            "rid": req.rid,
        }
        # 新增：如果指定了 dp_rank，传给 SGLang
        if req.data_parallel_rank is not None:
            payload["data_parallel_rank"] = req.data_parallel_rank

        return HttpRequest(endpoint="/generate", payload=payload)
```

### 5.4 组件四：rid 传递链路

#### 5.4.1 RolloutController.prepare_batch()

```python
# areal/infra/controller/rollout_controller.py

def _task_input_generator(self):
    for item in self.dataloader:
        # 新增：分配确定性 rid_base，传入当前训练 step
        RolloutIdBuilder.fill_rid_base(item, global_step=self.get_version())
        yield _RemoteRolloutTaskInput(data=item, ...)
```

#### 5.4.2 InteractionCache 添加 rid_base/sample_idx

```python
# areal/experimental/openai/cache.py

class InteractionCache(OrderedDict[str, InteractionWithTokenLogpReward]):
    def __init__(self, *args, rid_base: str | None = None,
                 sample_idx: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rid_base = rid_base
        self.sample_idx = sample_idx
        self._apply_reward_discount_called = False
        self._total_reward = 0.0
        self._lock = threading.Lock()
```

#### 5.4.3 OpenAIProxyWorkflow.arun_episode()

```python
# areal/experimental/openai/proxy/workflow.py

async def arun_episode(self, engine, data):
    # 仅在 fill_rid_base 已被调用时提取 rid_base
    rid_base = (
        RolloutIdBuilder.get_rid_base(data)
        if RolloutIdBuilder.EXT_QID_FIELD in data
        else None
    )

    proxy_client = OpenAIProxyClient(
        ...,
        rid_base=rid_base,
        sample_idx=data.get(RolloutIdBuilder.EXT_QID_IDX_FIELD, 0),
    )
    async with proxy_client:
        result = await self._run_agent(data, ...)
```

#### 5.4.4 共享 rid 构建函数 + AsyncCompletionsWithReward.create()

```python
# areal/experimental/openai/client.py

# 共享函数（避免 Completions 和 Responses 重复代码）
def _build_rid_and_dp_rank(
    cache: InteractionCache,
    messages_list: list[dict],
    sglang_dp_size: int,
) -> tuple[str, int | None]:
    rid_base = cache.rid_base
    sample_idx = cache.sample_idx
    if rid_base is not None:
        round_idx = RolloutIdBuilder.infer_round_idx(messages_list)
        rid = RolloutIdBuilder.build_rid(rid_base, sample_idx, round_idx)
        dp_rank = RolloutIdBuilder.compute_dp_rank(rid, sglang_dp_size)
        return rid, dp_rank
    return str(uuid.uuid4()), None


# 在 AsyncCompletionsWithReward.create() 和 AsyncResponsesWithReward.create() 中：
async def create(self, ...):
    ...
    cache_for_rid = areal_cache if areal_cache is not None else self._cache
    rid, dp_rank = _build_rid_and_dp_rank(
        cache_for_rid, messages_list, self.sglang_dp_size
    )

    model_request = ModelRequest(
        input_ids=prompt_token_ids,
        gconfig=gconfig,
        rid=rid,
        data_parallel_rank=dp_rank,
    )
```

`sglang_dp_size` 通过 `ArealOpenAI.__init__()` 传入，配置在 `OpenAIProxyConfig` 中。

### 5.5 组件五：rid_to_address 改用 rid_base

**修改文件**：`areal/infra/remote_inf_engine.py`

```python
# __init__ 中：
self.rid_to_address: dict[str, str] = {}
self.rid_queue: deque[str] = deque()  # O(1) popleft，替代 list.pop(0)

async def agenerate(self, req: ModelRequest) -> ModelResponse:
    ...
    # 改用 routing_key（parse_routing_key 去掉 round 部分）作为 server 亲和性 key
    routing_key = RolloutIdBuilder.parse_routing_key(req.rid)

    if routing_key in self.rid_to_address:
        server_addr = self.rid_to_address[routing_key]
    else:
        server_addr = self.choose_server()
        if len(self.rid_queue) >= RID_CACHE_SIZE:
            oldest_key = self.rid_queue.popleft()
            self.rid_to_address.pop(oldest_key, None)
        self.rid_to_address[routing_key] = server_addr
        self.rid_queue.append(routing_key)
    ...
```

### 5.6 组件六：Radix Cache 配置

```python
# areal/api/cli_args.py - SGLangConfig

# 当 enable_dp_attention=True 时，自动开启 radix cache
@property
def effective_disable_radix_cache(self) -> bool:
    if self.enable_dp_attention:
        return False  # DP attention 模式必须开启 radix cache
    return self.disable_radix_cache
```

或者更简单：在文档和配置示例中明确说明。

______________________________________________________________________

## 6. 数据流全链路

### 6.1 单次 Episode 的完整数据流

```
1. Dataloader 产出数据项:
   {"instance_id": "django-123", "problem_statement": "..."}

2. RolloutController.prepare_batch():
   RolloutIdBuilder.fill_rid_base(item, global_step=self.get_version())
   → item["_ext_query_id"] = ["django-123-5-0"]   # step=5, dup_cnt=0

3. Dispatcher → worker, 执行 workflow:
   OpenAIProxyWorkflow.arun_episode(engine, data)
   → rid_base = "django-123-5-0", sample_idx = 0

4. Agent Turn 1 调用 LLM:
   client.chat.completions.create(messages=[system, user])
   → round_idx = 0
   → rid = "django-123-5-0-r-0_0"
   → routing_key = parse_routing_key(rid) = "django-123-5-0_0"
   → dp_rank = sha256("django-123-5-0_0") % 4
   → ModelRequest(rid=rid, data_parallel_rank=2)

5. RemoteInfEngine.agenerate():
   → routing_key = "django-123-5-0_0"
   → rid_to_address 未命中 → choose_server() → "http://sglang-0:30000"
   → 缓存映射

6. SGLangBackend.build_generation_request():
   payload = {
       "input_ids": [...],
       "rid": "django-123-5-0-r-0_0",
       "data_parallel_rank": 2,         ← 关键：指定 DP rank
       "sampling_params": {...},
       "return_logprob": True,
   }
   → POST /generate

7. SGLang DataParallelController:
   → maybe_external_dp_rank_routing(req)
   → req.data_parallel_rank = 2 → 直接路由到 DP rank 2
   → DP rank 2 处理请求，KV cache 存储在 rank 2 的 radix tree

8. Agent Turn 2 调用 LLM:
   messages = [system, user, assistant_1, tool_result_1, user_2]
   → round_idx = 1
   → rid = "django-123-5-0-r-1_0"
   → routing_key = "django-123-5-0_0" (与 Turn 1 相同!)
   → dp_rank = sha256("django-123-5-0_0") % 4 (与 Turn 1 相同!)

9. RemoteInfEngine.agenerate():
   → routing_key = "django-123-5-0_0"
   → rid_to_address 命中! → "http://sglang-0:30000" (同一 server)

10. SGLang:
    → data_parallel_rank = 2 → 同一 DP rank
    → Radix cache 命中 Turn 1 的 prefix ✅
    → 只需计算新增部分的 attention
```

### 6.2 Async Training 完整流程与 Abort 机制

#### 6.2.1 真实的 Async 训练循环

```
Async Training 核心循环：
═══════════════════════════════════════════════════════════════════

1. rollout.resume()                    # 开闸：允许大量 prompt 投放
2. batch = rollout.prepare_batch()     # 投放大量 prompt 到 SGLang
                                       # 收集 batch_size 个完成的 rollout
3. rollout.pause()                     # 停止新投放（但已投放的继续跑）
4. actor.train_batch(batch)            # 用收集到的 batch 训练

   ─── 到达 max_staleness ───

5. actor.update_weights(meta)          # 推送新权重到 SGLang
   └→ POST /update_weights_from_distributed
      { ..., "abort_all_requests": True }    ← 关键：abort 正在推理的请求
   └→ SGLang 中断所有 in-flight 请求
   └→ 清空 radix cache
   └→ 加载新权重

6. rollout.set_version(new_version)    # 更新版本号
7. 回到步骤 1

═══════════════════════════════════════════════════════════════════
```

#### 6.2.2 Abort 对多轮 Episode 的影响

```
Timeline:
─────────────────────────────────────────────────────────────────────

rollout.resume() → 大量 prompt 投放
                                                           train 消费
  Episode A: [T1→rank2]──[T2→rank2]──[T3→rank2] → done ──→ batch
  Episode B: [T1→rank0]──[T2→rank0]──────────────→ done ──→ batch
  Episode C: [T1→rank1]──[T2→rank1]──────────...   (进行中)
  Episode D: [T1→rank3]──...                        (进行中)
  Episode E: [T1→rank2]──[T2→rank2]──...            (进行中)
                                                  ↓
                                     batch_size 个 rollout 完成
                                     rollout.pause() → 停止新投放
                                                  ↓
                                     max_staleness 达到 → 更新权重
                                                  ↓
━━━━━━━━━━━━━━━━━ WEIGHT UPDATE + ABORT ALL ━━━━━━━━━━━━━━━━━━━━━

  SGLang 收到 update_weights(abort_all_requests=True)：
  1. Episode C Turn2 被 abort → 返回 stop_reason="abort"
  2. Episode D Turn1 被 abort → 返回 stop_reason="abort"
  3. Episode E Turn2 被 abort → 返回 stop_reason="abort"
  4. Radix cache 全部清空
  5. 加载新权重 v_new

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  RemoteInfEngine.agenerate() 的 abort 重试循环：

  Episode C:
    → agenerate() 收到 abort → while loop 继续
    → while is_paused(): sleep  (等待权重更新完成)
    → 重新发送请求（input_ids = 原始 + 已生成的 token）
    → data_parallel_rank = 2 → 同一 DP rank ✅
    → 但 radix cache 已清空 → cache miss ❌（必须重新计算）
    → Turn2 完成 → [T3→rank1] → 新的 cache 开始积累
    → Turn3 可以命中 Turn2 的 cache ✅

  Episode E:
    → 同理：abort → 等待 → 重试 → 同一 rank → cache miss → 继续
─────────────────────────────────────────────────────────────────────
```

#### 6.2.3 agenerate() 的 Abort 重试循环（关键代码）

```python
# RemoteInfEngine.agenerate() 核心逻辑
original_max_new_tokens = gconfig.max_new_tokens

while (
    stop_reason not in ["stop", "tool_calls", "length"]
    and len(accumulated_output_tokens) < original_max_new_tokens
):
    # 1. 等待权重更新完成
    while self.workflow_executor.is_paused():
        await asyncio.sleep(0.5)

    # 2. 构建并发送请求
    #    → payload 中包含 data_parallel_rank → 路由到同一 DP rank
    http_req = self.backend.build_generation_request(req, ...)
    result = await arequest_with_retry(session, server_addr, ...)

    # 3. 解析响应
    gen_result = self.backend.parse_generation_response(result)
    stop_reason = gen_result.stop_reason  # 可能是 "abort"

    # 4. 累积 token（包括 abort 前已生成的）
    accumulated_output_tokens.extend(gen_result.output_tokens)
    accumulated_versions.extend([self.get_version()] * len(gen_result.output_tokens))

    # 5. 更新请求（追加已生成 token，减少剩余配额）
    req.input_ids += gen_result.output_tokens
    req.gconfig.max_new_tokens -= len(gen_result.output_tokens)

    # 6. 如果是 abort → loop 继续 → 回到步骤 1 等待 unpause → 重新发送
```

**关键行为**：

- abort 后**不丢弃**已生成的 token，追加到 input_ids 中继续
- 重试请求带着更长的 input_ids 和减少的 max_new_tokens
- `data_parallel_rank` 不变 → 路由到同一 DP rank
- 但 radix cache 已被 weight update 清空 → 重试时 cache miss
- **StalenessManager 看不到 abort**：从它的视角，workflow task 仍在 running

#### 6.2.4 Abort 对 Cache 效率的量化影响

| 场景                                         | Cache 效果                                                                                           |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Episode 全部在同一 weight version 内完成     | ✅ 所有 turn 命中 prefix cache                                                                       |
| Episode 被 abort，abort 时 Turn N 已部分生成 | ⚠️ 重试 Turn N 时 cache miss（weight reload 清空了 cache）；但后续 Turn N+1 可以命中 Turn N 的 cache |
| Episode 跨越多次 weight update               | ❌ 每次 abort 后 cache 全清，收益降低                                                                |

**优化方向**：减少单个 episode 跨越 weight update 的概率 → 控制 `max_head_offpolicyness` 和 episode 时长

______________________________________________________________________

## 7. 与内源实现的对比

| 方面               | 内源（AReaL-asystem）         | 开源方案（v2）                                        |
| ------------------ | ----------------------------- | ----------------------------------------------------- |
| **推理引擎**       | HybridEngine（包装 SGLang）   | SGLang 直连                                           |
| **DP rank 路由**   | HybridEngine arouter 解析 rid | AReaL 计算 rank → `data_parallel_rank` 传给 SGLang    |
| **路由决策在哪层** | 引擎内部                      | AReaL 客户端（hash 计算） + SGLang（执行路由）        |
| **rid 传递**       | payload 中的 `"rid"` 字段     | payload 中的 `"rid"` + `"data_parallel_rank"`         |
| **Server 亲和性**  | rid_to_address（完整 rid）    | rid_to_address（routing_key via `parse_routing_key`） |
| **SGLang 改动**    | N/A（用 HybridEngine）        | **零改动**                                            |
| **rid 构造**       | RolloutIdBuilder              | RolloutIdBuilder（移植）                              |

**核心区别**：内源中 HybridEngine arouter 解析 rid 来决定 DP rank；开源中 AReaL 自己算好 DP rank，通过 SGLang
已有的 `data_parallel_rank` API 直接指定。效果等价，但不依赖 SGLang 改动。

______________________________________________________________________

## 8. rid 格式设计说明

### 8.1 为什么加入 global_step

v4 曾去掉 `global_step`，改用 per-qid 全局计数器。v6 重新加入 `global_step`：

| 好处             | 说明                                                       |
| ---------------- | ---------------------------------------------------------- |
| **版本标识**     | rid 明确标识属于哪个训练版本，便于调试和日志分析           |
| **dup_cnt 分区** | 每个 step 独立计数，避免跨 epoch 累积导致 dup_cnt 持续增长 |
| **与内源一致**   | 保持与内源 `{qid}-{step}-{dup_cnt}` 格式一致，降低维护成本 |

counter key 为 `f"{qid}@{global_step}"`，当 step 变化时 dup_cnt 自动从 0 开始。

### 8.2 最终 rid 格式

```
rid_base: {qid}-{step}-{dup_cnt}
rid:      {qid}-{step}-{dup_cnt}-r-{round_idx}_{sample_idx}
routing_key: {qid}-{step}-{dup_cnt}_{sample_idx}

示例:
  rid_base    = "django-123-5-0"
  rid         = "django-123-5-0-r-2_0"
  routing_key = "django-123-5-0_0"
```

- `step`: 训练 global_step（版本），由 `RolloutController.get_version()` 提供
- `dup_cnt`: per-(qid, step) 递增，处理同一 step 内 qid 重复（epoch wrap 等）
- `round_idx`: 多轮对话轮次，路由时被去掉，保证同 episode 所有 turn 路由一致
- `sample_idx`: GRPO 并行样本索引，不同 sample 可路由到不同 rank（并行度更高）

### 8.3 Recover 安全性

Recover 时 `_qid_counter` 从 0 重新开始，可能与 recover 前的 rid 重复。但 SGLang 在 recover 时也会重启，radix
cache 和内部状态全清，不存在 rid 碰撞风险。

______________________________________________________________________

## 9. 目标分支现状（`chucai.dzq/feat-swe-agent-ray`）

### 9.1 分支新增内容

该分支新增了 SWE Agent 训练的完整支持：

| 文件                                               | 说明                                                                            |
| -------------------------------------------------- | ------------------------------------------------------------------------------- |
| `examples/swe/agent.py`                            | `SWEAgentWorkflow` — OpenAI 兼容的 SWE Agent，通过 AReaL proxy 调用 SGLang      |
| `examples/swe/agent_cc.py`                         | `CCAgentWorkflow` — Claude Code Agent，通过 golang proxy → AReaL proxy → SGLang |
| `examples/swe/train.py`                            | SWE 训练入口，使用 `PPOTrainer.train()`                                         |
| `examples/swe/train_cc.py`                         | CC 训练入口                                                                     |
| `examples/swe/utils.py`                            | `SWEEnvConfig`/`CCEnvConfig`/`SWEPPOConfig`/`CCPPOConfig` 配置                  |
| `examples/swe/config_swe_ray_megatron.yaml`        | 配置示例：`sglang:d4t4`，individual export                                      |
| `examples/swe/config_swe_ray_megatron_concat.yaml` | 配置示例：`sglang:d8t4`，concat export + CP=4                                   |

**Agent 调用链路**（inline 模式）：

```
PPOTrainer.train(workflow="examples.swe.agent_cc.CCAgentWorkflow")
  → RolloutController.prepare_batch()
    → dispatcher → worker 执行 OpenAIProxyWorkflow.arun_episode()
      → _run_agent(session_api_key, data)
        → CCAgentWorkflow.run(data, base_url=proxy_addr, api_key=session_key)
          → AenvCC sandbox → CCAgent → Claude Code CLI
            → Anthropic API → golang proxy → AReaL proxy /v1/messages
              → 格式转换 → AsyncCompletionsWithReward.create()
                → ModelRequest(rid=uuid4()) → RemoteInfEngine.agenerate()
                  → SGLangBackend → POST /generate
```

**关键观察**：

- `CCAgentWorkflow.run()` 不直接调用 engine，由 AReaL proxy 拦截 API 请求
- `data` 中有 `instance_id` 字段（SWE-bench ID），可直接作为 qid
- 现有配置**未开启 DP attention**（`sglang.dp_size` 和 `enable_dp_attention` 都是默认值）
- concat 模式配置已存在（CP=4 + concat chat_template），适合长序列 SWE RL

### 9.2 DP Attention 相关组件现状

| 组件                                        | 现状                               | 需要做什么                        |
| ------------------------------------------- | ---------------------------------- | --------------------------------- |
| `areal/utils/rollout_id.py`                 | **不存在**                         | 新增 RolloutIdBuilder 类          |
| `ModelRequest.data_parallel_rank`           | **不存在**                         | 新增字段                          |
| `SGLangBackend.build_generation_request()`  | 不传 `rid` 和 `data_parallel_rank` | 新增两个 payload 字段             |
| `RemoteInfEngine.rid_to_address`            | 已有，但用完整 UUID 做 key         | 改用 `parse_rid_base(rid)` 做 key |
| `InteractionCache`                          | 无 `rid_base`/`sample_idx` 属性    | 新增属性                          |
| `AsyncCompletionsWithReward.create()`       | `rid=str(uuid.uuid4())`            | 改为确定性 rid + 计算 dp_rank     |
| `AsyncResponsesWithReward.create()`         | `rid=str(uuid.uuid4())`            | 同上                              |
| `OpenAIProxyWorkflow.arun_episode()`        | 不传 rid_base                      | 新增 rid_base/sample_idx 传递     |
| `RolloutController._task_input_generator()` | 无 rid 分配                        | 调用 `fill_rid_base(item)`        |
| `SGLangConfig.disable_radix_cache`          | 默认 `True`                        | DP attention 时自动改为 `False`   |
| SWE 配置 YAML                               | 无 `dp_size`/`enable_dp_attention` | 新增 DP attention 配置项          |

### 9.3 配置适配

现有 `config_swe_ray_megatron_concat.yaml` 的 sglang 部分需要补充：

```yaml
# 当前（无 DP attention）
sglang:
  model_path: /storage/.../Qwen3-30B-A3B-Instruct-2507
  context_length: 131072
  mem_fraction_static: 0.70
  # dp_size 和 enable_dp_attention 未设置（默认 1/false）

# 开启 DP attention 后
sglang:
  model_path: /storage/.../Qwen3-30B-A3B-Instruct-2507
  context_length: 131072
  mem_fraction_static: 0.70
  enable_dp_attention: true     # 新增
  dp_size: 4                    # 新增（t=4，dp_size=4 → attn_tp=1）
  disable_radix_cache: false    # 新增（必须开启 radix cache）
```

注意：当前 `sglang:d8t4` 配置下如果 `dp_size=4`，则 `attn_tp_size = 4/4 = 1`（每个 DP rank 只有 1 个 GPU 做
attention）。这对 30B-A3B MoE 模型是合理的，因为 attention head 较少。

______________________________________________________________________

## 10. 实现状态

> **分支**: `feat/dp-attention-cache` (基于 `chucai.dzq/bailing-sft-v1.0.1`) **Commit**:
> `ef2d3cea` feat: add DP attention multi-turn cache affinity routing **14 files
> changed, 535 insertions(+), 12 deletions(-)**

### Phase 1：AReaL 核心改动 ✅ 已完成

| 文件                                                      | 改动                                                    | 状态    |
| --------------------------------------------------------- | ------------------------------------------------------- | ------- |
| `areal/utils/rollout_id.py`                               | 新增 RolloutIdBuilder 类（含 threading.Lock + hashlib） | ✅ 新增 |
| `areal/api/io_struct.py`                                  | ModelRequest 添加 `data_parallel_rank` 字段 + copy()    | ✅      |
| `areal/api/cli_args.py`                                   | OpenAIProxyConfig 添加 `sglang_dp_size`                 | ✅      |
| `areal/engine/sglang_remote.py`                           | SGLangBackend 传递 `rid` + `data_parallel_rank`         | ✅      |
| `areal/infra/remote_inf_engine.py`                        | rid_to_address 改用 routing_key + deque                 | ✅      |
| `areal/experimental/openai/cache.py`                      | InteractionCache 添加 rid_base/sample_idx               | ✅      |
| `areal/experimental/openai/client.py`                     | `_build_rid_and_dp_rank()` 共享函数 + sglang_dp_size    | ✅      |
| `areal/experimental/openai/proxy/server.py`               | StartSessionRequest + SessionData 添加 rid_base         | ✅      |
| `areal/experimental/openai/proxy/client_session.py`       | OpenAIProxyClient 传递 rid_base/sample_idx              | ✅      |
| `areal/experimental/openai/proxy/proxy_rollout_server.py` | 传递 sglang_dp_size 和 rid_base                         | ✅      |
| `areal/experimental/openai/proxy/workflow.py`             | arun_episode 提取并传递 rid_base                        | ✅      |
| `areal/infra/controller/rollout_controller.py`            | submit() + prepare_batch() 调用 fill_rid_base           | ✅      |

### Phase 2：监控与可观测性 ✅ 已完成（debug logging）

在 3 个关键路由决策点添加了 `logger.debug()` 日志：

- `client.py`: `_build_rid_and_dp_rank()` — 记录 rid, dp_rank, round
- `remote_inf_engine.py`: server affinity hit/miss — 记录 routing_key, server_addr

> **TODO（后续迭代）**：SGLang Prometheus 指标采集器（`sglang_metrics.py`）、stats_tracker 计数器

### Phase 3：测试 ✅ 已完成

| 文件                                 | 内容                                           | 测试数 |
| ------------------------------------ | ---------------------------------------------- | ------ |
| `tests/test_rollout_id.py`           | RolloutIdBuilder 全面单元测试                  | 29     |
| `tests/test_dp_attention_routing.py` | ModelRequest, SGLang payload, InteractionCache | 9      |

### Phase 4：SWE RL Workflow（独立设计）

SWE RL workflow 不阻塞 DP cache 机制实现，独立设计和开发。

### 已知限制

| 限制                            | 说明                                                              | 影响                                                                               |
| ------------------------------- | ----------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `EXT_QID_IDX_FIELD` 未写入      | `fill_rid_base` 不设置 `_ext_query_id_idx`，`sample_idx` 始终为 0 | 多 sample (group_size>1) 场景下不同 sample 路由到相同 rank。需在 sample 展开处补充 |
| `EXT_QID_FIELD` 存为 list       | 仅使用 `[0]` 元素，list 是历史设计遗留                            | 无功能影响，可后续简化为 str                                                       |
| 本地 pytest 被 torch 导入链阻断 | `DefaultStager` import error（环境问题）                          | 已通过 importlib 直接加载验证通过                                                  |

______________________________________________________________________

## 11. 配置示例

```yaml
# examples/swe/config.yaml

experiment_name: swe_rl_dp_attn
trial_name: run_01

# Async training
rollout:
  max_head_offpolicyness: 4
  max_concurrent_rollouts: 32
  consumer_batch_size: 8

  # 每个 AReaL worker 对应一个 SGLang server
  # 每个 server 内部有 dp_size 个 DP attention rank
  engine:
    type: sglang

actor:
  use_decoupled_loss: true
  recompute_logprobs: true

sglang:
  enable_dp_attention: true
  disable_radix_cache: false   # 必须开启 radix cache
  dp_size: 4                   # 每个 server 内部 4 个 DP rank
  context_length: 131072
  mem_fraction_static: 0.85

openai_proxy:
  mode: subproc
  export_style: concat
  turn_discount: 1.0
```

______________________________________________________________________

## 12. 监控设计

### 11.1 现有监控能力

| 层级   | 组件                  | 已有指标                                                                                               | 输出                |
| ------ | --------------------- | ------------------------------------------------------------------------------------------------------ | ------------------- |
| AReaL  | `stats_tracker`       | `reward`, `num_turns`, `timeperf/*`                                                                    | WandB / TensorBoard |
| AReaL  | `perf_tracer`         | Chrome Trace JSONL（训练/rollout 阶段耗时）                                                            | traces.jsonl        |
| SGLang | Prometheus `/metrics` | `cache_hit_rate`, `num_running_reqs`, `gen_throughput`, `num_aborted_requests_total`, per-DP-rank 标签 | Prometheus          |

**关键发现**：SGLang **已有** per-DP-rank 的 cache_hit_rate 指标（Prometheus label `dp_rank`），但
AReaL 从未采集这些指标到 WandB。

### 11.2 需要新增的监控

#### A. AReaL 层面（通过 stats_tracker → WandB）

```python
# 在 workflow arun_episode() 或 RemoteInfEngine.agenerate() 中添加

# 1. DP Rank 路由统计
stats_tracker.scalar(
    dp_rank_assigned=dp_rank,              # 分配到哪个 rank
    rid_routing_cache_hit=1 or 0,          # rid_to_address 是否命中
)

# 2. Abort 统计
stats_tracker.scalar(
    abort_count=num_aborts_in_episode,     # 本 episode 被 abort 几次
    abort_retry_tokens=retry_token_count,  # abort 后重试消耗的 token 数
)

# 3. 多轮 Episode 统计
stats_tracker.scalar(
    episode_num_turns=turn_count,          # 总轮数
    episode_wall_time=wall_time_seconds,   # 总耗时
    episode_cross_weight_updates=n_aborts, # 跨越几次 weight update
)
```

#### B. SGLang Prometheus 指标采集（新增组件）

```python
# areal/engine/sglang_metrics.py（新增）

class SGLangMetricsScraper:
    """定期从 SGLang /metrics 端点采集关键指标，注入 stats_tracker。"""

    METRICS_TO_SCRAPE = [
        "sglang:cache_hit_rate",           # per dp_rank
        "sglang:num_running_reqs",         # per dp_rank → 负载均衡
        "sglang:num_aborted_requests_total",
        "sglang:gen_throughput",
        "sglang:realtime_tokens_total",    # prefill_compute vs prefill_cache
    ]

    async def scrape_and_record(self, server_addr: str):
        resp = await aiohttp.get(f"{server_addr}/metrics")
        metrics = parse_prometheus_text(resp.text)
        for metric in self.METRICS_TO_SCRAPE:
            for label_set, value in metrics[metric]:
                key = f"sglang/{metric}"
                if "dp_rank" in label_set:
                    key += f"/rank{label_set['dp_rank']}"
                stats_tracker.scalar(**{key: value})
```

#### C. 关键 WandB Dashboard 指标

| 指标名                               | 含义                              | 告警条件                              |
| ------------------------------------ | --------------------------------- | ------------------------------------- |
| `sglang/cache_hit_rate/rank{N}`      | 各 DP rank 的 prefix cache 命中率 | \< 30% 说明路由可能有问题             |
| `sglang/num_running_reqs/rank{N}`    | 各 DP rank 的在途请求数           | 各 rank 差异 > 50% 说明负载不均       |
| `rollout/dp_rank_distribution`       | DP rank 分配分布（histogram）     | 严重偏斜说明 hash 分布不均            |
| `rollout/abort_count`                | 每步平均 abort 次数               | 持续 > 2 说明 episode 太慢            |
| `rollout/episode_wall_time`          | Episode 平均耗时                  | 接近 weight update 间隔说明频繁 abort |
| `rollout/rid_routing_cache_hit_rate` | rid_to_address 命中率             | \< 90% 说明 LRU 缓存太小              |

### 11.3 perf_tracer 插桩

```python
# 在关键路径添加 trace_scope

# RemoteInfEngine.agenerate() 中
with trace_scope("inf_engine.agenerate", category="compute",
                 metadata={"rid": req.rid, "dp_rank": req.data_parallel_rank}):
    ...
    # abort 重试循环内
    with trace_scope("inf_engine.abort_wait", category="sync"):
        while self.workflow_executor.is_paused():
            await asyncio.sleep(0.5)

# OpenAIProxyWorkflow.arun_episode() 中
with atrace_scope("proxy_workflow.episode", category="compute",
                  metadata={"rid_base": rid_base, "sample_idx": sample_idx}):
    ...
```

### 11.4 SGLang 已有但需要启用的配置

```yaml
sglang:
  enable_metrics: true   # 开启 Prometheus /metrics 端点
  # SGLang 已有的 per-DP-rank 指标会自动暴露
```

______________________________________________________________________

## 13. 风险与缓解

| 风险                              | 影响                                                        | 缓解措施                                                                                          |
| --------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| **Abort 导致 cache 频繁失效**     | Episode 跨越 weight update 时 cache 全清，重试时 cache miss | 监控 `abort_count`；调整 `max_head_offpolicyness` 控制 weight update 频率；缩短 episode 长度      |
| **Abort 后重试的 token 版本混合** | 同一 episode 的 token 来自不同 weight version               | 正确行为：`accumulated_versions` 记录每个 token 的 version，训练时 `recompute_logprobs=true` 处理 |
| **Hash 路由负载不均**             | 某些 DP rank 请求过多                                       | 监控 `sglang/num_running_reqs/rank{N}`；hash 分布足够均匀                                         |
| **Weight reload 清空 cache**      | 跨 step 的 episode 无 cache 收益                            | 正确行为，监控 `cache_hit_rate`                                                                   |
| **Radix cache OOM**               | GPU 内存不足                                                | SGLang 自动 LRU eviction，调整 `mem_fraction_static`                                              |
| **rid 格式 parse 错误**           | 路由退化为随机                                              | 单元测试 + fallback 到 `None`（round-robin）                                                      |
| **data_parallel_rank 超范围**     | SGLang 报错                                                 | AReaL 侧验证 `0 <= rank < dp_size`                                                                |
| **监控指标未采集**                | 问题无法定位                                                | Phase 1 即接入 SGLang Prometheus 指标采集                                                         |

______________________________________________________________________

## 14. 测试计划

### 14.1 单元测试（已实现）

详见 `tests/test_rollout_id.py`（29 个测试）和 `tests/test_dp_attention_routing.py`（9 个测试）。

测试覆盖：

- `fill_rid_base`: 基本功能、global_step 参数、计数器递增、计数器按 step 重置、不同 qid 独立计数、fallback key
  优先级、hash fallback
- `get_rid_base`: 正常和 UUID fallback
- `build_rid`: 有/无 round、不同 sample
- `infer_round_idx`: 空消息、单轮、多轮
- `parse_routing_key`: 有 round、无 round、不同 round 相同 key、不同 sample 不同 key、UUID 透传
- `compute_dp_rank`: dp_size=1、确定性、跨 round 亲和性、范围检查、分布均匀性
- `EndToEnd`: 多轮亲和性、不同 episode 独立性、跨 step 同 qid 独立性
- `ModelRequest`: dp_rank 默认值、设置、copy 保持
- `SGLangPayload`: rid 在 payload、dp_rank 有/无条件包含
- `InteractionCache`: rid_base/sample_idx 默认值和设置

### 14.2 集成测试

- 模拟 3 轮多轮对话，验证所有 turn 的 `data_parallel_rank` 相同
- 模拟 rid_to_address 缓存，验证多轮命中同一 server
- 模拟 async training，验证同 qid 不同 cnt 路由独立（不同 episode 互不干扰）

______________________________________________________________________

## 15. 总结

**核心思路**：利用 SGLang 已有的 `data_parallel_rank` API，AReaL 在客户端计算确定性 DP rank（通过
`hashlib.sha256(routing_key) % dp_size`），直接指定路由。

**零 SGLang 改动。全部改动在 AReaL 侧。** 核心逻辑集中在 `RolloutIdBuilder`（确定性
rid，`{qid}-{step}-{dup_cnt}` 格式，线程安全）和 `SGLangBackend`（传递 dp_rank）。

**分支**: `feat/dp-attention-cache`
