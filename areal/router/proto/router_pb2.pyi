from collections.abc import Iterable as _Iterable
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers

DESCRIPTOR: _descriptor.FileDescriptor

class AddBackendRequest(_message.Message):
    __slots__ = ("address",)
    ADDRESS_FIELD_NUMBER: _ClassVar[int]
    address: str
    def __init__(self, address: str | None = ...) -> None: ...

class AddBackendResponse(_message.Message):
    __slots__ = ("success", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: bool = ..., message: str | None = ...) -> None: ...

class DeleteBackendRequest(_message.Message):
    __slots__ = ("address",)
    ADDRESS_FIELD_NUMBER: _ClassVar[int]
    address: str
    def __init__(self, address: str | None = ...) -> None: ...

class DeleteBackendResponse(_message.Message):
    __slots__ = ("success", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: bool = ..., message: str | None = ...) -> None: ...

class ListBackendsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListBackendsResponse(_message.Message):
    __slots__ = ("backends",)
    BACKENDS_FIELD_NUMBER: _ClassVar[int]
    backends: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, backends: _Iterable[str] | None = ...) -> None: ...

class PrefillCompletionStatsProto(_message.Message):
    __slots__ = (
        "request_id",
        "prefill_latency_ms",
        "ttft_ms",
        "prompt_tokens",
        "cached_tokens",
        "queue_time_ms",
        "priority",
        "batch_size",
    )
    REQUEST_ID_FIELD_NUMBER: _ClassVar[int]
    PREFILL_LATENCY_MS_FIELD_NUMBER: _ClassVar[int]
    TTFT_MS_FIELD_NUMBER: _ClassVar[int]
    PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    CACHED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    QUEUE_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_FIELD_NUMBER: _ClassVar[int]
    BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    request_id: str
    prefill_latency_ms: float
    ttft_ms: float
    prompt_tokens: int
    cached_tokens: int
    queue_time_ms: float
    priority: int
    batch_size: int
    def __init__(
        self,
        request_id: str | None = ...,
        prefill_latency_ms: float | None = ...,
        ttft_ms: float | None = ...,
        prompt_tokens: int | None = ...,
        cached_tokens: int | None = ...,
        queue_time_ms: float | None = ...,
        priority: int | None = ...,
        batch_size: int | None = ...,
    ) -> None: ...

class PushStatsRequest(_message.Message):
    __slots__ = (
        "server_host",
        "server_port",
        "timestamp",
        "stats_type",
        "scheduler_stats",
        "tokenizer_stats",
        "cache_stats",
        "storage_stats",
        "prefill_completion_stats",
    )
    SERVER_HOST_FIELD_NUMBER: _ClassVar[int]
    SERVER_PORT_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    STATS_TYPE_FIELD_NUMBER: _ClassVar[int]
    SCHEDULER_STATS_FIELD_NUMBER: _ClassVar[int]
    TOKENIZER_STATS_FIELD_NUMBER: _ClassVar[int]
    CACHE_STATS_FIELD_NUMBER: _ClassVar[int]
    STORAGE_STATS_FIELD_NUMBER: _ClassVar[int]
    PREFILL_COMPLETION_STATS_FIELD_NUMBER: _ClassVar[int]
    server_host: str
    server_port: int
    timestamp: int
    stats_type: str
    scheduler_stats: SchedulerStatsProto
    tokenizer_stats: TokenizerStatsProto
    cache_stats: CacheStatsProto
    storage_stats: StorageStatsProto
    prefill_completion_stats: _containers.RepeatedCompositeFieldContainer[
        PrefillCompletionStatsProto
    ]
    def __init__(
        self,
        server_host: str | None = ...,
        server_port: int | None = ...,
        timestamp: int | None = ...,
        stats_type: str | None = ...,
        scheduler_stats: SchedulerStatsProto | _Mapping | None = ...,
        tokenizer_stats: TokenizerStatsProto | _Mapping | None = ...,
        cache_stats: CacheStatsProto | _Mapping | None = ...,
        storage_stats: StorageStatsProto | _Mapping | None = ...,
        prefill_completion_stats: _Iterable[PrefillCompletionStatsProto | _Mapping]
        | None = ...,
    ) -> None: ...

class PushStatsResponse(_message.Message):
    __slots__ = ("success", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: bool = ..., message: str | None = ...) -> None: ...

class SchedulerStatsProto(_message.Message):
    __slots__ = (
        "num_running_reqs",
        "num_used_tokens",
        "token_usage",
        "pending_prealloc_token_usage",
        "swa_token_usage",
        "mamba_usage",
        "gen_throughput",
        "num_queue_reqs",
        "num_grammar_queue_reqs",
        "num_running_reqs_offline_batch",
        "cache_hit_rate",
        "spec_accept_length",
        "spec_accept_rate",
        "num_retracted_reqs",
        "num_paused_reqs",
        "num_prefill_prealloc_queue_reqs",
        "num_prefill_inflight_queue_reqs",
        "num_decode_prealloc_queue_reqs",
        "num_decode_transfer_queue_reqs",
        "kv_transfer_speed_gb_s",
        "kv_transfer_latency_ms",
        "kv_transfer_bootstrap_ms",
        "kv_transfer_alloc_ms",
        "utilization",
        "max_running_requests_under_SLO",
        "engine_startup_time",
        "engine_load_weights_time",
        "new_token_ratio",
        "is_cuda_graph",
    )
    NUM_RUNNING_REQS_FIELD_NUMBER: _ClassVar[int]
    NUM_USED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    TOKEN_USAGE_FIELD_NUMBER: _ClassVar[int]
    PENDING_PREALLOC_TOKEN_USAGE_FIELD_NUMBER: _ClassVar[int]
    SWA_TOKEN_USAGE_FIELD_NUMBER: _ClassVar[int]
    MAMBA_USAGE_FIELD_NUMBER: _ClassVar[int]
    GEN_THROUGHPUT_FIELD_NUMBER: _ClassVar[int]
    NUM_QUEUE_REQS_FIELD_NUMBER: _ClassVar[int]
    NUM_GRAMMAR_QUEUE_REQS_FIELD_NUMBER: _ClassVar[int]
    NUM_RUNNING_REQS_OFFLINE_BATCH_FIELD_NUMBER: _ClassVar[int]
    CACHE_HIT_RATE_FIELD_NUMBER: _ClassVar[int]
    SPEC_ACCEPT_LENGTH_FIELD_NUMBER: _ClassVar[int]
    SPEC_ACCEPT_RATE_FIELD_NUMBER: _ClassVar[int]
    NUM_RETRACTED_REQS_FIELD_NUMBER: _ClassVar[int]
    NUM_PAUSED_REQS_FIELD_NUMBER: _ClassVar[int]
    NUM_PREFILL_PREALLOC_QUEUE_REQS_FIELD_NUMBER: _ClassVar[int]
    NUM_PREFILL_INFLIGHT_QUEUE_REQS_FIELD_NUMBER: _ClassVar[int]
    NUM_DECODE_PREALLOC_QUEUE_REQS_FIELD_NUMBER: _ClassVar[int]
    NUM_DECODE_TRANSFER_QUEUE_REQS_FIELD_NUMBER: _ClassVar[int]
    KV_TRANSFER_SPEED_GB_S_FIELD_NUMBER: _ClassVar[int]
    KV_TRANSFER_LATENCY_MS_FIELD_NUMBER: _ClassVar[int]
    KV_TRANSFER_BOOTSTRAP_MS_FIELD_NUMBER: _ClassVar[int]
    KV_TRANSFER_ALLOC_MS_FIELD_NUMBER: _ClassVar[int]
    UTILIZATION_FIELD_NUMBER: _ClassVar[int]
    MAX_RUNNING_REQUESTS_UNDER_SLO_FIELD_NUMBER: _ClassVar[int]
    ENGINE_STARTUP_TIME_FIELD_NUMBER: _ClassVar[int]
    ENGINE_LOAD_WEIGHTS_TIME_FIELD_NUMBER: _ClassVar[int]
    NEW_TOKEN_RATIO_FIELD_NUMBER: _ClassVar[int]
    IS_CUDA_GRAPH_FIELD_NUMBER: _ClassVar[int]
    num_running_reqs: int
    num_used_tokens: int
    token_usage: float
    pending_prealloc_token_usage: float
    swa_token_usage: float
    mamba_usage: float
    gen_throughput: float
    num_queue_reqs: int
    num_grammar_queue_reqs: int
    num_running_reqs_offline_batch: int
    cache_hit_rate: float
    spec_accept_length: float
    spec_accept_rate: float
    num_retracted_reqs: int
    num_paused_reqs: int
    num_prefill_prealloc_queue_reqs: int
    num_prefill_inflight_queue_reqs: int
    num_decode_prealloc_queue_reqs: int
    num_decode_transfer_queue_reqs: int
    kv_transfer_speed_gb_s: float
    kv_transfer_latency_ms: float
    kv_transfer_bootstrap_ms: float
    kv_transfer_alloc_ms: float
    utilization: float
    max_running_requests_under_SLO: int
    engine_startup_time: float
    engine_load_weights_time: float
    new_token_ratio: float
    is_cuda_graph: float
    def __init__(
        self,
        num_running_reqs: int | None = ...,
        num_used_tokens: int | None = ...,
        token_usage: float | None = ...,
        pending_prealloc_token_usage: float | None = ...,
        swa_token_usage: float | None = ...,
        mamba_usage: float | None = ...,
        gen_throughput: float | None = ...,
        num_queue_reqs: int | None = ...,
        num_grammar_queue_reqs: int | None = ...,
        num_running_reqs_offline_batch: int | None = ...,
        cache_hit_rate: float | None = ...,
        spec_accept_length: float | None = ...,
        spec_accept_rate: float | None = ...,
        num_retracted_reqs: int | None = ...,
        num_paused_reqs: int | None = ...,
        num_prefill_prealloc_queue_reqs: int | None = ...,
        num_prefill_inflight_queue_reqs: int | None = ...,
        num_decode_prealloc_queue_reqs: int | None = ...,
        num_decode_transfer_queue_reqs: int | None = ...,
        kv_transfer_speed_gb_s: float | None = ...,
        kv_transfer_latency_ms: float | None = ...,
        kv_transfer_bootstrap_ms: float | None = ...,
        kv_transfer_alloc_ms: float | None = ...,
        utilization: float | None = ...,
        max_running_requests_under_SLO: int | None = ...,
        engine_startup_time: float | None = ...,
        engine_load_weights_time: float | None = ...,
        new_token_ratio: float | None = ...,
        is_cuda_graph: float | None = ...,
    ) -> None: ...

class TokenizerStatsProto(_message.Message):
    __slots__ = (
        "prompt_tokens",
        "generation_tokens",
        "cached_tokens",
        "num_requests",
        "num_aborted_requests",
        "avg_ttft",
        "avg_itl",
        "avg_e2e_latency",
    )
    PROMPT_TOKENS_FIELD_NUMBER: _ClassVar[int]
    GENERATION_TOKENS_FIELD_NUMBER: _ClassVar[int]
    CACHED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    NUM_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    NUM_ABORTED_REQUESTS_FIELD_NUMBER: _ClassVar[int]
    AVG_TTFT_FIELD_NUMBER: _ClassVar[int]
    AVG_ITL_FIELD_NUMBER: _ClassVar[int]
    AVG_E2E_LATENCY_FIELD_NUMBER: _ClassVar[int]
    prompt_tokens: int
    generation_tokens: int
    cached_tokens: int
    num_requests: int
    num_aborted_requests: int
    avg_ttft: float
    avg_itl: float
    avg_e2e_latency: float
    def __init__(
        self,
        prompt_tokens: int | None = ...,
        generation_tokens: int | None = ...,
        cached_tokens: int | None = ...,
        num_requests: int | None = ...,
        num_aborted_requests: int | None = ...,
        avg_ttft: float | None = ...,
        avg_itl: float | None = ...,
        avg_e2e_latency: float | None = ...,
    ) -> None: ...

class CacheStatsProto(_message.Message):
    __slots__ = (
        "evicted_tokens",
        "loaded_back_tokens",
        "avg_eviction_duration",
        "avg_load_back_duration",
    )
    EVICTED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    LOADED_BACK_TOKENS_FIELD_NUMBER: _ClassVar[int]
    AVG_EVICTION_DURATION_FIELD_NUMBER: _ClassVar[int]
    AVG_LOAD_BACK_DURATION_FIELD_NUMBER: _ClassVar[int]
    evicted_tokens: int
    loaded_back_tokens: int
    avg_eviction_duration: float
    avg_load_back_duration: float
    def __init__(
        self,
        evicted_tokens: int | None = ...,
        loaded_back_tokens: int | None = ...,
        avg_eviction_duration: float | None = ...,
        avg_load_back_duration: float | None = ...,
    ) -> None: ...

class StorageStatsProto(_message.Message):
    __slots__ = ("prefetch_pgs", "backup_pgs", "prefetch_bandwidth", "backup_bandwidth")
    PREFETCH_PGS_FIELD_NUMBER: _ClassVar[int]
    BACKUP_PGS_FIELD_NUMBER: _ClassVar[int]
    PREFETCH_BANDWIDTH_FIELD_NUMBER: _ClassVar[int]
    BACKUP_BANDWIDTH_FIELD_NUMBER: _ClassVar[int]
    prefetch_pgs: _containers.RepeatedScalarFieldContainer[int]
    backup_pgs: _containers.RepeatedScalarFieldContainer[int]
    prefetch_bandwidth: _containers.RepeatedScalarFieldContainer[float]
    backup_bandwidth: _containers.RepeatedScalarFieldContainer[float]
    def __init__(
        self,
        prefetch_pgs: _Iterable[int] | None = ...,
        backup_pgs: _Iterable[int] | None = ...,
        prefetch_bandwidth: _Iterable[float] | None = ...,
        backup_bandwidth: _Iterable[float] | None = ...,
    ) -> None: ...
