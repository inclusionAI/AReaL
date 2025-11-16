from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AddBackendRequest(_message.Message):
    __slots__ = ("address",)
    ADDRESS_FIELD_NUMBER: _ClassVar[int]
    address: str
    def __init__(self, address: _Optional[str] = ...) -> None: ...

class AddBackendResponse(_message.Message):
    __slots__ = ("success", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: bool = ..., message: _Optional[str] = ...) -> None: ...

class DeleteBackendRequest(_message.Message):
    __slots__ = ("address",)
    ADDRESS_FIELD_NUMBER: _ClassVar[int]
    address: str
    def __init__(self, address: _Optional[str] = ...) -> None: ...

class DeleteBackendResponse(_message.Message):
    __slots__ = ("success", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: bool = ..., message: _Optional[str] = ...) -> None: ...

class ListBackendsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListBackendsResponse(_message.Message):
    __slots__ = ("backends",)
    BACKENDS_FIELD_NUMBER: _ClassVar[int]
    backends: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, backends: _Optional[_Iterable[str]] = ...) -> None: ...

class PushStatsRequest(_message.Message):
    __slots__ = ("server_host", "server_port", "timestamp", "stats_type", "scheduler_stats", "tokenizer_stats", "cache_stats", "storage_stats")
    SERVER_HOST_FIELD_NUMBER: _ClassVar[int]
    SERVER_PORT_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    STATS_TYPE_FIELD_NUMBER: _ClassVar[int]
    SCHEDULER_STATS_FIELD_NUMBER: _ClassVar[int]
    TOKENIZER_STATS_FIELD_NUMBER: _ClassVar[int]
    CACHE_STATS_FIELD_NUMBER: _ClassVar[int]
    STORAGE_STATS_FIELD_NUMBER: _ClassVar[int]
    server_host: str
    server_port: int
    timestamp: int
    stats_type: str
    scheduler_stats: SchedulerStatsProto
    tokenizer_stats: TokenizerStatsProto
    cache_stats: CacheStatsProto
    storage_stats: StorageStatsProto
    def __init__(self, server_host: _Optional[str] = ..., server_port: _Optional[int] = ..., timestamp: _Optional[int] = ..., stats_type: _Optional[str] = ..., scheduler_stats: _Optional[_Union[SchedulerStatsProto, _Mapping]] = ..., tokenizer_stats: _Optional[_Union[TokenizerStatsProto, _Mapping]] = ..., cache_stats: _Optional[_Union[CacheStatsProto, _Mapping]] = ..., storage_stats: _Optional[_Union[StorageStatsProto, _Mapping]] = ...) -> None: ...

class PushStatsResponse(_message.Message):
    __slots__ = ("success", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: bool = ..., message: _Optional[str] = ...) -> None: ...

class SchedulerStatsProto(_message.Message):
    __slots__ = ("num_running_reqs", "num_used_tokens", "token_usage", "pending_prealloc_token_usage", "swa_token_usage", "mamba_usage", "gen_throughput", "num_queue_reqs", "num_grammar_queue_reqs", "num_running_reqs_offline_batch", "cache_hit_rate", "spec_accept_length", "spec_accept_rate", "num_retracted_reqs", "num_paused_reqs", "num_prefill_prealloc_queue_reqs", "num_prefill_inflight_queue_reqs", "num_decode_prealloc_queue_reqs", "num_decode_transfer_queue_reqs", "kv_transfer_speed_gb_s", "kv_transfer_latency_ms", "kv_transfer_bootstrap_ms", "kv_transfer_alloc_ms", "utilization", "max_running_requests_under_SLO", "engine_startup_time", "engine_load_weights_time", "new_token_ratio", "is_cuda_graph")
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
    def __init__(self, num_running_reqs: _Optional[int] = ..., num_used_tokens: _Optional[int] = ..., token_usage: _Optional[float] = ..., pending_prealloc_token_usage: _Optional[float] = ..., swa_token_usage: _Optional[float] = ..., mamba_usage: _Optional[float] = ..., gen_throughput: _Optional[float] = ..., num_queue_reqs: _Optional[int] = ..., num_grammar_queue_reqs: _Optional[int] = ..., num_running_reqs_offline_batch: _Optional[int] = ..., cache_hit_rate: _Optional[float] = ..., spec_accept_length: _Optional[float] = ..., spec_accept_rate: _Optional[float] = ..., num_retracted_reqs: _Optional[int] = ..., num_paused_reqs: _Optional[int] = ..., num_prefill_prealloc_queue_reqs: _Optional[int] = ..., num_prefill_inflight_queue_reqs: _Optional[int] = ..., num_decode_prealloc_queue_reqs: _Optional[int] = ..., num_decode_transfer_queue_reqs: _Optional[int] = ..., kv_transfer_speed_gb_s: _Optional[float] = ..., kv_transfer_latency_ms: _Optional[float] = ..., kv_transfer_bootstrap_ms: _Optional[float] = ..., kv_transfer_alloc_ms: _Optional[float] = ..., utilization: _Optional[float] = ..., max_running_requests_under_SLO: _Optional[int] = ..., engine_startup_time: _Optional[float] = ..., engine_load_weights_time: _Optional[float] = ..., new_token_ratio: _Optional[float] = ..., is_cuda_graph: _Optional[float] = ...) -> None: ...

class TokenizerStatsProto(_message.Message):
    __slots__ = ("prompt_tokens", "generation_tokens", "cached_tokens", "num_requests", "num_aborted_requests", "avg_ttft", "avg_itl", "avg_e2e_latency")
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
    def __init__(self, prompt_tokens: _Optional[int] = ..., generation_tokens: _Optional[int] = ..., cached_tokens: _Optional[int] = ..., num_requests: _Optional[int] = ..., num_aborted_requests: _Optional[int] = ..., avg_ttft: _Optional[float] = ..., avg_itl: _Optional[float] = ..., avg_e2e_latency: _Optional[float] = ...) -> None: ...

class CacheStatsProto(_message.Message):
    __slots__ = ("evicted_tokens", "loaded_back_tokens", "avg_eviction_duration", "avg_load_back_duration")
    EVICTED_TOKENS_FIELD_NUMBER: _ClassVar[int]
    LOADED_BACK_TOKENS_FIELD_NUMBER: _ClassVar[int]
    AVG_EVICTION_DURATION_FIELD_NUMBER: _ClassVar[int]
    AVG_LOAD_BACK_DURATION_FIELD_NUMBER: _ClassVar[int]
    evicted_tokens: int
    loaded_back_tokens: int
    avg_eviction_duration: float
    avg_load_back_duration: float
    def __init__(self, evicted_tokens: _Optional[int] = ..., loaded_back_tokens: _Optional[int] = ..., avg_eviction_duration: _Optional[float] = ..., avg_load_back_duration: _Optional[float] = ...) -> None: ...

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
    def __init__(self, prefetch_pgs: _Optional[_Iterable[int]] = ..., backup_pgs: _Optional[_Iterable[int]] = ..., prefetch_bandwidth: _Optional[_Iterable[float]] = ..., backup_bandwidth: _Optional[_Iterable[float]] = ...) -> None: ...
