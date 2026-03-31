from dataclasses import dataclass


@dataclass
class DataProxyConfig:
    host: str = "0.0.0.0"
    port: int = 8082
    backend_addr: str = "http://localhost:30000"  # co-located SGLang/vLLM
    backend_type: str = "sglang"
    tokenizer_path: str = ""
    log_level: str = "info"
    request_timeout: float = 120.0  # seconds per SGLang call
    set_reward_finish_timeout: float = 0.0
    max_resubmit_retries: int = 20  # max abort/resubmit cycles before giving up
    resubmit_wait: float = 0.5  # seconds between is_paused polls
    admin_api_key: str = "areal-admin-key"  # admin key for authentication
    callback_server_addr: str = ""
    # Resolved serving address (host:port) used as node_addr for RTensor shards.
    # Set at startup by __main__.py after the host is resolved.
    serving_addr: str = ""
