# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from areal.experimental.inference_service.data_proxy.session import (
    SESSION_TIMEOUT_SECONDS,
)


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
    session_timeout_seconds: float = SESSION_TIMEOUT_SECONDS
    stale_session_cleanup_interval_seconds: float = 60.0
    stale_session_dump_path: str = ""
    max_resubmit_retries: int = 20  # max abort/resubmit cycles before giving up
    resubmit_wait: float = 0.5  # seconds between is_paused polls
    admin_api_key: str = "areal-admin-key"  # admin key for authentication
    callback_server_addr: str = ""
    # Resolved serving address (host:port) used as node_addr for RTensor shards.
    # Set at startup by __main__.py after the host is resolved.
    serving_addr: str = ""

    # ArealOpenAI client parameters (forwarded from OpenAIProxyConfig)
    tool_call_parser: str = "qwen"
    reasoning_parser: str = "qwen3"
    engine_max_tokens: int | None = None
    chat_template_type: str = "hf"
