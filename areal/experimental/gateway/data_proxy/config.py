from dataclasses import dataclass


@dataclass
class DataProxyConfig:
    host: str = "0.0.0.0"
    port: int = 8082
    backend_addr: str = "http://localhost:30000"  # co-located SGLang/vLLM
    tokenizer_path: str = ""
    log_level: str = "info"
    request_timeout: float = 120.0  # seconds per SGLang call
