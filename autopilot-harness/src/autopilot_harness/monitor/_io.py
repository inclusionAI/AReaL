"""Small shared helpers; no training or telemetry SDK dependencies."""

import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


class SourceError(RuntimeError):
    """An operational error with a credential-free, user-facing message."""


def timestamp(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(
            "timestamp must be Unix seconds or an ISO 8601 time with timezone"
        )
    if isinstance(value, str):
        try:
            result = float(value)
        except ValueError:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError("ISO 8601 timestamps must include a timezone")
            result = parsed.timestamp()
    else:
        result = float(value)
    if not math.isfinite(result):
        raise ValueError("timestamp must be finite")
    return result


def base_url(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("service URL must be a string")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "service URL must be http(s), without credentials, query, or fragment"
        )
    return value.rstrip("/")


def endpoint_reference(value: str | None) -> str | None:
    """Only serialize validated addresses, including addresses from the environment."""
    if value is None:
        return None
    try:
        return base_url(value)
    except ValueError:
        return "[invalid service URL]"


def positive_number(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if isinstance(value, bool) or not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def resolve_path(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


_SECRET_KEY = re.compile(
    r"(^|[_/.-])(api[_-]?key|password|passwd|secret|access[_-]?token|"
    r"auth[_-]?token|bearer[_-]?token|authorization|credential)(s)?($|[_/.-])",
    re.IGNORECASE,
)


def redacted(value: Any) -> Any:
    """Redact credential fields, while retaining metrics like token_usage."""
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]"
            if _SECRET_KEY.search(str(key)) and not str(key).endswith("_env")
            else redacted(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redacted(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(
            redacted(value), stream, ensure_ascii=False, indent=2, allow_nan=False
        )
        stream.write("\n")
