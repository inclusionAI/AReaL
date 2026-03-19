"""Deterministic rollout ID construction for DP attention cache affinity.

Ensures multi-turn conversation requests route to the same DP rank,
enabling prefix cache hits in SGLang's DP attention mode.

rid format: {qid}-{step}-{dup_cnt}-r-{round_idx}_{sample_idx}
rid_base:   {qid}-{step}-{dup_cnt}
routing_key (parse_routing_key): {qid}-{step}-{dup_cnt}_{sample_idx}
"""

from __future__ import annotations

import hashlib
import threading
import uuid
from collections import defaultdict
from typing import ClassVar


class RolloutIdBuilder:
    """Construct deterministic rids for DP attention routing affinity.

    Attributes:
        EXT_QID_FIELD: Key in data_item storing the rid_base list.
        EXT_QID_IDX_FIELD: Key in data_item storing the sample index.
    """

    EXT_QID_FIELD = "_ext_query_id"
    EXT_QID_IDX_FIELD = "_ext_query_id_idx"

    # Per-(qid, step) dup counter. Resets implicitly when a new step appears.
    _qid_counter: ClassVar[defaultdict[str, int]] = defaultdict(int)
    _counter_lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def fill_rid_base(cls, data_item: dict, global_step: int = 0) -> None:
        """Assign a deterministic rid_base to a data item.

        Format: "{qid}-{step}-{dup_cnt}" where:
        - qid: extracted from query_id/qid/id/instance_id, or hash(prompt)
        - step: training global step (version)
        - dup_cnt: per-(qid, step) duplicate counter
        """
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
        """Get the rid_base from a data item, or fallback to random UUID."""
        ext = data_item.get(cls.EXT_QID_FIELD)
        if ext and len(ext) > 0:
            return ext[0]
        return str(uuid.uuid4())

    @classmethod
    def build_rid(
        cls, rid_base: str, sample_idx: int, round_idx: int | None = None
    ) -> str:
        """Construct the final rid.

        Returns:
            With round: "{rid_base}-r-{round_idx}_{sample_idx}"
            Without round: "{rid_base}_{sample_idx}"
        """
        if round_idx is not None:
            return f"{rid_base}-r-{round_idx}_{sample_idx}"
        return f"{rid_base}_{sample_idx}"

    @classmethod
    def infer_round_idx(cls, messages: list[dict]) -> int:
        """Count assistant messages as round_idx."""
        return sum(1 for m in messages if m.get("role") == "assistant")

    @classmethod
    def parse_routing_key(cls, rid: str) -> str:
        """Extract routing key (strip round part, keep sample_idx).

        "django-123-5-0-r-2_0" → "django-123-5-0_0"
        "django-123-5-0_0"     → "django-123-5-0_0" (no round, unchanged)
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
        """Compute deterministic DP rank from rid.

        Same routing_key → same DP rank, ensuring cache affinity.
        Uses hashlib (not built-in hash()) for cross-process determinism.
        """
        if dp_size <= 1:
            return 0
        routing_key = cls.parse_routing_key(rid)
        h = int(hashlib.sha256(routing_key.encode()).hexdigest(), 16)
        return h % dp_size

    @classmethod
    def reset(cls) -> None:
        """Reset the global counter. Useful for testing."""
        cls._qid_counter.clear()
