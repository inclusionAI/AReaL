"""Unit tests for the wandb size-metric helpers added to LoRA disk sync.

Covers two production paths introduced alongside LoRA disk sync:

* ``FSDPEngine._log_disk_save_size`` -- after the FSDP engine writes a
  checkpoint directory (full HF or PEFT adapter), it reports the
  on-disk byte count via the **default** ``stats_tracker`` under flat
  top-level keys:

    - ``weight_update_disk_bytes``       (always populated)
    - ``weight_update_disk_lora_bytes``  (when ``use_lora=True``)
    - ``weight_update_disk_full_bytes``  (when ``use_lora=False``)

* ``SGLangBackend._log_disk_send_size`` -- the inference side records
  how many bytes the SGLang process will pull from disk for the disk
  weight-update HTTP call.  Same default-tracker, flat-key contract:

    - ``weight_update_send_bytes``       (always populated)
    - ``weight_update_send_lora_bytes``  (LoRA branch)
    - ``weight_update_send_full_bytes``  (full-model branch)

These tests are CPU-only, do not touch FSDP / SGLang / GPUs, and run
without a process group.  Failures of either helper must NEVER bubble
up to the caller (metric emission must not break training); these
tests assert that contract too.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from areal.api import WeightUpdateMeta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_files_with_total_size(directory: str, total_bytes: int) -> None:
    """Create a couple of files under ``directory`` summing to ``total_bytes``.

    Splits the requested size across two files so the recursive
    ``os.walk`` aggregation is genuinely exercised (not just a single
    ``os.path.getsize``).
    """
    os.makedirs(directory, exist_ok=True)
    if total_bytes <= 0:
        # Touch one empty file so the walk still has something to see.
        with open(os.path.join(directory, "empty.bin"), "wb") as f:
            f.write(b"")
        return
    half = total_bytes // 2
    rest = total_bytes - half
    with open(os.path.join(directory, "a.bin"), "wb") as f:
        f.write(b"\x00" * half)
    with open(os.path.join(directory, "b.bin"), "wb") as f:
        f.write(b"\x00" * rest)


# ---------------------------------------------------------------------------
# FSDPEngine._log_disk_save_size
# ---------------------------------------------------------------------------


class TestLogDiskSaveSize:
    """Exercise the unbound method on a stub object so we don't need FSDP."""

    @staticmethod
    def _make_engine_stub(*, use_lora: bool):
        """Build a SimpleNamespace that quacks like an FSDPEngine for the
        narrow surface ``_log_disk_save_size`` actually touches.
        """
        return SimpleNamespace(
            config=SimpleNamespace(use_lora=use_lora),
            logger=MagicMock(),
        )

    def _invoke(self, engine_stub, path):
        from areal.engine.fsdp_engine import FSDPEngine

        # Bind the unbound method to the stub.
        return FSDPEngine._log_disk_save_size(engine_stub, path)

    def test_lora_branch_writes_bytes_and_lora_bytes(self, tmp_path):
        adapter_dir = tmp_path / "weight_update_v1"
        _write_files_with_total_size(str(adapter_dir), 12345)

        engine = self._make_engine_stub(use_lora=True)
        with patch(
            "areal.engine.fsdp_engine.stats_tracker.scalar"
        ) as mock_scalar:
            self._invoke(engine, str(adapter_dir))

        mock_scalar.assert_called_once()
        kwargs = mock_scalar.call_args.kwargs
        assert kwargs["weight_update_disk_bytes"] == 12345.0
        assert kwargs["weight_update_disk_lora_bytes"] == 12345.0
        assert "weight_update_disk_full_bytes" not in kwargs
        # Logger should have emitted a `[weight_update_disk]` info line.
        assert engine.logger.info.called
        msg = engine.logger.info.call_args.args[0]
        assert "[weight_update_disk]" in msg
        assert "use_lora=True" in msg

    def test_full_branch_writes_bytes_and_full_bytes(self, tmp_path):
        full_dir = tmp_path / "full_model"
        _write_files_with_total_size(str(full_dir), 7777)

        engine = self._make_engine_stub(use_lora=False)
        with patch(
            "areal.engine.fsdp_engine.stats_tracker.scalar"
        ) as mock_scalar:
            self._invoke(engine, str(full_dir))

        kwargs = mock_scalar.call_args.kwargs
        assert kwargs["weight_update_disk_bytes"] == 7777.0
        assert kwargs["weight_update_disk_full_bytes"] == 7777.0
        assert "weight_update_disk_lora_bytes" not in kwargs

    def test_recursive_walk_sums_subdirectories(self, tmp_path):
        root = tmp_path / "weight_update_v9"
        sub = root / "nested"
        os.makedirs(sub, exist_ok=True)
        # 200 bytes top-level + 300 bytes nested = 500 total
        with open(root / "top.bin", "wb") as f:
            f.write(b"\x00" * 200)
        with open(sub / "deep.bin", "wb") as f:
            f.write(b"\x00" * 300)

        engine = self._make_engine_stub(use_lora=True)
        with patch(
            "areal.engine.fsdp_engine.stats_tracker.scalar"
        ) as mock_scalar:
            self._invoke(engine, str(root))

        kwargs = mock_scalar.call_args.kwargs
        assert kwargs["weight_update_disk_bytes"] == 500.0
        assert kwargs["weight_update_disk_lora_bytes"] == 500.0

    def test_nonexistent_path_does_not_raise(self, tmp_path):
        """Reporting a non-existent path must not break the training loop;
        ``os.walk`` simply yields nothing and the recorded size is 0.
        """
        bad = tmp_path / "does_not_exist"
        engine = self._make_engine_stub(use_lora=True)
        with patch(
            "areal.engine.fsdp_engine.stats_tracker.scalar"
        ) as mock_scalar:
            # Must not raise.
            self._invoke(engine, str(bad))
        # Either reported as zero (preferred) or skipped silently.
        if mock_scalar.called:
            kwargs = mock_scalar.call_args.kwargs
            assert kwargs["weight_update_disk_bytes"] == 0.0

    def test_scalar_failure_is_swallowed(self, tmp_path):
        """If ``stats_tracker.scalar`` raises, the helper must catch it
        and log a warning rather than propagate.
        """
        d = tmp_path / "wu"
        _write_files_with_total_size(str(d), 100)
        engine = self._make_engine_stub(use_lora=True)
        with patch(
            "areal.engine.fsdp_engine.stats_tracker.scalar",
            side_effect=RuntimeError("boom"),
        ):
            # Must not raise.
            self._invoke(engine, str(d))
        # The warning path should have been hit.
        assert engine.logger.warning.called


# ---------------------------------------------------------------------------
# SGLangBackend._log_disk_send_size
# ---------------------------------------------------------------------------


class TestLogDiskSendSize:
    def test_lora_meta_records_bytes_and_lora_bytes(self, tmp_path):
        from areal.engine.sglang_remote import SGLangBackend

        d = tmp_path / "weight_update_v3"
        _write_files_with_total_size(str(d), 4096)
        meta = WeightUpdateMeta(
            type="disk",
            use_lora=True,
            lora_name="L",
            version=3,
            path=str(d),
        )

        with patch(
            "areal.engine.sglang_remote.stats_tracker.scalar"
        ) as mock_scalar:
            SGLangBackend._log_disk_send_size(meta, use_lora=True)

        mock_scalar.assert_called_once()
        kwargs = mock_scalar.call_args.kwargs
        assert kwargs["weight_update_send_bytes"] == 4096.0
        assert kwargs["weight_update_send_lora_bytes"] == 4096.0
        assert "weight_update_send_full_bytes" not in kwargs

    def test_full_meta_records_bytes_and_full_bytes(self, tmp_path):
        from areal.engine.sglang_remote import SGLangBackend

        d = tmp_path / "full"
        _write_files_with_total_size(str(d), 999)
        meta = WeightUpdateMeta(type="disk", use_lora=False, path=str(d))

        with patch(
            "areal.engine.sglang_remote.stats_tracker.scalar"
        ) as mock_scalar:
            SGLangBackend._log_disk_send_size(meta, use_lora=False)

        kwargs = mock_scalar.call_args.kwargs
        assert kwargs["weight_update_send_bytes"] == 999.0
        assert kwargs["weight_update_send_full_bytes"] == 999.0
        assert "weight_update_send_lora_bytes" not in kwargs

    def test_meta_path_is_a_file_not_dir(self, tmp_path):
        """If ``meta.path`` happens to point at a single file rather than
        a directory, the helper must still record its size.
        """
        from areal.engine.sglang_remote import SGLangBackend

        f = tmp_path / "alone.safetensors"
        f.write_bytes(b"\x00" * 555)
        meta = WeightUpdateMeta(
            type="disk", use_lora=True, lora_name="L", version=0, path=str(f)
        )

        with patch(
            "areal.engine.sglang_remote.stats_tracker.scalar"
        ) as mock_scalar:
            SGLangBackend._log_disk_send_size(meta, use_lora=True)

        kwargs = mock_scalar.call_args.kwargs
        assert kwargs["weight_update_send_bytes"] == 555.0
        assert kwargs["weight_update_send_lora_bytes"] == 555.0

    def test_none_path_is_a_noop(self):
        from areal.engine.sglang_remote import SGLangBackend

        meta = WeightUpdateMeta(type="disk", use_lora=True, path=None)
        with patch(
            "areal.engine.sglang_remote.stats_tracker.scalar"
        ) as mock_scalar:
            SGLangBackend._log_disk_send_size(meta, use_lora=True)
        # No path -> nothing to size -> no scalar call.
        mock_scalar.assert_not_called()

    def test_scalar_failure_is_swallowed(self, tmp_path):
        from areal.engine.sglang_remote import SGLangBackend

        d = tmp_path / "wu"
        _write_files_with_total_size(str(d), 10)
        meta = WeightUpdateMeta(
            type="disk", use_lora=True, lora_name="L", version=0, path=str(d)
        )
        with patch(
            "areal.engine.sglang_remote.stats_tracker.scalar",
            side_effect=RuntimeError("boom"),
        ):
            # Must not raise.
            SGLangBackend._log_disk_send_size(meta, use_lora=True)


# ---------------------------------------------------------------------------
# build_disk_weight_update_requests integration: dispatch must call
# _log_disk_send_size on both branches.
# ---------------------------------------------------------------------------


class TestBuildDiskRequestsCallsSendSizeMetric:
    def test_lora_branch_invokes_log_disk_send_size(self, tmp_path):
        from areal.engine.sglang_remote import SGLangBackend

        d = tmp_path / "weight_update_v1"
        _write_files_with_total_size(str(d), 10)
        backend = SGLangBackend()
        meta = WeightUpdateMeta(
            type="disk",
            use_lora=True,
            lora_name="my-lora",
            version=1,
            path=str(d),
        )

        with patch.object(SGLangBackend, "_log_disk_send_size") as mock_metric:
            backend.build_disk_weight_update_requests(meta)
        mock_metric.assert_called_once()
        # use_lora kwarg must be True for the LoRA branch.
        assert mock_metric.call_args.kwargs["use_lora"] is True

    def test_full_branch_invokes_log_disk_send_size(self, tmp_path):
        from areal.engine.sglang_remote import SGLangBackend

        d = tmp_path / "full"
        _write_files_with_total_size(str(d), 10)
        backend = SGLangBackend()
        meta = WeightUpdateMeta(type="disk", use_lora=False, path=str(d))

        with patch.object(SGLangBackend, "_log_disk_send_size") as mock_metric:
            backend.build_disk_weight_update_requests(meta)
        mock_metric.assert_called_once()
        assert mock_metric.call_args.kwargs["use_lora"] is False
