# SPDX-License-Identifier: Apache-2.0

import hashlib

import pytest

from areal.engine import sglang_qsa_compress_gather as qsa_patch
from areal.engine import sglang_qwen4_vl as patch


def test_qsa_patch_allows_repeated_startup(tmp_path, monkeypatch):
    source = ("def gather():\n    if True:\n" + qsa_patch.OLD).encode()
    expected = source.replace(qsa_patch.OLD.encode(), qsa_patch.NEW.encode())
    monkeypatch.setattr(
        qsa_patch, "EXPECTED_SHA256", hashlib.sha256(source).hexdigest()
    )
    monkeypatch.setattr(
        qsa_patch, "PATCHED_SHA256", hashlib.sha256(expected).hexdigest()
    )
    patched = qsa_patch.patched_source(source)
    assert patched == expected
    assert qsa_patch.patched_source(patched) == patched
    target = tmp_path / "qsa_indexer.py"
    target.write_bytes(source)
    monkeypatch.setattr("sys.argv", ["patch", "--target", str(target)])
    qsa_patch.main()
    qsa_patch.main()
    assert target.read_bytes() == patched


def test_unknown_qsa_version_is_not_modified(tmp_path, monkeypatch):
    target = tmp_path / "qsa_indexer.py"
    target.write_bytes(b"# unsupported QSA indexer\n")
    before = target.read_bytes()
    monkeypatch.setattr("sys.argv", ["patch", "--target", str(target)])
    with pytest.raises(RuntimeError, match="Unknown SGLang"):
        qsa_patch.main()
    assert target.read_bytes() == before


def test_unknown_processor_version_is_not_modified(tmp_path, monkeypatch):
    source = tmp_path / "qwen_vl.py"
    source.write_bytes(b"# an unsupported SGLang version\n")
    before = source.read_bytes()
    monkeypatch.setattr("sys.argv", ["patch", "--target", str(source)])
    with pytest.raises(RuntimeError, match="Unknown SGLang"):
        patch.main()
    assert source.read_bytes() == before


def test_processor_patch_is_idempotent_and_keeps_other_model_policies(monkeypatch):
    # Isolate the source transformer from imports of the optional GPU runtime.
    source = (
        "class Base:\n"
        "    def _fast_image_processor_device(self, processor):\n"
        '        return "cuda:0"\n'
        "class Processor(Base):\n    def configure(self):\n"
        + patch.OLD
        + "            self.factor = 32\n"
        + '        if self.model_type == "qwen3_vl":\n'
        + "            self.max_pixels //= 16\n"
        + "    def should_resize(self, images):\n"
        + "        return (\n"
        + patch.RESIZE_OLD
        + "            and bool(images)\n"
        + "        )\n"
        + patch.DEVICE_OLD
        + "        self, *args, **kwargs\n    ):\n        pass\n"
    ).encode()
    legacy = source.replace(patch.OLD.encode(), patch.NEW.encode())
    expected = legacy.replace(patch.RESIZE_OLD.encode(), patch.RESIZE_NEW.encode())
    expected = expected.replace(patch.DEVICE_OLD.encode(), patch.DEVICE_NEW.encode())
    monkeypatch.setattr(patch, "EXPECTED_SHA256", hashlib.sha256(source).hexdigest())
    monkeypatch.setattr(
        patch, "LEGACY_PATCHED_SHA256", hashlib.sha256(legacy).hexdigest()
    )
    monkeypatch.setattr(patch, "PATCHED_SHA256", hashlib.sha256(expected).hexdigest())
    patched = patch.patched_source(source)
    assert patch.patched_source(patched) == patched
    assert patch.patched_source(legacy) == patched
    namespace = {}
    exec(compile(patched, "qwen_vl.py", "exec"), namespace)
    for model, factor, pixels in [
        ("qwen4_exp", 32, 16777216),
        ("qwen3_vl", 32, 1048576),
        ("qwen2_vl", 28, 16777216),
    ]:
        processor = namespace["Processor"]()
        processor.model_type = model
        processor.factor = 28
        processor.max_pixels = 16777216
        processor.configure()
        assert (processor.factor, processor.max_pixels) == (factor, pixels)
        assert processor.should_resize([object()]) == (model != "qwen4_exp")
        assert not processor.should_resize([])
        expected_device = "cpu" if model == "qwen4_exp" else "cuda:0"
        assert processor._fast_image_processor_device(None) == expected_device
    processor.model_type = "paddleocr_vl"
    assert not processor.should_resize([object()])
