# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest

from areal.engine import sglang_model_patches
from areal.engine import sglang_qwen35_vl as patch


@pytest.fixture
def sources(monkeypatch):
    qwen = (
        "class Processor:\n"
        + patch.DECODE_OLD
        + "    def configure(self):\n"
        + patch.LIMIT_OLD
        + "                self.max_pixels //= 16\n"
        + "    def should_resize(self, images):\n        return (\n"
        + patch.RESIZE_OLD
        + "            and bool(images)\n        )\n"
    ).encode()
    base = (
        "class Base:\n"
        "    def device(self):\n"
        "        if True:\n"
        + patch.DEVICE_OLD
        + '                return "cpu"\n'
        + "            else:\n"
        + '                return "cuda:0"\n'
    ).encode()
    qwen_expected = qwen
    for prefix in ("LIMIT", "RESIZE", "DECODE"):
        qwen_expected = qwen_expected.replace(
            getattr(patch, prefix + "_OLD").encode(),
            getattr(patch, prefix + "_NEW").encode(),
        )
    base_expected = base.replace(patch.DEVICE_OLD.encode(), patch.DEVICE_NEW.encode())
    for name, value in (
        ("EXPECTED_SHA256", qwen),
        ("PATCHED_SHA256", qwen_expected),
        ("BASE_EXPECTED_SHA256", base),
        ("BASE_PATCHED_SHA256", base_expected),
    ):
        monkeypatch.setattr(patch, name, hashlib.sha256(value).hexdigest())
    return qwen, base


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_moe"])
def test_qwen35_startup_installs_both_patches_atomically_and_is_idempotent(
    sources, tmp_path, monkeypatch, model_type
):
    root = tmp_path / "sglang"
    for relative, source in zip((patch.TARGET_RELATIVE, patch.BASE_RELATIVE), sources):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": model_type}))
    monkeypatch.setattr(sglang_model_patches, "installed_sglang_root", lambda: root)
    for _ in range(2):
        sglang_model_patches.apply_model_patches(["--model-path", str(model)])
    assert (root / patch.TARGET_RELATIVE).read_bytes() == patch.patched_source(
        sources[0]
    )
    assert (root / patch.BASE_RELATIVE).read_bytes() == patch.patched_base_source(
        sources[1]
    )


def test_qwen35_patch_keeps_other_models_pixel_policy_and_uses_cpu(sources):
    from types import SimpleNamespace

    namespace = {
        "envs": SimpleNamespace(
            SGLANG_VLM_QWEN_LIMIT_PIXEL=SimpleNamespace(get=lambda: True)
        ),
        "_is_cpu": False,
        "get_global_server_args": lambda: SimpleNamespace(rl_on_policy_target=None),
    }
    exec(patch.patched_source(sources[0]), namespace)
    exec(patch.patched_base_source(sources[1]), namespace)
    for model in ("qwen3_5", "qwen3_5_moe", "qwen3_vl", "qwen3_vl_moe"):
        processor = namespace["Processor"]()
        processor.model_type = model
        processor.max_pixels = 16777216
        processor.configure()
        fixed = model in ("qwen3_5", "qwen3_5_moe")
        assert processor.max_pixels == (16777216 if fixed else 1048576)
        assert processor.should_resize([object()]) == (not fixed)
        base = namespace["Base"]()
        base.model_type = model
        assert base.device() == ("cpu" if fixed else "cuda:0")


def test_unknown_base_source_leaves_both_files_unchanged(sources, tmp_path):
    qwen = tmp_path / patch.TARGET_RELATIVE
    base = tmp_path / patch.BASE_RELATIVE
    qwen.parent.mkdir(parents=True)
    qwen.write_bytes(sources[0])
    base.write_bytes(b"# unknown base\n")
    with pytest.raises(RuntimeError, match="Unknown SGLang"):
        patch.install(tmp_path)
    assert qwen.read_bytes() == sources[0]
    assert base.read_bytes() == b"# unknown base\n"
