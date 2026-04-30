"""Tests for Megatron engine VLM (Vision-Language Model) support.

Unit tests (CPU-only) test the helper functions and logic changes.
Integration tests (GPU-required) run via torchrun as subprocesses to avoid
device memory leaks in the parent pytest process, allowing the full
suite to run on just 2 GPUs.
"""

import os
import pathlib
import subprocess
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

_TORCHRUN_SCRIPT = (
    pathlib.Path(__file__).parent / "torchrun" / "run_megatron_engine_vlm.py"
).resolve()

CUDA_AVAILABLE = torch.cuda.is_available()


# ──────────────────────────────────────────────────────────────────────
# Unit tests: CPU-only, no GPU or model weights required
# ──────────────────────────────────────────────────────────────────────


class TestUnwrapToGptModel:
    """Test unwrap_to_gpt_model with VLM model structures."""

    def test_plain_gpt_model_unwraps(self):
        """Plain GPTModel (no wrapping) should return itself."""
        from megatron.core.models.gpt.gpt_model import GPTModel

        from areal.models.mcore.registry import unwrap_to_gpt_model

        mock_gpt = MagicMock(spec=GPTModel)
        mock_gpt.__class__ = GPTModel
        result = unwrap_to_gpt_model(mock_gpt)
        assert result is mock_gpt

    def test_ddp_wrapped_gpt_model_unwraps(self):
        """GPTModel wrapped in DDP (via .module) should unwrap correctly."""
        from megatron.core.models.gpt.gpt_model import GPTModel

        from areal.models.mcore.registry import unwrap_to_gpt_model

        mock_gpt = MagicMock(spec=GPTModel)
        mock_gpt.__class__ = GPTModel
        wrapper = MagicMock()
        wrapper.__class__ = type("DDPWrapper", (), {})
        wrapper.module = mock_gpt
        result = unwrap_to_gpt_model(wrapper)
        assert result is mock_gpt

    def test_vlm_model_with_language_model_unwraps(self):
        """VLM model (e.g. Qwen2_5VLModel) with .language_model attribute unwraps."""
        from megatron.core.models.gpt.gpt_model import GPTModel

        from areal.models.mcore.registry import unwrap_to_gpt_model

        mock_gpt = MagicMock(spec=GPTModel)
        mock_gpt.__class__ = GPTModel

        # VLM model: no .module, but has .language_model
        vlm_model = MagicMock()
        vlm_model.__class__ = type("Qwen2_5VLModel", (), {})
        del vlm_model.module  # ensure no .module attribute
        vlm_model.language_model = mock_gpt

        result = unwrap_to_gpt_model(vlm_model)
        assert result is mock_gpt

    def test_ddp_wrapped_vlm_model_unwraps(self):
        """VLM model wrapped in DDP should unwrap to language_model."""
        from megatron.core.models.gpt.gpt_model import GPTModel

        from areal.models.mcore.registry import unwrap_to_gpt_model

        mock_gpt = MagicMock(spec=GPTModel)
        mock_gpt.__class__ = GPTModel

        vlm_model = MagicMock()
        vlm_model.__class__ = type("Qwen2_5VLModel", (), {})
        del vlm_model.module
        vlm_model.language_model = mock_gpt

        wrapper = MagicMock()
        wrapper.__class__ = type("DDPWrapper", (), {})
        wrapper.module = vlm_model

        result = unwrap_to_gpt_model(wrapper)
        assert result is mock_gpt

    def test_unsupported_model_raises_type_error(self):
        """Model with neither GPTModel nor .language_model should raise TypeError."""
        from areal.models.mcore.registry import unwrap_to_gpt_model

        unsupported = MagicMock()
        unsupported.__class__ = type("RandomModel", (), {})
        del unsupported.module
        del unsupported.language_model

        with pytest.raises(TypeError, match="could not be unwrapped"):
            unwrap_to_gpt_model(unsupported)


class TestExtractVisionFromMultiModal:
    """Test _extract_vision_from_multi_modal helper."""

    def test_extracts_pixel_values_and_grid(self):
        """Vision tensors should land on padded_mb only, not duplicated on mb."""
        from areal.engine.megatron_utils.packed_context_parallel import (
            extract_vision_from_multi_modal,
        )

        mb: dict[str, Any] = {
            "multi_modal_input": [
                {
                    "pixel_values": torch.randn(10, 3, 14, 14),
                    "image_grid_thw": torch.tensor([[1, 2, 2]]),
                },
                {
                    "pixel_values": torch.randn(5, 3, 14, 14),
                    "image_grid_thw": torch.tensor([[1, 1, 1]]),
                },
            ]
        }
        padded_mb: dict[str, Any] = dict(mb)

        extract_vision_from_multi_modal(mb, padded_mb)

        # Forward side (padded_mb) gets the concatenated vision tensors.
        assert padded_mb["pixel_values"].shape[0] == 15  # 10 + 5
        assert padded_mb["image_grid_thw"].shape[0] == 2  # 2 grids
        # Loss side (mb) does not carry them.
        assert "pixel_values" not in mb
        assert "image_grid_thw" not in mb
        # multi_modal_input is consumed and removed from both sides.
        assert "multi_modal_input" not in mb
        assert "multi_modal_input" not in padded_mb

    def test_handles_video_grid_thw(self):
        """Should extract video_grid_thw onto padded_mb."""
        from areal.engine.megatron_utils.packed_context_parallel import (
            extract_vision_from_multi_modal,
        )

        mb: dict[str, Any] = {
            "multi_modal_input": [
                {
                    "pixel_values": torch.randn(4, 3, 14, 14),
                    "image_grid_thw": torch.tensor([[1, 2, 2]]),
                    "video_grid_thw": torch.tensor([[2, 2, 2]]),
                },
            ]
        }
        padded_mb: dict[str, Any] = dict(mb)

        extract_vision_from_multi_modal(mb, padded_mb)

        assert padded_mb["video_grid_thw"].shape[0] == 1
        assert "video_grid_thw" not in mb

    def test_no_multi_modal_input_is_noop(self):
        """Should do nothing if multi_modal_input not in dict."""
        from areal.engine.megatron_utils.packed_context_parallel import (
            extract_vision_from_multi_modal,
        )

        mb: dict[str, Any] = {"input_ids": torch.tensor([1, 2, 3])}
        padded_mb: dict[str, Any] = dict(mb)

        extract_vision_from_multi_modal(mb, padded_mb)

        assert "pixel_values" not in mb
        assert "image_grid_thw" not in mb

    def test_empty_multi_modal_input_is_noop(self):
        """Should not add keys if multi_modal_input items have no vision data,
        but should still pop multi_modal_input itself."""
        from areal.engine.megatron_utils.packed_context_parallel import (
            extract_vision_from_multi_modal,
        )

        mb: dict[str, Any] = {"multi_modal_input": [{}]}
        padded_mb: dict[str, Any] = dict(mb)

        extract_vision_from_multi_modal(mb, padded_mb)

        assert "pixel_values" not in mb
        assert "pixel_values" not in padded_mb
        assert "multi_modal_input" not in mb
        assert "multi_modal_input" not in padded_mb

    def test_falls_back_to_padded_mb(self):
        """When mb lacks multi_modal_input but padded_mb has it, the fallback
        branch should still concatenate vision tensors onto padded_mb."""
        from areal.engine.megatron_utils.packed_context_parallel import (
            extract_vision_from_multi_modal,
        )

        pixel_values = [torch.randn(2, 4)]
        mb: dict[str, Any] = {"input_ids": torch.ones(2, dtype=torch.long)}
        padded_mb: dict[str, Any] = {
            "input_ids": torch.ones(2, dtype=torch.long),
            "multi_modal_input": [{"pixel_values": pixel_values[0]}],
        }

        extract_vision_from_multi_modal(mb, padded_mb)

        assert "multi_modal_input" not in mb
        assert "multi_modal_input" not in padded_mb
        assert "pixel_values" not in mb
        assert torch.equal(padded_mb["pixel_values"], torch.cat(pixel_values, dim=0))


class TestPrepareMbListRebindCallerSafety:
    """Verify _prepare_mb_list rebinds mb_list.data to a filtered copy so the
    caller's input dict survives across repeated forward() calls.

    `split_padded_tensor_dict_into_mb_list` constructs `MicroBatchList(data=input_)`
    where `mb_list.data` is the *same object* as the caller's input dict (not a
    copy). An earlier draft did `_drop_multi_modal_payload(mb_list.data)` in-place,
    which mutated the caller's dict and broke `engine.forward(input_)` when the
    same `input_` was forwarded a second time (e.g. the save/load round-trip
    test). The current implementation rebinds `mb_list.data = {filtered copy}`
    instead — this test pins that contract.
    """

    def test_rebind_does_not_mutate_caller_input(self):
        from areal.engine.megatron_utils.packed_context_parallel import (
            _is_multi_modal_payload_key,
        )
        from areal.utils.data import MicroBatchList, MicroBatchSpec

        input_ = {
            "input_ids": torch.ones(4, dtype=torch.long),
            "multi_modal_input": [{"pixel_values": torch.randn(2, 4)}],
            "pixel_values": torch.randn(2, 4),
            "image_grid_thw": torch.tensor([[1, 1, 2]]),
        }
        snapshot_keys = set(input_.keys())

        mb_list = MicroBatchList(
            data=input_,
            mb_spec=MicroBatchSpec(),
            mbs=[],
            group_lens=[],
        )
        assert mb_list.data is input_, (
            "MicroBatchList.data must alias caller's input dict to mirror "
            "split_padded_tensor_dict_into_mb_list semantics."
        )

        mb_list.data = {
            k: v for k, v in mb_list.data.items() if not _is_multi_modal_payload_key(k)
        }

        assert set(input_.keys()) == snapshot_keys, (
            "Rebind must not mutate the caller's input dict — regression to "
            "in-place `_drop_multi_modal_payload(mb_list.data)` would fail here."
        )
        assert "multi_modal_input" not in mb_list.data
        assert "pixel_values" not in mb_list.data
        assert "image_grid_thw" not in mb_list.data
        assert "input_ids" in mb_list.data
        assert mb_list.data is not input_


class TestVisionModelDetection:
    """Test is_valid_vision_model detection."""

    def test_qwen2_5_vl_detected(self):
        from areal.engine.core.model import is_valid_vision_model

        assert is_valid_vision_model("qwen2_5_vl") is True

    def test_qwen2_vl_detected(self):
        from areal.engine.core.model import is_valid_vision_model

        assert is_valid_vision_model("qwen2_vl") is True

    def test_qwen3_not_detected(self):
        from areal.engine.core.model import is_valid_vision_model

        assert is_valid_vision_model("qwen3") is False

    def test_text_model_not_detected(self):
        from areal.engine.core.model import is_valid_vision_model

        assert is_valid_vision_model("llama") is False


class TestConvertQwen25VLToHF:
    """Test mcore→HF weight name conversion for Qwen2.5-VL."""

    @pytest.fixture()
    def tf_config(self):
        """Mock TransformerConfig with Qwen2.5-VL-3B values."""
        cfg = MagicMock()
        cfg.hidden_size = 2048
        cfg.num_attention_heads = 16
        cfg.num_query_groups = 2
        cfg.kv_channels = 128
        return cfg

    @pytest.fixture()
    def hf_config(self):
        """Mock HF config with vision_config for Qwen2.5-VL-3B."""
        cfg = MagicMock()
        cfg.vision_config.num_heads = 16
        cfg.vision_config.hidden_size = 1280
        return cfg

    # --- Registry dispatch ---

    def test_registry_dispatches_qwen2_5_vl_before_qwen2(self, tf_config, hf_config):
        """convert_to_hf with model_name='qwen2_5_vl' must use the VLM converter."""
        from areal.engine.megatron_utils.megatron import convert_to_hf

        param = torch.randn(2048)
        result = convert_to_hf(
            tf_config,
            "qwen2_5_vl",
            "module.module.vision_model.decoder.final_layernorm.weight",
            param,
            hf_config=hf_config,
        )
        assert result[0][0] == "visual.merger.ln_q.weight"

    # --- Language model delegation ---

    def test_language_model_embedding_delegates(self, tf_config):
        """language_model.embedding should strip prefix and delegate to qwen2."""
        from areal.engine.megatron_utils.megatron import convert_qwen2_5_vl_to_hf

        param = torch.randn(151936, 2048)
        result = convert_qwen2_5_vl_to_hf(
            tf_config,
            "module.module.language_model.embedding.word_embeddings.weight",
            param,
        )
        assert len(result) == 1
        assert result[0][0] == "model.embed_tokens.weight"

    def test_language_model_output_layer_delegates(self, tf_config):
        """language_model.output_layer should map to lm_head."""
        from areal.engine.megatron_utils.megatron import convert_qwen2_5_vl_to_hf

        param = torch.randn(151936, 2048)
        result = convert_qwen2_5_vl_to_hf(
            tf_config,
            "module.module.language_model.output_layer.weight",
            param,
        )
        assert result[0][0] == "lm_head.weight"

    def test_language_model_decoder_layer_qkv_delegates(self, tf_config):
        """language_model decoder QKV should split into q/k/v projections."""
        from areal.engine.megatron_utils.megatron import convert_qwen2_5_vl_to_hf

        # Qwen2.5-VL-3B: num_query_groups=2, value_num_per_group=8, kv_channels=128
        # QKV weight shape: (num_query_groups * (value_num_per_group + 2) * kv_channels, hidden)
        qkv_dim = tf_config.num_query_groups * (8 + 1 + 1) * tf_config.kv_channels
        param = torch.randn(qkv_dim, tf_config.hidden_size)
        result = convert_qwen2_5_vl_to_hf(
            tf_config,
            "module.module.language_model.decoder.layers.5.self_attention.linear_qkv.weight",
            param,
        )
        names = [n for n, _ in result]
        assert "model.layers.5.self_attn.q_proj.weight" in names
        assert "model.layers.5.self_attn.k_proj.weight" in names
        assert "model.layers.5.self_attn.v_proj.weight" in names

    def test_language_model_mlp_fc1_splits_gate_up(self, tf_config):
        """language_model MLP fc1 should split into gate_proj and up_proj."""
        from areal.engine.megatron_utils.megatron import convert_qwen2_5_vl_to_hf

        param = torch.randn(22016, 2048)  # gate + up stacked
        result = convert_qwen2_5_vl_to_hf(
            tf_config,
            "module.module.language_model.decoder.layers.0.mlp.linear_fc1.weight",
            param,
        )
        names = [n for n, _ in result]
        assert "model.layers.0.mlp.gate_proj.weight" in names
        assert "model.layers.0.mlp.up_proj.weight" in names

    # --- Vision model direct mappings ---

    def test_vision_patch_embed(self, tf_config):
        """vision_model.patch_embed maps to visual.patch_embed."""
        from areal.engine.megatron_utils.megatron import convert_qwen2_5_vl_to_hf

        param = torch.randn(1280, 3, 2, 14, 14)
        result = convert_qwen2_5_vl_to_hf(
            tf_config,
            "module.module.vision_model.patch_embed.proj.weight",
            param,
        )
        assert result == [("visual.patch_embed.proj.weight", param)]

    def test_vision_merger_ln(self, tf_config):
        """vision final layernorm maps to visual.merger.ln_q."""
        from areal.engine.megatron_utils.megatron import convert_qwen2_5_vl_to_hf

        param = torch.randn(1280)
        result = convert_qwen2_5_vl_to_hf(
            tf_config,
            "module.module.vision_model.decoder.final_layernorm.weight",
            param,
        )
        assert result[0][0] == "visual.merger.ln_q.weight"

    def test_vision_merger_mlp(self, tf_config):
        """vision projection MLP maps to visual.merger.mlp."""
        from areal.engine.megatron_utils.megatron import convert_qwen2_5_vl_to_hf

        param = torch.randn(5120, 5120)
        result = convert_qwen2_5_vl_to_hf(
            tf_config,
            "module.module.vision_model.projection.encoder.linear_fc1.weight",
            param,
        )
        assert result[0][0] == "visual.merger.mlp.0.weight"

    # --- Vision model per-layer params ---

    def test_vision_layer_attn_qkv(self, tf_config, hf_config):
        """Vision attention QKV maps to visual.blocks.<idx>.attn.qkv with reordering."""
        from areal.engine.megatron_utils.megatron import convert_qwen2_5_vl_to_hf

        param = torch.randn(3840, 1280)
        result = convert_qwen2_5_vl_to_hf(
            tf_config,
            "module.module.vision_model.decoder.layers.7.self_attention.linear_qkv.weight",
            param,
            hf_config=hf_config,
        )
        assert result[0][0] == "visual.blocks.7.attn.qkv.weight"
        assert result[0][1].shape == param.shape  # same shape, different ordering

    def test_vision_layer_attn_proj(self, tf_config):
        """Vision attention proj maps to visual.blocks.<idx>.attn.proj."""
        from areal.engine.megatron_utils.megatron import convert_qwen2_5_vl_to_hf

        param = torch.randn(1280, 1280)
        result = convert_qwen2_5_vl_to_hf(
            tf_config,
            "module.module.vision_model.decoder.layers.3.self_attention.linear_proj.weight",
            param,
        )
        assert result[0][0] == "visual.blocks.3.attn.proj.weight"

    def test_vision_layer_norm1(self, tf_config):
        """Vision input layernorm maps to visual.blocks.<idx>.norm1."""
        from areal.engine.megatron_utils.megatron import convert_qwen2_5_vl_to_hf

        param = torch.randn(1280)
        result = convert_qwen2_5_vl_to_hf(
            tf_config,
            "module.module.vision_model.decoder.layers.0.self_attention.linear_qkv.layer_norm_weight",
            param,
        )
        assert result[0][0] == "visual.blocks.0.norm1.weight"

    def test_vision_layer_mlp_fc1_splits_gate_up(self, tf_config):
        """Vision MLP fc1 splits into gate_proj and up_proj."""
        from areal.engine.megatron_utils.megatron import convert_qwen2_5_vl_to_hf

        param = torch.randn(6840, 1280)  # gate + up stacked
        result = convert_qwen2_5_vl_to_hf(
            tf_config,
            "module.module.vision_model.decoder.layers.2.mlp.linear_fc1.weight",
            param,
        )
        names = [n for n, _ in result]
        assert "visual.blocks.2.mlp.gate_proj.weight" in names
        assert "visual.blocks.2.mlp.up_proj.weight" in names

    def test_vision_layer_mlp_fc2(self, tf_config):
        """Vision MLP fc2 maps to visual.blocks.<idx>.mlp.down_proj."""
        from areal.engine.megatron_utils.megatron import convert_qwen2_5_vl_to_hf

        param = torch.randn(1280, 3420)
        result = convert_qwen2_5_vl_to_hf(
            tf_config,
            "module.module.vision_model.decoder.layers.10.mlp.linear_fc2.weight",
            param,
        )
        assert result[0][0] == "visual.blocks.10.mlp.down_proj.weight"

    def test_vision_layer_norm2(self, tf_config):
        """Vision post-attention layernorm maps to visual.blocks.<idx>.norm2."""
        from areal.engine.megatron_utils.megatron import convert_qwen2_5_vl_to_hf

        param = torch.randn(1280)
        result = convert_qwen2_5_vl_to_hf(
            tf_config,
            "module.module.vision_model.decoder.layers.1.mlp.linear_fc1.layer_norm_weight",
            param,
        )
        assert result[0][0] == "visual.blocks.1.norm2.weight"

    # --- Error handling ---

    def test_unknown_param_raises_value_error(self, tf_config):
        """Unknown parameter name should raise ValueError."""
        from areal.engine.megatron_utils.megatron import convert_qwen2_5_vl_to_hf

        with pytest.raises(ValueError, match="Unknown parameter name"):
            convert_qwen2_5_vl_to_hf(
                tf_config,
                "module.module.some_unknown.weight",
                torch.randn(10),
            )


class TestRemovePaddingVLM:
    """Test remove_padding handles VLM language_model prefixed params."""

    def test_vlm_embedding_padding_removed(self):
        """VLM language_model embedding should be trimmed to vocab_size."""
        from areal.engine.megatron_utils.megatron import remove_padding

        vocab_size = 151936
        padded = torch.randn(vocab_size + 64, 2048)  # padded for TP alignment
        result = remove_padding(
            "module.module.language_model.embedding.word_embeddings.weight",
            padded,
            vocab_size,
        )
        assert result.shape[0] == vocab_size

    def test_vlm_output_layer_padding_removed(self):
        """VLM language_model output_layer should be trimmed to vocab_size."""
        from areal.engine.megatron_utils.megatron import remove_padding

        vocab_size = 151936
        padded = torch.randn(vocab_size + 64, 2048)
        result = remove_padding(
            "module.module.language_model.output_layer.weight",
            padded,
            vocab_size,
        )
        assert result.shape[0] == vocab_size

    def test_non_embedding_param_unchanged(self):
        """Non-embedding params should pass through unchanged."""
        from areal.engine.megatron_utils.megatron import remove_padding

        param = torch.randn(1280)
        result = remove_padding(
            "module.module.vision_model.decoder.final_layernorm.weight",
            param,
            151936,
        )
        assert result is param


# ──────────────────────────────────────────────────────────────────────
# Integration tests: subprocess-based to avoid device memory leaks
# ──────────────────────────────────────────────────────────────────────


def _run_vlm_test(
    test_type: str,
    output_path: str,
    *,
    backend: str = "megatron:d1p1t1",
    nproc: int = 1,
    extra_args: list[str] | None = None,
    timeout: int = 300,
):
    """Launch a VLM integration test via torchrun subprocess.

    Each test runs in its own process, ensuring complete device memory
    cleanup between tests. This allows the full suite to run on just 2 devices.
    """
    from areal.utils.network import find_free_ports

    port = find_free_ports(1)[0]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(nproc))

    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        "--nnodes=1",
        "--master-addr=localhost",
        f"--master_port={port}",
        str(_TORCHRUN_SCRIPT),
        f"--backend={backend}",
        f"--test_type={test_type}",
        f"--output={output_path}",
    ]
    if extra_args:
        cmd.extend(extra_args)

    try:
        subprocess.run(
            cmd,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as e:
        if "out of memory" in (e.stderr or "").lower():
            pytest.skip(f"OOM: 3B VLM {test_type} requires more memory")
        pytest.fail(
            f"VLM {test_type} test failed:\nstdout: {e.stdout}\nstderr: {e.stderr}"
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"VLM {test_type} test timed out ({timeout}s)")

    with open(output_path) as f:
        result = f.read().strip()
    if result == "OOM":
        pytest.skip(f"OOM: 3B VLM {test_type} requires more memory than single device")
    assert result == "Passed", f"VLM {test_type} test failed: {result}"


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_vlm_engine_initializes(tmp_path_factory):
    """Verify VLM engine detects vision model and loads processor."""
    output = str(tmp_path_factory.mktemp("vlm_test") / "init.out")
    _run_vlm_test("init", output)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_vlm_simple_forward(tmp_path_factory):
    """Verify forward pass with VLM inputs completes."""
    output = str(tmp_path_factory.mktemp("vlm_test") / "forward.out")
    _run_vlm_test("forward", output)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_vlm_hf_save_load_weights(tmp_path_factory):
    """Verify save/load preserves VLM weights and saves processor."""
    save_dir = str(tmp_path_factory.mktemp("vlm_save"))
    output = str(tmp_path_factory.mktemp("vlm_test") / "save_load.out")
    _run_vlm_test("save_load", output, extra_args=[f"--save_dir={save_dir}"])


@pytest.mark.gpu
@pytest.mark.multi_gpu
@pytest.mark.slow
@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
def test_vlm_train_tensor_parallel(tmp_path_factory):
    """VLM training with TP=2 to avoid single-device OOM."""
    if torch.cuda.device_count() < 2:
        pytest.skip("VLM TP training requires at least 2 GPUs")
    output = str(tmp_path_factory.mktemp("vlm_test") / "train_tp2.out")
    _run_vlm_test("train", output, backend="megatron:d1p1t2", nproc=2)
