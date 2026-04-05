"""Tests for SAM3 tool — model loading, inference modes, and draw_path.

Requires GPU with the SAM3.1 checkpoint at:
    /storage/openpsi/models/sam3.1/sam3.1_multiplex.pt
"""

import json
import os

import pytest
import torch

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAM3_CKPT = "/storage/openpsi/models/sam3.1/sam3.1_multiplex.pt"

skip_no_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)
skip_no_ckpt = pytest.mark.skipif(
    not os.path.isfile(SAM3_CKPT), reason=f"Checkpoint not found: {SAM3_CKPT}"
)


def _make_test_image():
    """Create a simple 256x256 RGB test image."""
    from PIL import Image

    return Image.new("RGB", (256, 256), color=(128, 64, 200))


@pytest.fixture(scope="module")
def sam3_model():
    """Build SAM3 image model once for the module."""
    from geo_edit.models.sam3 import build_sam3_image_model

    model = build_sam3_image_model(
        checkpoint_path=SAM3_CKPT,
        device="cuda:0",
        eval_mode=True,
        load_from_HF=False,
    )
    return model


@pytest.fixture(scope="module")
def sam3_processor(sam3_model):
    """Build Sam3Processor once for the module."""
    from geo_edit.models.sam3.model.sam3_image_processor import Sam3Processor

    return Sam3Processor(
        sam3_model,
        device="cuda:0",
        confidence_threshold=0.25,
    )


# ---------------------------------------------------------------------------
# 1. Model loading & dtype sanity
# ---------------------------------------------------------------------------


@skip_no_gpu
@skip_no_ckpt
class TestSam3ModelLoading:
    """Verify model loads to correct device and dtype."""

    def test_model_on_cuda(self, sam3_model):
        param = next(sam3_model.parameters())
        assert param.is_cuda, "Model should be on CUDA"

    def test_model_dtype_float32(self, sam3_model):
        for name, param in sam3_model.named_parameters():
            assert param.dtype == torch.float32, (
                f"Parameter {name} has dtype {param.dtype}, expected float32"
            )
            break  # spot-check first param is enough

    def test_all_params_float32(self, sam3_model):
        """Check that no parameter is bfloat16 after loading."""
        bf16_params = [
            name
            for name, p in sam3_model.named_parameters()
            if p.dtype == torch.bfloat16
        ]
        assert bf16_params == [], f"Found bfloat16 parameters: {bf16_params[:5]}"


# ---------------------------------------------------------------------------
# 2. Backbone forward — dtype consistency
# ---------------------------------------------------------------------------


@skip_no_gpu
@skip_no_ckpt
class TestSam3BackboneForward:
    """Test that backbone forward produces float32 outputs (no bfloat16 leak)."""

    def test_forward_image_dtype(self, sam3_model):
        """forward_image should return float32 features."""
        img = _make_test_image()
        from torchvision.transforms import v2

        transform = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(1008, 1008)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        tensor = v2.functional.to_image(img).to("cuda:0")
        tensor = transform(tensor).unsqueeze(0)

        with torch.inference_mode():
            backbone_out = sam3_model.backbone.forward_image(tensor)

        # Check key features are float32
        for key, val in backbone_out.items():
            if isinstance(val, torch.Tensor):
                assert val.dtype == torch.float32, (
                    f"backbone_out['{key}'] has dtype {val.dtype}, expected float32"
                )
            elif isinstance(val, list):
                for i, v in enumerate(val):
                    if isinstance(v, torch.Tensor):
                        assert v.dtype == torch.float32, (
                            f"backbone_out['{key}'][{i}] has dtype {v.dtype}"
                        )

    def test_forward_text_dtype(self, sam3_model):
        """forward_text should return float32 features."""
        with torch.inference_mode():
            text_out = sam3_model.backbone.forward_text(
                ["test object"], device="cuda:0"
            )

        for key, val in text_out.items():
            if isinstance(val, torch.Tensor) and val.is_floating_point():
                assert val.dtype == torch.float32, (
                    f"text_out['{key}'] has dtype {val.dtype}, expected float32"
                )


# ---------------------------------------------------------------------------
# 3. Processor-level end-to-end tests
# ---------------------------------------------------------------------------


@skip_no_gpu
@skip_no_ckpt
class TestSam3Processor:
    """Test Sam3Processor set_image + set_text_prompt / add_geometric_prompt."""

    def test_set_image(self, sam3_processor):
        img = _make_test_image()
        state = sam3_processor.set_image(img)
        assert "backbone_out" in state
        assert "original_height" in state
        assert state["original_height"] == 256
        assert state["original_width"] == 256

    def test_text_prompt(self, sam3_processor):
        """text_segment path: set_image + set_text_prompt should not crash."""
        img = _make_test_image()
        state = sam3_processor.set_image(img)
        state = sam3_processor.set_text_prompt(prompt="object", state=state)
        # Should have scores (possibly empty, that's ok for a blank image)
        assert "scores" in state
        assert "boxes" in state
        assert "masks" in state

    def test_geometric_prompt(self, sam3_processor):
        """bbox_segment path: set_image + add_geometric_prompt."""
        img = _make_test_image()
        state = sam3_processor.set_image(img)
        # Box in [cx, cy, w, h] normalized to 0-1
        state = sam3_processor.add_geometric_prompt(
            box=[0.5, 0.5, 0.5, 0.5], label=True, state=state
        )
        assert "scores" in state
        assert "boxes" in state


# ---------------------------------------------------------------------------
# 4. SAM3Actor.analyze (full tool path)
# ---------------------------------------------------------------------------


@skip_no_gpu
@skip_no_ckpt
class TestSam3Actor:
    """Test SAM3Actor.analyze for each mode."""

    @pytest.fixture(scope="class")
    def actor(self):
        from geo_edit.tool_definitions.agents.sam3 import SAM3Actor

        return SAM3Actor(model_name=SAM3_CKPT)

    @staticmethod
    def _encode_test_image():
        import base64
        from io import BytesIO

        img = _make_test_image()
        buf = BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def test_auto_segment(self, actor):
        b64 = self._encode_test_image()
        result = actor.analyze(b64, mode="auto")
        data = json.loads(result)
        assert "image_size" in data
        assert "proposals" in data
        assert "error" not in data

    def test_text_segment(self, actor):
        b64 = self._encode_test_image()
        result = actor.analyze(b64, mode="text_segment", text_prompt="object")
        data = json.loads(result)
        assert "image_size" in data
        assert "error" not in data

    def test_concept_count(self, actor):
        b64 = self._encode_test_image()
        result = actor.analyze(b64, mode="concept_count", text_prompt="circle")
        data = json.loads(result)
        assert "count" in data
        assert "error" not in data

    def test_presence_check(self, actor):
        b64 = self._encode_test_image()
        result = actor.analyze(b64, mode="presence_check", text_prompt="cat")
        data = json.loads(result)
        assert "present" in data
        assert "error" not in data

    def test_bbox_segment(self, actor):
        b64 = self._encode_test_image()
        result = actor.analyze(
            b64,
            mode="bbox",
            bounding_box="\\boxed{200,200,800,800}",
        )
        data = json.loads(result)
        assert "image_size" in data
        assert "error" not in data

    def test_exemplar_segment(self, actor):
        b64 = self._encode_test_image()
        result = actor.analyze(
            b64,
            mode="exemplar_segment",
            bounding_box="\\boxed{100,100,500,500}",
        )
        data = json.loads(result)
        assert "image_size" in data
        assert "error" not in data


# ---------------------------------------------------------------------------
# 5. draw_path tool
# ---------------------------------------------------------------------------


class TestDrawPath:
    """Test draw_path function tool."""

    def test_basic_path(self):
        from PIL import Image

        from geo_edit.tool_definitions.functions.draw_path import execute

        img = Image.new("RGB", (200, 200), color=(255, 255, 255))
        image_list = [img]
        result = execute(image_list, 0, "\\boxed{100,100,500,500,900,100}")
        # Should return an Image with path drawn
        assert isinstance(result, Image.Image)
        assert result.size == (200, 200)

    def test_path_pixels_changed(self):
        """Verify draw_path actually modifies the image."""
        from PIL import Image
        import numpy as np

        from geo_edit.tool_definitions.functions.draw_path import execute

        img = Image.new("RGB", (200, 200), color=(255, 255, 255))
        original = np.array(img).copy()
        image_list = [img]
        result = execute(image_list, 0, "\\boxed{100,100,500,500,900,100}")
        modified = np.array(result)
        assert not np.array_equal(original, modified), (
            "draw_path should modify the image pixels"
        )

    def test_invalid_index(self):
        from PIL import Image

        from geo_edit.tool_definitions.functions.draw_path import execute

        result = execute([Image.new("RGB", (10, 10))], 5, "\\boxed{0,0,500,500}")
        assert isinstance(result, str) and "Error" in result

    def test_insufficient_points(self):
        from PIL import Image

        from geo_edit.tool_definitions.functions.draw_path import execute

        result = execute([Image.new("RGB", (10, 10))], 0, "\\boxed{100}")
        assert isinstance(result, str) and "Error" in result
