"""Tests for tool functions - tool_definitions/functions/*.py"""

import pytest
from PIL import Image


def _create_test_image(width=100, height=100, color="white"):
    """Helper to create a test image."""
    return Image.new("RGB", (width, height), color)


class TestImageCrop:
    """Test image_crop tool."""

    def test_crop_returns_image(self):
        from geo_edit.tool_definitions.functions.crop import execute

        img = _create_test_image()
        image_list = [img]
        result = execute(image_list, image_index=0, bounding_box="\\boxed{0,0,500,500}")
        assert isinstance(result, Image.Image)

    def test_crop_invalid_index_returns_error(self):
        from geo_edit.tool_definitions.functions.crop import execute

        img = _create_test_image()
        image_list = [img]
        result = execute(image_list, image_index=5, bounding_box="\\boxed{0,0,500,500}")
        assert isinstance(result, str)
        assert "Error" in result

    def test_crop_negative_index_returns_error(self):
        from geo_edit.tool_definitions.functions.crop import execute

        img = _create_test_image()
        image_list = [img]
        result = execute(image_list, image_index=-1, bounding_box="\\boxed{0,0,500,500}")
        assert isinstance(result, str)
        assert "Error" in result

    def test_crop_declaration_structure(self):
        from geo_edit.tool_definitions.functions.crop import DECLARATION, RETURN_TYPE

        assert DECLARATION["name"] == "image_crop"
        assert "description" in DECLARATION
        assert "parameters" in DECLARATION
        assert "image_index" in DECLARATION["parameters"]["properties"]
        assert "bounding_box" in DECLARATION["parameters"]["properties"]
        assert RETURN_TYPE == "image"


class TestImageLabel:
    """Test image_label tool."""

    def test_label_returns_image(self):
        from geo_edit.tool_definitions.functions.label import execute

        img = _create_test_image()
        image_list = [img]
        result = execute(image_list, image_index=0, text="Test", position="(100,100)")
        assert isinstance(result, Image.Image)

    def test_label_invalid_index_returns_error(self):
        from geo_edit.tool_definitions.functions.label import execute

        img = _create_test_image()
        image_list = [img]
        result = execute(image_list, image_index=5, text="Test", position="(100,100)")
        assert isinstance(result, str)
        assert "Error" in result

    def test_label_declaration_structure(self):
        from geo_edit.tool_definitions.functions.label import DECLARATION, RETURN_TYPE

        assert DECLARATION["name"] == "image_label"
        assert "description" in DECLARATION
        assert "parameters" in DECLARATION
        assert "image_index" in DECLARATION["parameters"]["properties"]
        assert "text" in DECLARATION["parameters"]["properties"]
        assert "position" in DECLARATION["parameters"]["properties"]
        assert RETURN_TYPE == "image"


class TestDrawLine:
    """Test draw_line tool."""

    def test_draw_line_returns_image(self):
        from geo_edit.tool_definitions.functions.draw_line import execute

        img = _create_test_image()
        image_list = [img]
        result = execute(image_list, image_index=0, coordinates="\\boxed{0,0,500,500}")
        assert isinstance(result, Image.Image)

    def test_draw_line_invalid_index_returns_error(self):
        from geo_edit.tool_definitions.functions.draw_line import execute

        img = _create_test_image()
        image_list = [img]
        result = execute(image_list, image_index=5, coordinates="\\boxed{0,0,500,500}")
        assert isinstance(result, str)
        assert "Error" in result

    def test_draw_line_declaration_structure(self):
        from geo_edit.tool_definitions.functions.draw_line import DECLARATION, RETURN_TYPE

        assert DECLARATION["name"] == "draw_line"
        assert "description" in DECLARATION
        assert "parameters" in DECLARATION
        assert "image_index" in DECLARATION["parameters"]["properties"]
        assert "coordinates" in DECLARATION["parameters"]["properties"]
        assert RETURN_TYPE == "image"


class TestBoundingBox:
    """Test bounding_box tool."""

    def test_bbox_returns_image(self):
        from geo_edit.tool_definitions.functions.bbox import execute

        img = _create_test_image()
        image_list = [img]
        result = execute(image_list, image_index=0, bounding_box="\\boxed{100,100,300,300}")
        assert isinstance(result, Image.Image)

    def test_bbox_invalid_index_returns_error(self):
        from geo_edit.tool_definitions.functions.bbox import execute

        img = _create_test_image()
        image_list = [img]
        result = execute(image_list, image_index=5, bounding_box="\\boxed{100,100,300,300}")
        assert isinstance(result, str)
        assert "Error" in result

    def test_bbox_declaration_structure(self):
        from geo_edit.tool_definitions.functions.bbox import DECLARATION, RETURN_TYPE

        assert DECLARATION["name"] == "bounding_box"
        assert "description" in DECLARATION
        assert "parameters" in DECLARATION
        assert "image_index" in DECLARATION["parameters"]["properties"]
        assert "bounding_box" in DECLARATION["parameters"]["properties"]
        assert RETURN_TYPE == "image"


class TestImageHighlight:
    """Test image_highlight tool."""

    def test_highlight_returns_image(self):
        from geo_edit.tool_definitions.functions.highlight import execute

        img = _create_test_image()
        image_list = [img]
        result = execute(image_list, image_index=0, bounding_box="\\boxed{100,100,300,300}")
        assert isinstance(result, Image.Image)

    def test_highlight_invalid_index_returns_error(self):
        from geo_edit.tool_definitions.functions.highlight import execute

        img = _create_test_image()
        image_list = [img]
        result = execute(image_list, image_index=5, bounding_box="\\boxed{100,100,300,300}")
        assert isinstance(result, str)
        assert "Error" in result

    def test_highlight_converts_to_rgb(self):
        from geo_edit.tool_definitions.functions.highlight import execute

        img = _create_test_image()
        image_list = [img]
        result = execute(image_list, image_index=0, bounding_box="\\boxed{100,100,300,300}")
        assert result.mode == "RGB"

    def test_highlight_declaration_structure(self):
        from geo_edit.tool_definitions.functions.highlight import DECLARATION, RETURN_TYPE

        assert DECLARATION["name"] == "image_highlight"
        assert "description" in DECLARATION
        assert "parameters" in DECLARATION
        assert "image_index" in DECLARATION["parameters"]["properties"]
        assert "bounding_box" in DECLARATION["parameters"]["properties"]
        assert RETURN_TYPE == "image"


class TestAllToolDeclarations:
    """Test all tool declarations have consistent structure."""

    def test_all_declarations_have_name_field(self):
        from geo_edit.tool_definitions.functions import FUNCTION_TOOLS

        for name, (decl, _, _, _) in FUNCTION_TOOLS.items():
            assert decl["name"] == name, f"Tool {name} has mismatched name in declaration"

    def test_all_declarations_have_required_params(self):
        from geo_edit.tool_definitions.functions import FUNCTION_TOOLS

        for name, (decl, _, _, _) in FUNCTION_TOOLS.items():
            params = decl["parameters"]
            assert "type" in params
            assert params["type"] == "object"
            assert "properties" in params
            assert "required" in params
            assert "image_index" in params["required"]
