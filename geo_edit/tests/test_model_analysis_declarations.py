from geo_edit.environment.action.model_analysis_tool import (
    bbox_agent_function_declaration,
    chartmoe_function_declaration,
    gllava_function_declaration,
    multimath_function_declaration,
)


def _assert_base_contract(decl: dict):
    assert decl["name"]
    assert "description" in decl
    desc = decl["description"].lower()
    assert "final answer" in desc
    assert "analysis" in desc or "bounding box" in desc
    params = decl["parameters"]
    assert params["type"] == "object"
    assert "image_index" in params["required"]
    assert "image_index" in params["properties"]


def test_multimath_declaration_contract():
    _assert_base_contract(multimath_function_declaration)
    props = multimath_function_declaration["parameters"]["properties"]
    assert "question" in props
    assert set(props.keys()) == {"image_index", "question"}


def test_gllava_declaration_contract():
    _assert_base_contract(gllava_function_declaration)
    props = gllava_function_declaration["parameters"]["properties"]
    assert "question" in props
    assert set(props.keys()) == {"image_index", "question"}


def test_chartmoe_declaration_contract():
    _assert_base_contract(chartmoe_function_declaration)
    props = chartmoe_function_declaration["parameters"]["properties"]
    assert "question" in props
    assert set(props.keys()) == {"image_index", "question"}


def test_bbox_agent_declaration_contract():
    _assert_base_contract(bbox_agent_function_declaration)
    props = bbox_agent_function_declaration["parameters"]["properties"]
    assert set(props.keys()) == {"image_index", "question"}
