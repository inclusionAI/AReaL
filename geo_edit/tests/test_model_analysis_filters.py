from geo_edit.environment.action.model_analysis_tool import (
    _contains_final_answer,
    _extract_analysis_from_json,
)


def test_contains_final_answer_patterns():
    assert _contains_final_answer("<answer>42</answer>")
    assert _contains_final_answer("Final answer: 7")
    assert _contains_final_answer("答案是 3")
    assert _contains_final_answer(r"\boxed{9}")
    assert not _contains_final_answer("step 1: read the axis and compare trends")


def test_extract_analysis_from_json():
    payload = '{"analysis":"compare slope then estimate value","final_answer_provided":false}'
    assert _extract_analysis_from_json(payload) == "compare slope then estimate value"
    assert _extract_analysis_from_json("not-json") is None
    assert _extract_analysis_from_json('{"foo":"bar"}') is None
