"""Tests for utils/ module - text_utils.py and stats.py."""

import pytest
import math


class TestExtractAnswer:
    """Test extract_answer function."""

    def test_extract_answer_split_mode(self):
        from geo_edit.utils.text_utils import extract_answer

        text = "Some text <answer>The answer is 42</answer> more text"
        result = extract_answer(text, mode="split")
        assert result == "The answer is 42"

    def test_extract_answer_split_mode_returns_last_match(self):
        from geo_edit.utils.text_utils import extract_answer

        text = "<answer>First</answer> <answer>Second</answer>"
        result = extract_answer(text, mode="split")
        assert result == "Second"

    def test_extract_answer_strict_mode(self):
        from geo_edit.utils.text_utils import extract_answer

        text = "Some text <answer>The answer</answer> more"
        result = extract_answer(text, mode="strict")
        assert result == "The answer"

    def test_extract_answer_returns_none_when_no_tags(self):
        from geo_edit.utils.text_utils import extract_answer

        text = "No tags here"
        result = extract_answer(text, mode="split")
        assert result is None

    def test_extract_answer_invalid_mode_raises_error(self):
        from geo_edit.utils.text_utils import extract_answer

        with pytest.raises(ValueError, match="Unknown extract mode"):
            extract_answer("text", mode="invalid")


class TestGetFinalPrediction:
    """Test get_final_prediction function."""

    def test_get_final_prediction_returns_last_item(self):
        from geo_edit.utils.text_utils import get_final_prediction

        result = get_final_prediction(["first", "second", "third"], extract_mode=None)
        assert result == "third"

    def test_get_final_prediction_with_extract_mode(self):
        from geo_edit.utils.text_utils import get_final_prediction

        result = get_final_prediction(
            ["ignored", "<answer>extracted</answer>"],
            extract_mode="split"
        )
        assert result == "extracted"

    def test_get_final_prediction_empty_list(self):
        from geo_edit.utils.text_utils import get_final_prediction

        result = get_final_prediction([], extract_mode=None)
        assert result == ""

    def test_get_final_prediction_fallback_when_no_tags(self):
        from geo_edit.utils.text_utils import get_final_prediction

        result = get_final_prediction(["no tags here"], extract_mode="split")
        assert result == "no tags here"


class TestParseScore:
    """Test parse_score function."""

    def test_parse_score_finds_score_1(self):
        from geo_edit.utils.text_utils import parse_score

        result = parse_score("The score: 1 is correct")
        assert result == "1"

    def test_parse_score_finds_score_0(self):
        from geo_edit.utils.text_utils import parse_score

        result = parse_score("Score: 0")
        assert result == "0"

    def test_parse_score_case_insensitive(self):
        from geo_edit.utils.text_utils import parse_score

        result = parse_score("SCORE: 1")
        assert result == "1"

    def test_parse_score_returns_empty_when_no_match(self):
        from geo_edit.utils.text_utils import parse_score

        result = parse_score("no score here")
        assert result == ""


class TestExtractChoiceLetter:
    """Test extract_choice_letter function."""

    def test_extract_choice_letter_simple(self):
        from geo_edit.utils.text_utils import extract_choice_letter

        result = extract_choice_letter("A")
        assert result == "A"

    def test_extract_choice_letter_with_period(self):
        from geo_edit.utils.text_utils import extract_choice_letter

        result = extract_choice_letter("B. This is option B")
        assert result == "B"

    def test_extract_choice_letter_answer_is_pattern(self):
        from geo_edit.utils.text_utils import extract_choice_letter

        result = extract_choice_letter("The answer is C")
        assert result == "C"

    def test_extract_choice_letter_returns_uppercase(self):
        from geo_edit.utils.text_utils import extract_choice_letter

        result = extract_choice_letter("answer is d")
        assert result == "D"

    def test_extract_choice_letter_returns_none_when_no_match(self):
        from geo_edit.utils.text_utils import extract_choice_letter

        result = extract_choice_letter("no choice letter here 123")
        assert result is None


class TestCleanResponse:
    """Test clean_response function."""

    def test_clean_response_removes_prefix(self):
        from geo_edit.utils.text_utils import clean_response

        result = clean_response("Based on the image, the answer is 42")
        assert result == "the answer is 42"

    def test_clean_response_truncates_long_text(self):
        from geo_edit.utils.text_utils import clean_response

        long_text = "word " * 100
        result = clean_response(long_text, max_length=50)
        assert len(result) <= 53  # 50 + "..."


class TestCalculateConfidenceScore:
    """Test calculate_confidence_score function."""

    def test_calculate_confidence_score_empty_list(self):
        from geo_edit.utils.text_utils import calculate_confidence_score

        result = calculate_confidence_score([])
        assert result == 0.5

    def test_calculate_confidence_score_returns_bounded_value(self):
        from geo_edit.utils.text_utils import calculate_confidence_score

        result = calculate_confidence_score([-1.0, -2.0, -3.0])
        assert 0.0 <= result <= 1.0


class TestGetOutputTokensTotal:
    """Test get_output_tokens_total function."""

    def test_get_output_tokens_total_direct_field(self):
        from geo_edit.utils.stats import get_output_tokens_total

        item = {"tokens_output_total": 100}
        result = get_output_tokens_total(item)
        assert result == 100.0

    def test_get_output_tokens_total_from_per_step(self):
        from geo_edit.utils.stats import get_output_tokens_total

        item = {"tokens_used_per_step": [10, 20, 30]}
        result = get_output_tokens_total(item)
        assert result == 60.0

    def test_get_output_tokens_total_returns_none_when_missing(self):
        from geo_edit.utils.stats import get_output_tokens_total

        item = {}
        result = get_output_tokens_total(item)
        assert result is None


class TestGetTotalTokens:
    """Test get_total_tokens function."""

    def test_get_total_tokens_direct_field(self):
        from geo_edit.utils.stats import get_total_tokens

        item = {"tokens_used_total": 500}
        result = get_total_tokens(item)
        assert result == 500.0

    def test_get_total_tokens_from_per_step(self):
        from geo_edit.utils.stats import get_total_tokens

        item = {"tokens_total_per_step": [100, 200, 300]}
        result = get_total_tokens(item)
        assert result == 600.0


class TestComputeToolCombinationStatistics:
    """Test compute_tool_combination_statistics function."""

    def test_compute_tool_combination_statistics_basic(self):
        from geo_edit.utils.stats import compute_tool_combination_statistics

        eval_results = [
            {"result": 1.0, "function_call_each_count": {"tool_a": 1}},
            {"result": 0.0, "function_call_each_count": {"tool_a": 1}},
            {"result": 1.0, "function_call_each_count": {}},
        ]
        result = compute_tool_combination_statistics(eval_results)
        assert "tool_a" in result
        assert "no_tool" in result

    def test_compute_tool_combination_statistics_skips_filtered(self):
        from geo_edit.utils.stats import compute_tool_combination_statistics

        eval_results = [
            {"result": {"is_filter": True}, "function_call_each_count": {"tool_a": 1}},
            {"result": 1.0, "function_call_each_count": {}},
        ]
        result = compute_tool_combination_statistics(eval_results)
        assert "no_tool" in result


class TestAggregateMetaInfo:
    """Test aggregate_meta_info function."""

    def test_aggregate_meta_info_counts_tasks(self):
        from geo_edit.utils.stats import aggregate_meta_info

        meta_info_list = [
            {"id": "task1", "function_call_total_count": 2},
            {"id": "task2", "function_call_total_count": 3},
            {"id": "task1", "function_call_total_count": 1},  # Duplicate task
        ]
        result = aggregate_meta_info(meta_info_list)
        assert result["total_tasks"] == 2
        assert result["total_trajectories"] == 3
        assert result["total_tool_calls"] == 6

    def test_aggregate_meta_info_counts_direct_answers(self):
        from geo_edit.utils.stats import aggregate_meta_info

        meta_info_list = [
            {"id": "task1", "function_call_total_count": 0},
            {"id": "task2", "function_call_total_count": 1},
        ]
        result = aggregate_meta_info(meta_info_list)
        assert result["direct_answer_count"] == 1

    def test_aggregate_meta_info_sums_tokens(self):
        from geo_edit.utils.stats import aggregate_meta_info

        meta_info_list = [
            {"id": "task1", "tokens_used_total": 100},
            {"id": "task2", "tokens_used_total": 200},
        ]
        result = aggregate_meta_info(meta_info_list)
        assert result["total_tokens"] == 300.0
