"""Tests for evaluation parsers - eval_shortest_path.py and eval_stmf_counting.py."""

import pytest


class TestParsePathFunction:
    """Test parse_path function from eval_shortest_path.py."""

    def test_parse_path_with_answer_tags(self):
        from geo_edit.evaluation.eval_shortest_path import parse_path

        text = "<answer>A, B, C, D</answer>"
        result = parse_path(text)
        assert result == "A,B,C,D"

    def test_parse_path_with_lowercase_answer_tags(self):
        from geo_edit.evaluation.eval_shortest_path import parse_path

        text = "<ANSWER>x, y, z</ANSWER>"
        result = parse_path(text)
        assert result == "X,Y,Z"

    def test_parse_path_without_tags(self):
        from geo_edit.evaluation.eval_shortest_path import parse_path

        text = "A, B, C"
        result = parse_path(text)
        assert result == "A,B,C"

    def test_parse_path_returns_uppercase(self):
        from geo_edit.evaluation.eval_shortest_path import parse_path

        text = "a, b, c"
        result = parse_path(text)
        assert result == "A,B,C"

    def test_parse_path_returns_none_for_no_comma(self):
        from geo_edit.evaluation.eval_shortest_path import parse_path

        text = "ABCD"
        result = parse_path(text)
        assert result is None

    def test_parse_path_returns_none_for_empty_string(self):
        from geo_edit.evaluation.eval_shortest_path import parse_path

        result = parse_path("")
        assert result is None

    def test_parse_path_returns_none_for_single_item(self):
        from geo_edit.evaluation.eval_shortest_path import parse_path

        text = "A,"
        result = parse_path(text)
        assert result is None

    def test_parse_path_strips_whitespace(self):
        from geo_edit.evaluation.eval_shortest_path import parse_path

        text = "  A  ,  B  ,  C  "
        result = parse_path(text)
        assert result == "A,B,C"

    def test_parse_path_with_multiline_answer(self):
        from geo_edit.evaluation.eval_shortest_path import parse_path

        text = "<answer>\nA, B, C\n</answer>"
        result = parse_path(text)
        assert result == "A,B,C"


class TestGtToStringFunction:
    """Test gt_to_string function from eval_shortest_path.py."""

    def test_gt_to_string_with_list(self):
        from geo_edit.evaluation.eval_shortest_path import gt_to_string

        result = gt_to_string(["a", "b", "c"])
        assert result == "A,B,C"

    def test_gt_to_string_with_string(self):
        from geo_edit.evaluation.eval_shortest_path import gt_to_string

        result = gt_to_string("a, b, c")
        assert result == "A,B,C"

    def test_gt_to_string_removes_spaces(self):
        from geo_edit.evaluation.eval_shortest_path import gt_to_string

        result = gt_to_string("A B C")
        assert result == "ABC"


class TestParseCountingFromLinesFunction:
    """Test parse_counting_from_lines function from eval_stmf_counting.py."""

    def test_parse_counting_simple_number(self):
        from geo_edit.evaluation.eval_stmf_counting import parse_counting_from_lines

        result = parse_counting_from_lines("42")
        assert result == 42

    def test_parse_counting_with_answer_prefix(self):
        from geo_edit.evaluation.eval_stmf_counting import parse_counting_from_lines

        result = parse_counting_from_lines("answer: 5")
        assert result == 5

    def test_parse_counting_with_final_answer_prefix(self):
        from geo_edit.evaluation.eval_stmf_counting import parse_counting_from_lines

        result = parse_counting_from_lines("final answer - 10")
        assert result == 10

    def test_parse_counting_with_negative_number(self):
        from geo_edit.evaluation.eval_stmf_counting import parse_counting_from_lines

        result = parse_counting_from_lines("-5")
        assert result == -5

    def test_parse_counting_with_positive_sign(self):
        from geo_edit.evaluation.eval_stmf_counting import parse_counting_from_lines

        result = parse_counting_from_lines("+3")
        assert result == 3

    def test_parse_counting_multiline_returns_first_number(self):
        from geo_edit.evaluation.eval_stmf_counting import parse_counting_from_lines

        text = "Some text\nanswer: 7\nmore text"
        result = parse_counting_from_lines(text)
        assert result == 7

    def test_parse_counting_returns_none_for_no_number(self):
        from geo_edit.evaluation.eval_stmf_counting import parse_counting_from_lines

        result = parse_counting_from_lines("no numbers here")
        assert result is None

    def test_parse_counting_returns_none_for_empty_string(self):
        from geo_edit.evaluation.eval_stmf_counting import parse_counting_from_lines

        result = parse_counting_from_lines("")
        assert result is None

    def test_parse_counting_skips_empty_lines(self):
        from geo_edit.evaluation.eval_stmf_counting import parse_counting_from_lines

        text = "\n\n\n42\n\n"
        result = parse_counting_from_lines(text)
        assert result == 42

    def test_parse_counting_case_insensitive_prefix(self):
        from geo_edit.evaluation.eval_stmf_counting import parse_counting_from_lines

        result = parse_counting_from_lines("ANSWER: 100")
        assert result == 100

        result = parse_counting_from_lines("Final Answer: 200")
        assert result == 200
