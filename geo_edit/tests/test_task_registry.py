"""Tests for datasets/task_registry.py - DatasetSpec and DATASET_SPECS."""

import pytest


class TestDatasetSpec:
    """Test DatasetSpec dataclass."""

    def test_build_prompt_with_string_field_source(self):
        from geo_edit.datasets.task_registry import DatasetSpec

        spec = DatasetSpec(
            name="test",
            id_key="id",
            answer_key="answer",
            prompt_template="Question: {question}",
            template_fields={"question": "question_text"},
        )

        item = {"question_text": "What is 2+2?"}
        result = spec.build_prompt(item, use_tools=True)
        assert result == "Question: What is 2+2?"

    def test_build_prompt_with_callable_field_source(self):
        from geo_edit.datasets.task_registry import DatasetSpec

        spec = DatasetSpec(
            name="test",
            id_key="id",
            answer_key="answer",
            prompt_template="Sum: {total}",
            template_fields={"total": lambda item: item["a"] + item["b"]},
        )

        item = {"a": 10, "b": 20}
        result = spec.build_prompt(item, use_tools=True)
        assert result == "Sum: 30"

    def test_build_prompt_uses_notool_template_when_use_tools_false(self):
        from geo_edit.datasets.task_registry import DatasetSpec

        spec = DatasetSpec(
            name="test",
            id_key="id",
            answer_key="answer",
            prompt_template="With tools: {q}",
            notool_prompt_template="No tools: {q}",
            template_fields={"q": "question"},
        )

        item = {"question": "test"}
        result = spec.build_prompt(item, use_tools=False)
        assert result == "No tools: test"

    def test_build_prompt_falls_back_to_tool_template_when_no_notool_template(self):
        from geo_edit.datasets.task_registry import DatasetSpec

        spec = DatasetSpec(
            name="test",
            id_key="id",
            answer_key="answer",
            prompt_template="With tools: {q}",
            notool_prompt_template=None,
            template_fields={"q": "question"},
        )

        item = {"question": "test"}
        result = spec.build_prompt(item, use_tools=False)
        assert result == "With tools: test"

    def test_build_prompt_handles_missing_field(self):
        from geo_edit.datasets.task_registry import DatasetSpec

        spec = DatasetSpec(
            name="test",
            id_key="id",
            answer_key="answer",
            prompt_template="Value: {value}",
            template_fields={"value": "missing_key"},
        )

        item = {}
        result = spec.build_prompt(item, use_tools=True)
        assert result == "Value: "

    def test_build_task_kwargs_with_string_source(self):
        from geo_edit.datasets.task_registry import DatasetSpec

        spec = DatasetSpec(
            name="test",
            id_key="id",
            answer_key="answer",
            prompt_template="",
            template_fields={},
            task_kwargs_fields={"extra_data": "data_field"},
        )

        item = {"data_field": "extra_value"}
        result = spec.build_task_kwargs(item)
        assert result == {"extra_data": "extra_value"}

    def test_build_task_kwargs_with_callable_source(self):
        from geo_edit.datasets.task_registry import DatasetSpec

        spec = DatasetSpec(
            name="test",
            id_key="id",
            answer_key="answer",
            prompt_template="",
            template_fields={},
            task_kwargs_fields={
                "meta_info_extra": lambda item: {"level": item.get("level", 0)}
            },
        )

        item = {"level": 5}
        result = spec.build_task_kwargs(item)
        assert result == {"meta_info_extra": {"level": 5}}

    def test_build_task_kwargs_handles_missing_field(self):
        from geo_edit.datasets.task_registry import DatasetSpec

        spec = DatasetSpec(
            name="test",
            id_key="id",
            answer_key="answer",
            prompt_template="",
            template_fields={},
            task_kwargs_fields={"extra": "missing_field"},
        )

        item = {}
        result = spec.build_task_kwargs(item)
        assert result == {"extra": None}


class TestDatasetSpecs:
    """Test DATASET_SPECS registry."""

    def test_all_expected_datasets_registered(self):
        from geo_edit.datasets.task_registry import DATASET_SPECS

        expected = [
            "sudoku",
            "cartomapqa_srn",
            "mathvisionqa",
            "cartomapqa_stmf_presence",
            "cartomapqa_stmf_counting",
            "cartomapqa_stmf_name_listing",
            "shortest_path_text",
            "shortest_path_image",
            "shortest_path_image_text",
        ]
        for name in expected:
            assert name in DATASET_SPECS, f"Missing dataset: {name}"

    def test_all_specs_have_required_fields(self):
        from geo_edit.datasets.task_registry import DATASET_SPECS

        for name, spec in DATASET_SPECS.items():
            assert spec.name == name, f"Dataset {name} has mismatched name"
            assert spec.id_key is not None, f"Dataset {name} missing id_key"
            assert spec.answer_key is not None, f"Dataset {name} missing answer_key"
            assert spec.prompt_template is not None, f"Dataset {name} missing prompt_template"

    def test_sudoku_spec(self):
        from geo_edit.datasets.task_registry import DATASET_SPECS

        spec = DATASET_SPECS["sudoku"]
        assert spec.id_key == "puzzle_id"
        assert spec.answer_key == "solution"
        assert spec.image_key == "board_image"
        assert spec.notool_prompt_template is not None

    def test_mathvisionqa_spec(self):
        from geo_edit.datasets.task_registry import DATASET_SPECS

        spec = DATASET_SPECS["mathvisionqa"]
        assert spec.id_key == "id"
        assert spec.answer_key == "answer"
        assert spec.image_key == "decoded_image"
        assert "question" in spec.template_fields

    def test_shortest_path_specs_have_task_kwargs(self):
        from geo_edit.datasets.task_registry import DATASET_SPECS

        for name in ["shortest_path_text", "shortest_path_image", "shortest_path_image_text"]:
            spec = DATASET_SPECS[name]
            assert "meta_info_extra" in spec.task_kwargs_fields


class TestGetDatasetSpec:
    """Test get_dataset_spec function."""

    def test_get_existing_dataset(self):
        from geo_edit.datasets.task_registry import get_dataset_spec

        spec = get_dataset_spec("sudoku")
        assert spec.name == "sudoku"

    def test_get_nonexistent_dataset_raises_error(self):
        from geo_edit.datasets.task_registry import get_dataset_spec

        with pytest.raises(KeyError, match="Unknown dataset name"):
            get_dataset_spec("nonexistent_dataset")


class TestSudokuHelperFunctions:
    """Test sudoku helper functions."""

    def test_sudoku_total_cells(self):
        from geo_edit.datasets.task_registry import _sudoku_total_cells

        item = {"rows": 9, "cols": 9}
        result = _sudoku_total_cells(item)
        assert result == 81

    def test_sudoku_initial_board(self):
        from geo_edit.datasets.task_registry import _sudoku_initial_board

        item = {
            "rows": 2,
            "cols": 2,
            "initial_board": [1, 2, 3, 4],
        }
        result = _sudoku_initial_board(item)
        assert result == "1 2\n3 4"
