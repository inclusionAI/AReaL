from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional

from geo_edit.datasets.input_template import (
    CARTOMAPQA_INPUT_TEMPLATE,
    CARTOMAPQA_SRN_INPUT_TEMPLATE,
    CARTOMAPQA_STMF_COUNTING_TEMPLATE,
    CARTOMAPQA_STMF_NAME_LISTING_TEMPLATE,
    CARTOMAPQA_STMF_PRESENCE_TEMPLATE,
    MATHVISION_INPUT_TEMPLATE,
    MATHVISION_NOTOOL_INPUT_TEMPLATE,
    SUDOKU_TEXT_INPUT_TEMPLATE,
    SUDOKU_TOOL_CALL_INPUT_TEMPLATE,
)

FieldSource = str | Callable[[Mapping[str, Any]], str | int | float]


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    name: str
    id_key: str
    answer_key: str
    prompt_template: str
    template_fields: Dict[str, FieldSource]
    notool_prompt_template: Optional[str] = None
    image_key: Optional[str] = None

    def build_prompt(self, item: Mapping[str, Any], use_tools: bool) -> str:
        values: Dict[str, str | int | float] = {}
        for template_key, source in self.template_fields.items():
            if callable(source):
                values[template_key] = source(item)
            else:
                values[template_key] = item[source] if source in item else ""
        template = self.prompt_template
        if not use_tools and self.notool_prompt_template:
            template = self.notool_prompt_template
        return template.format(**values)

def _sudoku_total_cells(item: Mapping[str, Any]) -> int:
    return int(item["rows"]) * int(item["cols"])

def _sudoku_initial_board(item: Mapping[str, Any]) -> str:
    board=item["initial_board"]
    rows=int(item["rows"])
    cols=int(item["cols"])
    lines=[]
    for r in range(rows):
        line=" ".join(str(board[r*cols + c]) for c in range(cols))
        lines.append(line)
    return "\n".join(lines)

DATASET_SPECS: Dict[str, DatasetSpec] = {
    "sudoku": DatasetSpec(
        name="sudoku",
        id_key="puzzle_id",
        answer_key="solution",
        image_key="board_image",
        prompt_template=SUDOKU_TOOL_CALL_INPUT_TEMPLATE,
        notool_prompt_template=SUDOKU_TEXT_INPUT_TEMPLATE,
        template_fields={
            "rules": "rules",
            "rows": "rows",
            "cols": "cols",
            "total_cells": _sudoku_total_cells,
            "initial_board": _sudoku_initial_board,
            "visual_elements": "visual_elements",
        },
    ),
    "cartomapqa_srn": DatasetSpec(
        name="cartomapqa_srn",
        id_key="id",
        answer_key="route_directions",
        image_key="image",
        prompt_template=CARTOMAPQA_INPUT_TEMPLATE+ CARTOMAPQA_SRN_INPUT_TEMPLATE,
        notool_prompt_template=CARTOMAPQA_INPUT_TEMPLATE+ CARTOMAPQA_SRN_INPUT_TEMPLATE,
        template_fields={},
    ),
    "mathvisionqa": DatasetSpec(
        name="mathvisionqa",
        id_key="id",
        answer_key="answer",
        image_key="decoded_image",
        prompt_template=MATHVISION_INPUT_TEMPLATE,
        notool_prompt_template=MATHVISION_NOTOOL_INPUT_TEMPLATE,
        template_fields={
            "question": "question",
            "options": "options",
        },
    ),
    "cartomapqa_stmf_presence": DatasetSpec(
        name="cartomapqa_stmf_presence",
        id_key="id",
        answer_key="correct_answer",
        image_key="image",
        prompt_template=CARTOMAPQA_INPUT_TEMPLATE+CARTOMAPQA_STMF_PRESENCE_TEMPLATE,
        notool_prompt_template=CARTOMAPQA_INPUT_TEMPLATE+CARTOMAPQA_STMF_PRESENCE_TEMPLATE,
        template_fields={
            "mf_type": "mf_type",
        },
    ),
    "cartomapqa_stmf_counting": DatasetSpec(
        name="cartomapqa_stmf_counting",
        id_key="id",
        answer_key="correct_answer",
        image_key="image",
        prompt_template=CARTOMAPQA_INPUT_TEMPLATE+CARTOMAPQA_STMF_COUNTING_TEMPLATE,
        notool_prompt_template=CARTOMAPQA_INPUT_TEMPLATE+CARTOMAPQA_STMF_COUNTING_TEMPLATE,
        template_fields={
            "mf_type": "mf_type",
        },
    ),
    "cartomapqa_stmf_name_listing": DatasetSpec(
        name="cartomapqa_stmf_name_listing",
        id_key="id",
        answer_key="correct_answer",
        image_key="image",
        prompt_template=CARTOMAPQA_INPUT_TEMPLATE+CARTOMAPQA_STMF_NAME_LISTING_TEMPLATE,
        notool_prompt_template=CARTOMAPQA_INPUT_TEMPLATE+CARTOMAPQA_STMF_NAME_LISTING_TEMPLATE,
        template_fields={
            "mf_type": "mf_type",
        },
    ),
    "shortest_path_text": DatasetSpec(
        name="shortest_path_text",
        id_key="case_id",
        answer_key="answer",
        image_key="image",
        prompt_template="{prompt}",
        notool_prompt_template="{prompt}",
        template_fields={
            "prompt": "prompt",
        },
    ),
    "shortest_path_image": DatasetSpec(
        name="shortest_path_image",
        id_key="case_id",
        answer_key="answer",
        image_key="image",
        prompt_template="{prompt}",
        notool_prompt_template="{prompt}",
        template_fields={
            "prompt": "prompt",
        },
    ),
    "shortest_path_image_text": DatasetSpec(
        name="shortest_path_image_text",
        id_key="case_id",
        answer_key="answer",
        image_key="image",
        prompt_template="{prompt}",
        notool_prompt_template="{prompt}",
        template_fields={
            "prompt": "prompt",
        },
    ),
}

def get_dataset_spec(name: str) -> DatasetSpec:
    try:
        return DATASET_SPECS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset name: {name}") from exc
