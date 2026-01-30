from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional

from geo_edit.datasets.input_template import SUDOKU_TEXT_INPUT_TEMPLATE, SUDOKU_TOOL_CALL_INPUT_TEMPLATE, CARTOMAPQA_INPUT_TEMPLATE, CARTOMAPQA_SRN_INPUT_TEMPLATE, MATHVISION_INPUT_TEMPLATE, MATHVISION_NOTOOL_INPUT_TEMPLATE

FieldSource = str | Callable[[Mapping[str, Any]], Any]


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
        values: Dict[str, Any] = {}
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
}

def get_dataset_spec(name: str) -> DatasetSpec:
    try:
        return DATASET_SPECS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset name: {name}") from exc
