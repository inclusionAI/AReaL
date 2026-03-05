from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional

from geo_edit.datasets.input_template import (
    CARTOMAPQA_INPUT_TEMPLATE,
    CARTOMAPQA_SRN_INPUT_TEMPLATE,
    CARTOMAPQA_STMF_COUNTING_TEMPLATE,
    CARTOMAPQA_STMF_NAME_LISTING_TEMPLATE,
    CARTOMAPQA_STMF_PRESENCE_TEMPLATE,
    MATHVISION_INPUT_TEMPLATE,
    MATHVISION_NOTOOL_INPUT_TEMPLATE,
    VISWORLD_EVAL_INPUT_TEMPLATE,
    VISWORLD_EVAL_NOTOOL_INPUT_TEMPLATE,
)

FieldSource = str | Callable[[Mapping[str, Any]], Any]


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    name: str
    id_key: str
    answer_key: str
    prompt_template: str
    template_fields: Dict[str, FieldSource]
    task_kwargs_fields: Dict[str, FieldSource] = field(default_factory=dict)
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

    def build_task_kwargs(self, item: Mapping[str, Any]) -> Dict[str, Any]:
        values: Dict[str, Any] = {}
        for key, source in self.task_kwargs_fields.items():
            if callable(source):
                values[key] = source(item)
            else:
                values[key] = item[source] if source in item else None
        return values


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "cartomapqa_srn": DatasetSpec(
        name="cartomapqa_srn",
        id_key="id",
        answer_key="route_directions",
        image_key="image",
        prompt_template=CARTOMAPQA_INPUT_TEMPLATE + CARTOMAPQA_SRN_INPUT_TEMPLATE,
        notool_prompt_template=CARTOMAPQA_INPUT_TEMPLATE + CARTOMAPQA_SRN_INPUT_TEMPLATE,
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
        prompt_template=CARTOMAPQA_INPUT_TEMPLATE + CARTOMAPQA_STMF_PRESENCE_TEMPLATE,
        notool_prompt_template=CARTOMAPQA_INPUT_TEMPLATE + CARTOMAPQA_STMF_PRESENCE_TEMPLATE,
        template_fields={
            "mf_type": "mf_type",
        },
    ),
    "cartomapqa_stmf_counting": DatasetSpec(
        name="cartomapqa_stmf_counting",
        id_key="id",
        answer_key="correct_answer",
        image_key="image",
        prompt_template=CARTOMAPQA_INPUT_TEMPLATE + CARTOMAPQA_STMF_COUNTING_TEMPLATE,
        notool_prompt_template=CARTOMAPQA_INPUT_TEMPLATE + CARTOMAPQA_STMF_COUNTING_TEMPLATE,
        template_fields={
            "mf_type": "mf_type",
        },
    ),
    "cartomapqa_stmf_name_listing": DatasetSpec(
        name="cartomapqa_stmf_name_listing",
        id_key="id",
        answer_key="correct_answer",
        image_key="image",
        prompt_template=CARTOMAPQA_INPUT_TEMPLATE + CARTOMAPQA_STMF_NAME_LISTING_TEMPLATE,
        notool_prompt_template=CARTOMAPQA_INPUT_TEMPLATE + CARTOMAPQA_STMF_NAME_LISTING_TEMPLATE,
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
        task_kwargs_fields={
            "meta_info_extra": lambda item: {"level_nodes": int(item["level_nodes"])} if "level_nodes" in item else {},
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
        task_kwargs_fields={
            "meta_info_extra": lambda item: {"level_nodes": int(item["level_nodes"])} if "level_nodes" in item else {},
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
        task_kwargs_fields={
            "meta_info_extra": lambda item: {"level_nodes": int(item["level_nodes"])} if "level_nodes" in item else {},
        },
    ),
    "visworld_eval": DatasetSpec(
        name="visworld_eval",
        id_key="index",
        answer_key="answer",
        image_key="image",
        prompt_template=VISWORLD_EVAL_INPUT_TEMPLATE,
        notool_prompt_template=VISWORLD_EVAL_NOTOOL_INPUT_TEMPLATE,
        template_fields={
            "prompt": "prompt",  # Use the pre-formatted prompt from dataset
        },
        task_kwargs_fields={
            "meta_info_extra": lambda item: {
                "category": item.get("category", ""),
            },
        },
    ),
}


def get_dataset_spec(name: str) -> DatasetSpec:
    try:
        return DATASET_SPECS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset name: {name}") from exc
