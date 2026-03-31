from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional

from geo_edit.datasets.input_template import (
    BABYVISION_INPUT_TEMPLATE,
    BABYVISION_NOTOOL_INPUT_TEMPLATE,
    CARTOMAPQA_INPUT_TEMPLATE,
    CARTOMAPQA_SRN_INPUT_TEMPLATE,
    CARTOMAPQA_STMF_COUNTING_TEMPLATE,
    CARTOMAPQA_STMF_NAME_LISTING_TEMPLATE,
    CARTOMAPQA_STMF_PRESENCE_TEMPLATE,
    CHARTQA_ANSWER_FORMAT,
    CHARTQA_INPUT_TEMPLATE,
    CHARTQA_NOTOOL_INPUT_TEMPLATE,
    CHARTQA_SEPARATED_TEMPLATE,
    CHARTQAPRO_ANSWER_FORMAT,
    CHARTQAPRO_INPUT_TEMPLATE,
    CHARTQAPRO_NOTOOL_INPUT_TEMPLATE,
    CHARTQAPRO_SEPARATED_TEMPLATE,
    MAPEVAL_VISUAL_ANSWER_FORMAT,
    MAPEVAL_VISUAL_INPUT_TEMPLATE,
    MAPEVAL_VISUAL_NOTOOL_INPUT_TEMPLATE,
    MAPEVAL_VISUAL_SEPARATED_TEMPLATE,
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
    answer_key: str | Callable[[Mapping[str, Any]], Any]
    prompt_template: str
    template_fields: Dict[str, FieldSource]
    task_kwargs_fields: Dict[str, FieldSource] = field(default_factory=dict)
    notool_prompt_template: Optional[str] = None
    image_key: Optional[str] = None
    # Separated reasoning mode: question only (no role/answer format)
    separated_prompt_template: Optional[str] = None
    # Answer format instruction (added only in final answer phase)
    answer_format: Optional[str] = None
    # Per-dataset tool usage guidance (injected in Phase 1 reasoning)
    # Can be a static string or a callable(item) -> Optional[str] for per-item guidance
    tool_guidance: "Optional[str | Callable[[Mapping[str, Any]], Optional[str]]]" = None

    def get_tool_guidance(self, item: Optional[Mapping[str, Any]] = None) -> Optional[str]:
        """Get tool guidance string, resolving callable if needed."""
        if self.tool_guidance is None:
            return None
        if callable(self.tool_guidance):
            return self.tool_guidance(item) if item is not None else None
        return self.tool_guidance

    def build_prompt(self, item: Mapping[str, Any], use_tools: bool, separated: bool = False) -> str:
        values: Dict[str, Any] = {}
        for template_key, source in self.template_fields.items():
            if callable(source):
                values[template_key] = source(item)
            else:
                values[template_key] = item[source] if source in item else ""
        # Choose template based on mode
        if separated and self.separated_prompt_template:
            template = self.separated_prompt_template
        elif not use_tools and self.notool_prompt_template:
            template = self.notool_prompt_template
        else:
            template = self.prompt_template
        return template.format(**values)

    def build_task_kwargs(self, item: Mapping[str, Any]) -> Dict[str, Any]:
        values: Dict[str, Any] = {}
        for key, source in self.task_kwargs_fields.items():
            if callable(source):
                values[key] = source(item)
            else:
                values[key] = item[source] if source in item else None
        return values

    def get_answer(self, item: Mapping[str, Any]) -> Any:
        """Extract answer from item, supporting both string key and callable."""
        if callable(self.answer_key):
            return self.answer_key(item)
        return item[self.answer_key]


# =============================================================================
# Per-dataset tool guidance
# =============================================================================
VISWORLD_TOOL_GUIDANCE = {
    "ballgame": (
        "Strategy: First use a tool to understand the image layout and identify key elements "
        "(balls, holes, obstacles). Then use draw_line repeatedly to trace the ball's trajectory "
        "step by step through the game board. Verify your traced path before answering."
    ),
}


def _get_visworld_tool_guidance(item: Mapping[str, Any]) -> Optional[str]:
    """Return tool guidance based on VisWorld-Eval category."""
    category = item.get("category", "")
    return VISWORLD_TOOL_GUIDANCE.get(category)


# =============================================================================
# Helper functions for dataset specs
# =============================================================================


def _format_babyvision_options(item: Mapping[str, Any]) -> str:
    """Format options for BabyVision dataset."""
    options = item.get("options", [])
    if not options:
        return ""
    option_lines = [f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)]
    return "Options:\n" + "\n".join(option_lines)


def _get_babyvision_answer(item: Mapping[str, Any]) -> str:
    """Get answer based on answer type (choice or blank)."""
    if item.get("ansType") == "choice":
        choice_idx = item.get("choiceAns")
        if choice_idx is not None:
            return chr(65 + int(choice_idx))  # Convert to A, B, C, D...
    return str(item.get("blankAns", ""))


def _format_mapeval_visual_options(item: Mapping[str, Any]) -> str:
    """Format options for MapEval-Visual dataset.

    Options are 1-indexed. Answer=0 means unanswerable/none of the above.
    """
    options = item.get("options", [])
    if not options:
        return ""
    # Options start from 1, answer=0 means unanswerable
    option_lines = [f"{i}. {opt}" for i, opt in enumerate(options, start=1)]
    return "\n".join(option_lines)


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
        tool_guidance=_get_visworld_tool_guidance,
    ),
    "babyvision": DatasetSpec(
        name="babyvision",
        id_key="taskId",
        answer_key=_get_babyvision_answer,
        image_key="image",
        prompt_template=BABYVISION_INPUT_TEMPLATE,
        notool_prompt_template=BABYVISION_NOTOOL_INPUT_TEMPLATE,
        template_fields={
            "question": "question",
            "options_text": _format_babyvision_options,
        },
        task_kwargs_fields={
            "meta_info_extra": lambda item: {
                "type": item.get("type", ""),
                "subtype": item.get("subtype", ""),
                "ansType": item.get("ansType", ""),
            },
        },
    ),
    "mapeval_visual": DatasetSpec(
        name="mapeval_visual",
        id_key="id",
        answer_key="answer",
        image_key="image",
        prompt_template=MAPEVAL_VISUAL_INPUT_TEMPLATE,
        notool_prompt_template=MAPEVAL_VISUAL_NOTOOL_INPUT_TEMPLATE,
        template_fields={
            "question": "question",
            "options_text": _format_mapeval_visual_options,
        },
        task_kwargs_fields={
            "meta_info_extra": lambda item: {
                "classification": item.get("classification", ""),
            },
        },
        separated_prompt_template=MAPEVAL_VISUAL_SEPARATED_TEMPLATE,
        answer_format=MAPEVAL_VISUAL_ANSWER_FORMAT,
    ),
    "chartqa": DatasetSpec(
        name="chartqa",
        id_key="id",
        answer_key="answer",
        image_key="image",
        prompt_template=CHARTQA_INPUT_TEMPLATE,
        notool_prompt_template=CHARTQA_NOTOOL_INPUT_TEMPLATE,
        template_fields={
            "question": "question",
        },
        task_kwargs_fields={
            "meta_info_extra": lambda item: {
                "type": item.get("type", ""),
            },
        },
        separated_prompt_template=CHARTQA_SEPARATED_TEMPLATE,
        answer_format=CHARTQA_ANSWER_FORMAT,
    ),
    "chartqapro": DatasetSpec(
        name="chartqapro",
        id_key="id",
        answer_key="answer",
        image_key="image",
        prompt_template=CHARTQAPRO_INPUT_TEMPLATE,
        notool_prompt_template=CHARTQAPRO_NOTOOL_INPUT_TEMPLATE,
        template_fields={
            "question": "question",
        },
        task_kwargs_fields={
            "meta_info_extra": lambda item: {
                "question_type": item.get("question_type", ""),
                "chart_id": item.get("chart_id", -1),
                "question_idx": item.get("question_idx", 0),
                "year": item.get("year", ""),
            },
        },
        separated_prompt_template=CHARTQAPRO_SEPARATED_TEMPLATE,
        answer_format=CHARTQAPRO_ANSWER_FORMAT,
    ),
}


def get_dataset_spec(name: str) -> DatasetSpec:
    try:
        return DATASET_SPECS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset name: {name}") from exc
