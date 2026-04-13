from __future__ import annotations

import json as _json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from geo_edit.prompts.system_prompts import (
    DATASET_TASK_TYPES,
    DEFAULT_OUTPUT_FORMAT,
    build_user_message,
)
from geo_edit.datasets.input_template import (
    BABYVISION_INPUT_TEMPLATE,
    BABYVISION_NOTOOL_INPUT_TEMPLATE,
    CARTOMAPQA_MFS_ANSWER_FORMAT,
    CARTOMAPQA_MML_ANSWER_FORMAT,
    CARTOMAPQA_MTMF_ANSWER_FORMAT,
    CARTOMAPQA_RLE_ANSWER_FORMAT,
    CARTOMAPQA_SRN_ANSWER_FORMAT,
    CARTOMAPQA_STMF_COUNTING_ANSWER_FORMAT,
    CARTOMAPQA_STMF_NAME_LISTING_ANSWER_FORMAT,
    CARTOMAPQA_STMF_PRESENCE_ANSWER_FORMAT,
    MAPTRACE_INPUT_TEMPLATE,
    MAPTRACE_NOTOOL_INPUT_TEMPLATE,
    MAPTRACE_SEPARATED_TEMPLATE,
    CARTOMAPQA_UNIFIED_TEMPLATE,
    CHARTQA_ANSWER_FORMAT,
    CHARTQA_INPUT_TEMPLATE,
    CHARTQA_NOTOOL_INPUT_TEMPLATE,
    CHARTQA_SEPARATED_TEMPLATE,
    CHARTQAPRO_ANSWER_FORMAT,
    CHARTQAPRO_INPUT_TEMPLATE,
    CHARTQAPRO_NOTOOL_INPUT_TEMPLATE,
    CHARTQAPRO_SEPARATED_TEMPLATE,
    REASONMAP_BASE_INPUT_TEMPLATE,
    REASONMAP_BASE_NOTOOL_INPUT_TEMPLATE,
    REASONMAP_BASE_SEPARATED_TEMPLATE,
    MM_MAPQA_ANSWER_FORMAT,
    MM_MAPQA_INPUT_TEMPLATE,
    MM_MAPQA_NOTOOL_INPUT_TEMPLATE,
    REASONMAP_INPUT_TEMPLATE,
    REASONMAP_NOTOOL_INPUT_TEMPLATE,
    REASONMAP_SEPARATED_TEMPLATE,
    MAPEVAL_VISUAL_ANSWER_FORMAT,
    MAPEVAL_VISUAL_INPUT_TEMPLATE,
    MAPEVAL_VISUAL_NOTOOL_INPUT_TEMPLATE,
    MAPEVAL_VISUAL_SEPARATED_TEMPLATE,
    VISWORLD_EVAL_INPUT_TEMPLATE,
    VISWORLD_EVAL_NOTOOL_INPUT_TEMPLATE,
    DEEPEYES_CHART_ANSWER_FORMAT,
    DEEPEYES_CHART_INPUT_TEMPLATE,
    DEEPEYES_CHART_NOTOOL_INPUT_TEMPLATE,
    DEEPEYES_CHART_SEPARATED_TEMPLATE,
    DEEPEYES_VSTAR_ANSWER_FORMAT,
    DEEPEYES_VSTAR_INPUT_TEMPLATE,
    DEEPEYES_VSTAR_NOTOOL_INPUT_TEMPLATE,
    DEEPEYES_VSTAR_SEPARATED_TEMPLATE,
    AIIC_ANSWER_FORMAT,
    AIIC_INPUT_TEMPLATE,
    AIIC_NOTOOL_INPUT_TEMPLATE,
    AIIC_SEPARATED_TEMPLATE,
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
    # Column to deduplicate images by (e.g. "city" for ReasonMap-Plus where
    # 30 unique maps are shared across 2570 items). When set, prepare_images()
    # saves each unique image once and drops the image column from the dataset.
    image_dedup_key: Optional[str] = None
    # Additional prompt appended to LLM judge query for task-specific evaluation hints.
    # Can be a static string or a callable(item) -> Optional[str] for per-item hints.
    judge_prompt: "Optional[str | Callable[[Mapping[str, Any]], Optional[str]]]" = None

    def prepare_images(self, dataset, output_dir: str) -> Tuple:
        """Pre-save deduplicated images to disk, return (dataset, image_map).

        When image_dedup_key is set, saves one copy per unique key value and
        returns a mapping {item_id: image_path}. The image column is removed
        so items in pending lists stay lightweight.

        Returns (dataset, {}) unchanged when dedup is not applicable.
        """
        if (
            not self.image_dedup_key
            or not self.image_key
            or self.image_key not in dataset.column_names
            or self.image_dedup_key not in dataset.column_names
        ):
            return dataset, {}

        from datasets import Image as HFImage

        dataset = dataset.cast_column(self.image_key, HFImage(decode=False))
        shared_dir = os.path.join(output_dir, "_shared_images")
        os.makedirs(shared_dir, exist_ok=True)

        image_map: Dict[str, str] = {}
        saved_keys: set = set()
        for item in dataset:
            dedup_val = item[self.image_dedup_key]
            item_id = str(item[self.id_key])
            if dedup_val not in saved_keys:
                raw = item[self.image_key]
                if raw and isinstance(raw, dict) and raw.get("bytes"):
                    path = os.path.join(shared_dir, f"{dedup_val}.png")
                    if not os.path.exists(path):
                        with open(path, "wb") as f:
                            f.write(raw["bytes"])
                    saved_keys.add(dedup_val)
            image_map[item_id] = os.path.join(shared_dir, f"{dedup_val}.png")

        dataset = dataset.remove_columns([self.image_key])
        return dataset, image_map

    def get_tool_guidance(
        self, item: Optional[Mapping[str, Any]] = None
    ) -> Optional[str]:
        """Get tool guidance string, resolving callable if needed."""
        if self.tool_guidance is None:
            return None
        if callable(self.tool_guidance):
            return self.tool_guidance(item) if item is not None else None
        return self.tool_guidance

    def get_judge_prompt(
        self, item: Optional[Mapping[str, Any]] = None
    ) -> Optional[str]:
        """Get judge additional prompt string, resolving callable if needed."""
        if self.judge_prompt is None:
            return None
        if callable(self.judge_prompt):
            return self.judge_prompt(item) if item is not None else None
        return self.judge_prompt

    def build_prompt(
        self,
        item: Mapping[str, Any],
        use_tools: bool,
        separated: bool = False,
        unified: bool = False,
    ) -> str:
        values: Dict[str, Any] = {}
        for template_key, source in self.template_fields.items():
            if callable(source):
                values[template_key] = source(item)
            else:
                values[template_key] = item[source] if source in item else ""

        if unified:
            question = values.get("question", values.get("prompt", ""))
            task_type = DATASET_TASK_TYPES.get(self.name, "visual question answering")
            output_format = self.answer_format or DEFAULT_OUTPUT_FORMAT
            return build_user_message(
                question=question,
                num_images=0,  # image prefix handled separately by caller
                task_type=task_type,
                output_format=output_format,
            )

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
    option_lines = [f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)]
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


def _get_reasonmap_answer(item: Mapping[str, Any]) -> str:
    """Get answer for ReasonMap dataset (merged from ReasonMap-Plus and ReasonMap-Train).

    - Counting1 (ABCD): answer is 0-indexed option index -> convert to A/B/C/D
    - Counting2/3: answer is the actual count
    - TorF1/TorF2: answer 0=No, 1=Yes
    """
    answer = item.get("answer", 0)
    qtype = item.get("type", "")
    if qtype == "Counting1":
        return chr(65 + int(answer))  # 0->A, 1->B, 2->C, 3->D
    if qtype in ("TorF1", "TorF2"):
        return "Yes" if int(answer) == 1 else "No"
    return str(answer)


def _get_reasonmap_base_answer(item: Mapping[str, Any]) -> str:
    """Serialise the ``routes`` ground truth for ReasonMap base (route planning)."""
    routes = item.get("routes", {})
    if isinstance(routes, str):
        return routes
    return _json.dumps(routes, ensure_ascii=False)


# =============================================================================
# Per-dataset judge prompts (additional hints for LLM-as-judge evaluation)
# =============================================================================
CARTOMAPQA_STMF_COUNTING_JUDGE_PROMPT = (
    "Additional evaluation rules for counting answers:\n"
    "- Allow an absolute tolerance of 1: if the predicted count differs from the "
    "ground truth by at most 1, consider it correct.\n"
    "- For example, if the ground truth is 2, then 1, 2, or 3 are all correct."
)

CARTOMAPQA_STMF_COUNTING_TOOL_GUIDANCE = (
    "Strategy: The map may contain many small POI icons spread across a large area. "
    "To count accurately, first use image_crop to divide the map into smaller regions "
    "(e.g. quadrants), then analyze each cropped region individually using tools like "
    "presence_check or map_text_ocr. Finally, sum the counts from all regions. "
    "This divide-and-conquer approach reduces missed detections."
)

CARTOMAPQA_SRN_JUDGE_PROMPT = (
    "Additional evaluation rules for route navigation answers:\n"
    "- 'road_1, continue straight, road_1' is equivalent to 'road_1' only. "
    "Redundant straight continuations on the same road should be ignored.\n"
    "- The direction sequence matters: compare step by step.\n"
    "- Road name synonyms or abbreviations (e.g. 'St' vs 'Street') are acceptable."
)

CARTOMAPQA_MML_JUDGE_PROMPT = (
    "Additional evaluation rules for map marker localization answers:\n"
    "- The answer is a JSON with 'road_1' and 'road_2' fields.\n"
    "- The order of road_1 and road_2 does NOT matter: "
    "{road_1: A, road_2: B} is equivalent to {road_1: B, road_2: A}.\n"
    "- Road name abbreviations (e.g. 'Rd' vs 'Road', 'St' vs 'Street') are acceptable."
)

CARTOMAPQA_RLE_JUDGE_PROMPT = (
    "Additional evaluation rules for route length estimation answers:\n"
    "- The answer is a numeric value with unit (e.g. '1194.497 m' or '3918.95 ft').\n"
    "- Allow a relative tolerance of 15%: if the predicted value is within 15% "
    "of the ground truth, consider it correct.\n"
    "- Unit must match (meters vs feet)."
)

CARTOMAPQA_STMF_NAME_LISTING_JUDGE_PROMPT = (
    "Additional evaluation rules for name listing answers:\n"
    "- The answer is a list of POI names separated by newlines.\n"
    "- The order of names does NOT matter.\n"
    "- Minor spelling variations or abbreviations are acceptable."
)


DATASET_SPECS: Dict[str, DatasetSpec] = {
    # -- CartoMapQA tasks (unified: HF dataset `question` contains full prompt) --
    "cartomapqa_mfs": DatasetSpec(
        name="cartomapqa_mfs",
        id_key="id",
        answer_key="answer",
        image_key="image",
        prompt_template=CARTOMAPQA_UNIFIED_TEMPLATE,
        notool_prompt_template=CARTOMAPQA_UNIFIED_TEMPLATE,
        template_fields={"question": "question"},
        answer_format=CARTOMAPQA_MFS_ANSWER_FORMAT,
    ),
    "cartomapqa_stmf_presence": DatasetSpec(
        name="cartomapqa_stmf_presence",
        id_key="id",
        answer_key="answer",
        image_key="image",
        prompt_template=CARTOMAPQA_UNIFIED_TEMPLATE,
        notool_prompt_template=CARTOMAPQA_UNIFIED_TEMPLATE,
        template_fields={"question": "question"},
        answer_format=CARTOMAPQA_STMF_PRESENCE_ANSWER_FORMAT,
    ),
    "cartomapqa_stmf_counting": DatasetSpec(
        name="cartomapqa_stmf_counting",
        id_key="id",
        answer_key="answer",
        image_key="image",
        prompt_template=CARTOMAPQA_UNIFIED_TEMPLATE,
        notool_prompt_template=CARTOMAPQA_UNIFIED_TEMPLATE,
        template_fields={"question": "question"},
        answer_format=CARTOMAPQA_STMF_COUNTING_ANSWER_FORMAT,
        judge_prompt=CARTOMAPQA_STMF_COUNTING_JUDGE_PROMPT,
        tool_guidance=CARTOMAPQA_STMF_COUNTING_TOOL_GUIDANCE,
    ),
    "cartomapqa_stmf_name_listing": DatasetSpec(
        name="cartomapqa_stmf_name_listing",
        id_key="id",
        answer_key="answer",
        image_key="image",
        prompt_template=CARTOMAPQA_UNIFIED_TEMPLATE,
        notool_prompt_template=CARTOMAPQA_UNIFIED_TEMPLATE,
        template_fields={"question": "question"},
        judge_prompt=CARTOMAPQA_STMF_NAME_LISTING_JUDGE_PROMPT,
        answer_format=CARTOMAPQA_STMF_NAME_LISTING_ANSWER_FORMAT,
    ),
    "cartomapqa_mtmf": DatasetSpec(
        name="cartomapqa_mtmf",
        id_key="id",
        answer_key="answer",
        image_key="image",
        prompt_template=CARTOMAPQA_UNIFIED_TEMPLATE,
        notool_prompt_template=CARTOMAPQA_UNIFIED_TEMPLATE,
        template_fields={"question": "question"},
        answer_format=CARTOMAPQA_MTMF_ANSWER_FORMAT,
    ),
    "cartomapqa_rle": DatasetSpec(
        name="cartomapqa_rle",
        id_key="id",
        answer_key="answer",
        image_key="image",
        prompt_template=CARTOMAPQA_UNIFIED_TEMPLATE,
        notool_prompt_template=CARTOMAPQA_UNIFIED_TEMPLATE,
        template_fields={"question": "question"},
        judge_prompt=CARTOMAPQA_RLE_JUDGE_PROMPT,
        answer_format=CARTOMAPQA_RLE_ANSWER_FORMAT,
    ),
    "cartomapqa_mml": DatasetSpec(
        name="cartomapqa_mml",
        id_key="id",
        answer_key="answer",
        image_key="image",
        prompt_template=CARTOMAPQA_UNIFIED_TEMPLATE,
        notool_prompt_template=CARTOMAPQA_UNIFIED_TEMPLATE,
        template_fields={"question": "question"},
        judge_prompt=CARTOMAPQA_MML_JUDGE_PROMPT,
        answer_format=CARTOMAPQA_MML_ANSWER_FORMAT,
    ),
    "cartomapqa_srn": DatasetSpec(
        name="cartomapqa_srn",
        id_key="id",
        answer_key="answer",
        image_key="image",
        prompt_template=CARTOMAPQA_UNIFIED_TEMPLATE,
        notool_prompt_template=CARTOMAPQA_UNIFIED_TEMPLATE,
        template_fields={"question": "question"},
        judge_prompt=CARTOMAPQA_SRN_JUDGE_PROMPT,
        answer_format=CARTOMAPQA_SRN_ANSWER_FORMAT,
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
    # Merged from FSCCS/ReasonMap-Plus (test) and FSCCS/ReasonMap-Train (train).
    # Planning tasks are excluded during preprocessing; only Counting1/2/3 and
    # TorF1/2 question types are kept.
    "reason_map_plus": DatasetSpec(
        name="reason_map_plus",
        id_key="id",
        answer_key=_get_reasonmap_answer,
        image_key="image",
        prompt_template=REASONMAP_INPUT_TEMPLATE,
        notool_prompt_template=REASONMAP_NOTOOL_INPUT_TEMPLATE,
        template_fields={
            "question": "question",
        },
        task_kwargs_fields={
            "meta_info_extra": lambda item: {
                "type": item.get("type", ""),
                "difficulty_city": item.get("difficulty_city", ""),
                "difficulty_question": item.get("difficulty_question", ""),
                "city": item.get("city", ""),
                "country": item.get("country", ""),
            },
        },
        separated_prompt_template=REASONMAP_SEPARATED_TEMPLATE,
        image_dedup_key="city",
    ),
    "reason_map": DatasetSpec(
        name="reason_map",
        id_key="id",
        answer_key=_get_reasonmap_base_answer,
        image_key="image",
        prompt_template=REASONMAP_BASE_INPUT_TEMPLATE,
        notool_prompt_template=REASONMAP_BASE_NOTOOL_INPUT_TEMPLATE,
        template_fields={
            "question": "question_long",
        },
        task_kwargs_fields={
            "meta_info_extra": lambda item: {
                "category": "reason_map",
                "country": item.get("country", ""),
                "city": item.get("city", ""),
                "station_1": item.get("station_1", ""),
                "station_2": item.get("station_2", ""),
                "difficulty_question": item.get("difficulty_question", ""),
                "difficulty_city": item.get("difficulty_city", ""),
                "city_line_count": item.get("city_line_count", 0),
                "city_transfer_count": item.get("city_transfer_count", 0),
                "question_transfer_count": item.get("question_transfer_count", 0),
                "metro_data": item.get("json", ""),
            },
        },
        separated_prompt_template=REASONMAP_BASE_SEPARATED_TEMPLATE,
        image_dedup_key="city",
        answer_format=(
            "Provide your final answer in <answer></answer> tags using this EXACT format "
            "(multiple route sections separated by a line containing only --):\n\n"
            "Route Name: [line name exactly as shown on the map]\n"
            "Departure Stop: [station name]\n"
            "Arrival Stop: [station name]\n"
            "Number of Via Stops: [number of intermediate stops]\n"
            "--\n"
            "Route Name: [next line name]\n"
            "Departure Stop: [transfer station name]\n"
            "Arrival Stop: [destination station name]\n"
            "Number of Via Stops: [number]\n\n"
            "Rules:\n"
            "- The first section's Departure Stop must be the origin station.\n"
            "- The last section's Arrival Stop must be the destination station.\n"
            "- Adjacent sections must share a transfer station "
            "(section N's Arrival Stop = section N+1's Departure Stop).\n"
            "- Every station must exist on the specified route."
        ),
    ),
    "mm_mapqa": DatasetSpec(
        name="mm_mapqa",
        id_key="id",
        answer_key="answer",
        image_key="image",
        prompt_template=MM_MAPQA_INPUT_TEMPLATE,
        notool_prompt_template=MM_MAPQA_NOTOOL_INPUT_TEMPLATE,
        template_fields={"question": "question"},
        answer_format=MM_MAPQA_ANSWER_FORMAT,
    ),
    "map_trace": DatasetSpec(
        name="map_trace",
        id_key="id",
        answer_key="label",
        image_key="image",
        prompt_template=MAPTRACE_INPUT_TEMPLATE,
        notool_prompt_template=MAPTRACE_NOTOOL_INPUT_TEMPLATE,
        template_fields={
            "question": "input",
        },
        task_kwargs_fields={
            "meta_info_extra": lambda item: {
                "category": "map_trace",
            },
        },
        separated_prompt_template=MAPTRACE_SEPARATED_TEMPLATE,
    ),
    "deepeyes_chart": DatasetSpec(
        name="deepeyes_chart",
        id_key="sample_index",
        answer_key="answer",
        image_key="image",
        prompt_template=DEEPEYES_CHART_INPUT_TEMPLATE,
        notool_prompt_template=DEEPEYES_CHART_NOTOOL_INPUT_TEMPLATE,
        template_fields={"question": "question"},
        separated_prompt_template=DEEPEYES_CHART_SEPARATED_TEMPLATE,
        answer_format=DEEPEYES_CHART_ANSWER_FORMAT,
    ),
    "deepeyes_vstar": DatasetSpec(
        name="deepeyes_vstar",
        id_key="sample_index",
        answer_key="answer",
        image_key="image",
        prompt_template=DEEPEYES_VSTAR_INPUT_TEMPLATE,
        notool_prompt_template=DEEPEYES_VSTAR_NOTOOL_INPUT_TEMPLATE,
        template_fields={"question": "question"},
        separated_prompt_template=DEEPEYES_VSTAR_SEPARATED_TEMPLATE,
        answer_format=DEEPEYES_VSTAR_ANSWER_FORMAT,
    ),
    "aiic_0509": DatasetSpec(
        name="aiic_0509",
        id_key="sample_index",
        answer_key="answer",
        image_key="image",
        prompt_template=AIIC_INPUT_TEMPLATE,
        notool_prompt_template=AIIC_NOTOOL_INPUT_TEMPLATE,
        template_fields={"question": "question"},
        separated_prompt_template=AIIC_SEPARATED_TEMPLATE,
        answer_format=AIIC_ANSWER_FORMAT,
    ),
    "aiic_0512": DatasetSpec(
        name="aiic_0512",
        id_key="sample_index",
        answer_key="answer",
        image_key="image",
        prompt_template=AIIC_INPUT_TEMPLATE,
        notool_prompt_template=AIIC_NOTOOL_INPUT_TEMPLATE,
        template_fields={"question": "question"},
        separated_prompt_template=AIIC_SEPARATED_TEMPLATE,
        answer_format=AIIC_ANSWER_FORMAT,
    ),
    "aiic_0704": DatasetSpec(
        name="aiic_0704",
        id_key="sample_index",
        answer_key="answer",
        image_key="image",
        prompt_template=AIIC_INPUT_TEMPLATE,
        notool_prompt_template=AIIC_NOTOOL_INPUT_TEMPLATE,
        template_fields={"question": "question"},
        separated_prompt_template=AIIC_SEPARATED_TEMPLATE,
        answer_format=AIIC_ANSWER_FORMAT,
    ),
    "aiic_0716": DatasetSpec(
        name="aiic_0716",
        id_key="sample_index",
        answer_key="answer",
        image_key="image",
        prompt_template=AIIC_INPUT_TEMPLATE,
        notool_prompt_template=AIIC_NOTOOL_INPUT_TEMPLATE,
        template_fields={"question": "question"},
        separated_prompt_template=AIIC_SEPARATED_TEMPLATE,
        answer_format=AIIC_ANSWER_FORMAT,
    ),
}


def get_dataset_spec(name: str) -> DatasetSpec:
    try:
        return DATASET_SPECS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown dataset name: {name}") from exc
