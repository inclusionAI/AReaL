# PIR-8B Qualitative Examples

Selected cases where PIR-8B uses **>=2 distinct tools (Perceive + Interact)**, **answer is correct**, and **all tool calls return non-empty substantive results** (no `proposals: []`, `detections: []`, `not_found`, or execution failures).

## Primary 3 (main)

| # | Case | Domain | Tools | Steps |
|---|------|--------|-------|-------|
| 01 | `vstar_126` | **OOD** vstar_bench | image_crop + auto_segment + grounding_dino + text_spotting | 5 |
| 02 | `visworld_ballgame / 926` | **OOD** visworld_ballgame | grounding_dino + image_crop + image_label + text_spotting | 5 |
| 03 | `reason_map_plus / 1390` | **ID** reason_map_plus | image_crop + map_text_ocr + text_spotting | 4 |

## Backup 3 (alternative)

| # | Case | Domain | Tools | Steps |
|---|------|--------|-------|-------|
| 04 | `mapeval_visual / 292` | **OOD** mapeval_visual | image_crop + auto_segment + presence_check | 4 |
| 05 | `visual_probe_easy / 141` | **ID** visual_probe_easy | image_crop ×2 + text_spotting (2 intermediate images) | 4 |
| 06 | `visworld_cube / 114` | **OOD** visworld_cube | auto_segment + draw_path | 3 |

## Extra 9 (clean tool-output, added later)

All 9 below are verified: every tool call returns non-empty content; final answer scored 1.0.

### ID (5)

| # | Case | Dataset | Tools | Steps | Intermediate imgs |
|---|------|---------|-------|-------|-------------------|
| 07 | `reason_map_plus / 924` | reason_map_plus | image_crop + map_text_ocr + text_spotting | 4 | 1 |
| 08 | `reason_map_plus / 912` | reason_map_plus | image_crop ×2 + map_text_ocr ×2 | 5 | 2 |
| 09 | `visual_probe_medium / 6` | visual_probe_medium | image_crop ×2 + text_spotting | 4 | 2 |
| 10 | `visual_probe_medium / 256` | visual_probe_medium | image_crop + map_text_ocr (phone-number OCR) | 3 | 1 |
| 11 | `visual_probe_medium / 173` | visual_probe_medium | image_crop + text_ocr (year OCR) | 3 | 1 |

### OOD (4)

| # | Case | Dataset | Tools | Steps | Intermediate imgs |
|---|------|---------|-------|-------|-------------------|
| 12 | `vstar_bench / 60` | vstar_bench | image_crop + text_spotting | 3 | 1 |
| 13 | `visworld_cube / 352` | visworld_cube | image_crop + text_spotting | 3 | 1 |
| 14 | `visworld_ballgame / 288` | visworld_ballgame | draw_path + text_spotting | 3 | 1 |
| 15 | `visworld_ballgame / 323` | visworld_ballgame | draw_path + text_segment | 3 | 1 |

## Notes on Tool Coverage

The filtered clean pool (answer correct + P+I multi-tool + every tool result non-empty) contained **32 ID + 7 OOD** candidates. PIR-8B overwhelmingly selects the same small set of perception tools — `image_crop` combined with `text_spotting`, `map_text_ocr`, or `text_ocr` — plus occasional `draw_path` on ballgame trajectories. Rare perception/interaction tools (`image_highlight`, `bbox_segment`, `concept_count`, `exemplar_segment`, `bounding_box`, `draw_line`) were **not present** in the clean-output set for this checkpoint. Cases 01, 02, 04, 06 are the only examples in the pool exercising `auto_segment` / `grounding_dino` / `image_label` / `presence_check`.

## File Layout (per case folder)

- `input_image.png` — original image input
- `images/` — intermediate images produced by tools (crop / highlight / etc.)
- `trajectory.json` — full multi-turn conversation (user / assistant / tool messages)
- `meta_info.jsonl` — single-line json with all aggregated stats
- `output.jsonl` — per-step model outputs
- `extra_info.jsonl` — auxiliary info
- `CASE_INFO.json` — summarized info (question / gt / pred / tools / steps / tokens / intermediate_images list)
