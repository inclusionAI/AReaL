# PIR-8B Qualitative Examples

Selected 6 cases where PIR-8B uses **>=2 distinct tools (Perceive + Interact)** and **answer is correct**.

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

## File Layout (per case folder)

- `input_image.png` — original image input
- `images/` — intermediate images produced by tools (crop / highlight / etc.)
- `trajectory.json` — full multi-turn conversation (user / assistant / tool messages)
- `meta_info.jsonl` — single-line json with all aggregated stats
- `output.jsonl` — per-step model outputs
- `extra_info.jsonl` — auxiliary info
- `CASE_INFO.json` — summarized info (question / gt / pred / tools / steps / tokens / intermediate_images list)
