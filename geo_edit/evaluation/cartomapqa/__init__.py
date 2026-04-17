"""CartoMapQA unified evaluation package.

Provides structured answer extraction and paper-exact metric computation
for all 8 CartoMapQA sub-tasks: MFS, STMF (presence/counting/name_listing),
MTMF, RLE, MML, SRN.

Usage:
    python -m geo_edit.evaluation.cartomapqa.evaluate \
        --task cartomapqa_stmf_counting \
        --result_path /path/to/inference_output \
        --output_path /path/to/eval_output
"""
