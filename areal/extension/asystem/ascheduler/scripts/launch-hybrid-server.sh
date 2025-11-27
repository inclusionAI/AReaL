#!/usr/bin/env bash

set -e

GPU_COUNT=$(nvidia-smi -L 2>/dev/null | grep "UUID" | wc -l)

if [[ ${GPU_COUNT} -ne 0 ]]; then
    export CUDA_VISIBLE_DEVICES=$(seq -s ',' 0 $((GPU_COUNT - 1)))
fi

# ENV:
#   WORKER_INDEX: worker index of a worker group
#   WORKER_SHARE_DIR: share with paired ModelWorker
#   IMAGE: singularity image
#   WORKER_COMMAND: worker startup command
#   LOG_DIR: worker log dir
#   REAL_PACKAGE_PATH: package path in shared storage
WORKER_TYPE=${ROLE}-${TYPE}

if [[ "${WORKER_INDEX_OFFSET}" != "" ]]; then
    WORKER_INDEX=$((WORKER_INDEX + WORKER_INDEX_OFFSET))
fi

export NODE_ID=$((${WORKER_INDEX} / 8))

LOG_DIR=${LOG_DIR}/${UUID}
mkdir -p ${LOG_DIR}

WORKER_LOG_FILE=${LOG_DIR}/${WORKER_TYPE}-${WORKER_INDEX}.log

LOCAL_WORKER_DIR=/home/admin/logs/experiment/

mkdir -p ${LOCAL_WORKER_DIR}/${EXPR_NAME}/${TRIAL_NAME}
WORKER_ANTLOGS_LOG_FILE=${LOCAL_WORKER_DIR}/${EXPR_NAME}/${TRIAL_NAME}/${WORKER_TYPE}-${WORKER_INDEX}.log

log_to_both_files() {
    echo "$@" | tee -a ${WORKER_ANTLOGS_LOG_FILE} >> ${WORKER_LOG_FILE}
}

log_to_both_files "ASystem Entrypoint Env:"
env >> ${WORKER_LOG_FILE}
env >> ${WORKER_ANTLOGS_LOG_FILE}
log_to_both_files "ASystem Worker Command: ${WORKER_COMMAND}"

log_to_both_files `uname -a`

CMD="python -m asystem_runtime.engine_server --worker-type ${WORKER_TYPE} --worker-index ${WORKER_INDEX}"

echo "before start." >> ${WORKER_LOG_FILE}
singularity --debug exec --nv --no-home --writable-tmpfs --bind /storage:/storage --bind /home/admin/logs:/home/admin/logs "${IMAGE}" bash -c "export PATH=/opt/conda/bin:$PATH; export PYTHONPATH=${EXTRA_PYTHONPATH}:/home/admin/antllm:/home/admin/Megatron-LM:/home/admin/Megatron-LM/ant_utils/dcp_utils:/root/workdir/astra-build/Asystem-HybridEngine/astra_cache/astra-client/python:$PYTHONPATH;$CMD" 2>&1 | tee -a ${WORKER_ANTLOGS_LOG_FILE} >> ${WORKER_LOG_FILE}
