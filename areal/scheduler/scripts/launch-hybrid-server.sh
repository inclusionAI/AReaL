#!/usr/bin/env bash

set -e

GPU_COUNT=$(nvidia-smi -L 2>/dev/null | grep "UUID" | wc -l)

if [[ ${GPU_COUNT} -ne 0 ]]; then
    export CUDA_VISIBLE_DEVICES=$(seq -s ',' 0 $((GPU_COUNT - 1)))
fi

# ENV:
#   WORKER_INDEX: worker index of a worker group
#   WORKER_SHARE_DIR: share with paired ModelWorker
#   WORKER_IMAGE: singularity image
#   WORKER_COMMAND: worker startup command
#   WORKER_LOG_DIR: worker log dir
#   REAL_PACKAGE_PATH: package path in shared storage

if [[ "${WORKER_INDEX_OFFSET}" != "" ]]; then
    WORKER_INDEX=$((WORKER_INDEX + WORKER_INDEX_OFFSET))
fi

export NODE_ID=$((${WORKER_INDEX} / 8))

#mkdir -p /tmp${WORKER_SHARE_DIR}

WORKER_LOG_DIR=${WORKER_LOG_DIR}/${UUID}
mkdir -p ${WORKER_LOG_DIR}

WORKER_LOG_FILE=${WORKER_LOG_DIR}/${WORKER_TYPE}-${WORKER_INDEX}.log

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

#systemctl stop nanovisor
systemctl stop nanovisor


# 定义同步使用的文件和路径
LOCK_FILE="/tmp/copy_hybrid_engine.lock"
DONE_FILE="/tmp/copy_hybrid_engine_done.flag"

# 同步复制函数
sync_copy() {
    # 检查是否已完成复制
    if [[ -f "$DONE_FILE" ]]; then
        log_to_both_files "Copy already completed. Proceeding."
        return 0
    fi

    # 尝试获取文件锁 (非阻塞模式)
    exec 9>"$LOCK_FILE"
    if flock -n 9; then
        log_to_both_files "Acquired lock. Starting copy..."
        # 双重检查：获取锁后再次确认状态
        if [[ ! -f "$DONE_FILE" ]]; then
            # 执行复制操作 (替换为您的实际命令)
            if cp ${WORKER_IMAGE} /tmp/image_hybrid_engine.sif; then
                touch "$DONE_FILE"
                log_to_both_files "Copy completed successfully."
            else
                log_to_both_files "Error: Copy failed!"
                exit 1
            fi
        fi
        # 释放锁
        flock -u 9
    else
        # 等待复制完成
        log_to_both_files "Waiting for copy to complete..."
        while [[ ! -f "$DONE_FILE" ]]; do
            sleep 1
        done
        log_to_both_files "Copy completed by another process. Proceeding."
    fi
}


CMD="python -m asystem_runtime.engine_server --worker-type ${WORKER_TYPE} --worker-index ${WORKER_INDEX}"


sync_copy
WORKER_IMAGE="/tmp/image_hybrid_engine.sif"
echo "before start." >> ${WORKER_LOG_FILE}
singularity --debug exec --nv --no-home --writable-tmpfs --bind /storage:/storage --bind /home/admin/logs:/home/admin/logs "${WORKER_IMAGE}" bash -c "export PATH=/opt/conda/bin:$PATH; export PYTHONPATH=${ENGINE_PACKAGE_PATH}:/home/admin/antllm:/home/admin/Megatron-LM:/home/admin/Megatron-LM/ant_utils/dcp_utils:/root/workdir/astra-build/Asystem-HybridEngine/astra_cache/astra-client/python:$PYTHONPATH;$CMD" 2>&1 | tee -a ${WORKER_ANTLOGS_LOG_FILE} >> ${WORKER_LOG_FILE}
