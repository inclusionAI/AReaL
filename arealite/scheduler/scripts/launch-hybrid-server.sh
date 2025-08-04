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

mkdir -p ${WORKER_LOG_DIR}
WORKER_LOG_FILE=${WORKER_LOG_DIR}/${WORKER_TYPE}-hybrid-${WORKER_INDEX}.log

echo "ASystem Entrypoint Env:" >> ${WORKER_LOG_FILE}

env >> ${WORKER_LOG_FILE}

echo "ASystem Worker Command: ${WORKER_COMMAND}" >> ${WORKER_LOG_FILE}

uname -a >> ${WORKER_LOG_FILE}

#systemctl stop nanovisor
systemctl stop nanovisor


# 定义同步使用的文件和路径
LOCK_FILE="/tmp/copy_hybrid_engine.lock"
DONE_FILE="/tmp/copy_hybrid_engine_done.flag"

# 同步复制函数
sync_copy() {
    # 检查是否已完成复制
    if [[ -f "$DONE_FILE" ]]; then
        echo "Copy already completed. Proceeding." >> ${WORKER_LOG_FILE}
        return 0
    fi

    # 尝试获取文件锁 (非阻塞模式)
    exec 9>"$LOCK_FILE"
    if flock -n 9; then
        echo "Acquired lock. Starting copy..." >> ${WORKER_LOG_FILE}
        # 双重检查：获取锁后再次确认状态
        if [[ ! -f "$DONE_FILE" ]]; then
            # 执行复制操作 (替换为您的实际命令)
            if cp ${WORKER_IMAGE} /tmp/image_hybrid_engine.sif; then
                touch "$DONE_FILE"
                echo "Copy completed successfully." >> ${WORKER_LOG_FILE}
            else
                echo "Error: Copy failed!" >> ${WORKER_LOG_FILE}
                exit 1
            fi
        fi
        # 释放锁
        flock -u 9
    else
        # 等待复制完成
        echo "Waiting for copy to complete..." >> ${WORKER_LOG_FILE}
        while [[ ! -f "$DONE_FILE" ]]; do
            sleep 1
        done
        echo "Copy completed by another process. Proceeding." >> ${WORKER_LOG_FILE}
    fi
}


CMD="python -m asystem_runtime.engine_server"


sync_copy
WORKER_IMAGE="/tmp/image_hybrid_engine.sif"
echo "before start." >> ${WORKER_LOG_FILE}
singularity --debug exec --pid --nv --no-home --writable-tmpfs --bind /storage:/storage "${WORKER_IMAGE}" bash -c "export PATH=/opt/conda/bin:$PATH; export PYTHONPATH=${ENGINE_PACKAGE_PATH}:/storage/openpsi/codes/antllm:/storage/openpsi/codes/atorch:/storage/openpsi/codes/Megatron-LM:/storage/openpsi/codes/Megatron-LM/ant_utils/dcp_utils:$PYTHONPATH;$CMD" >> ${WORKER_LOG_FILE} 2>&1
