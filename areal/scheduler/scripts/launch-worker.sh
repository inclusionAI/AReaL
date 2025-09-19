#!/usr/bin/env bash

set -e

GPU_COUNT=$(nvidia-smi -L 2>/dev/null | grep "UUID" | wc -l)

if [[ ${GPU_COUNT} -ne 0 ]]; then
    export CUDA_VISIBLE_DEVICES=$(seq -s ',' 0 $((GPU_COUNT - 1)))
fi

export GLOO_RDMA_TCP_READ_TIMEOUT=3000000
export GLOO_RDMA_TCP_CONNECT_TIMEOUT=3000000
#export GLOO_SOCKET_IFNAME=eth0
export GLOO_TIMEOUT_SECONDS=1800

# ENV:
#   WORKER_INDEX: worker index of a worker group
#   WORKER_COUNT: worker count of a worker group
#   WORKER_IMAGE: singularity image
#   WORKER_COMMAND: worker startup command
#   WORKER_LOG_DIR: worker log dir
#   WORKER_INDEX_OFFSET: offset from WORKER_INDEX
#   WORKER_TOTAL_COUNT: total worker count of worker groups which have same worker type
#   REAL_PACKAGE_PATH: package path in shared storage

if [[ "${WORKER_INDEX_OFFSET}" != "" ]]; then
    WORKER_INDEX=$((WORKER_INDEX + WORKER_INDEX_OFFSET))
fi

if [[ "${WORKER_TOTAL_COUNT}" != "" ]]; then
    WORKER_COUNT=${WORKER_TOTAL_COUNT}
fi

JOBSTEP_ID='{jobstep_id}'
N_JOBSTEPS='{n_jobsteps}'
WORKER_SUBMISSION_INDEX='{worker_submission_index}'
WPROCS_PER_JOBSTEP='{wprocs_per_jobstep}'
WPROCS_IN_JOB='{wprocs_in_job}'
WPROC_OFFSET='{wproc_offset}'

WORKER_COMMAND="/usr/bin/python -u -m areal.scheduler.rpc.rpc_server --worker-type ${WORKER_TYPE} --worker-index ${WORKER_INDEX}"

#log output to local worker dir
LOCAL_WORKER_DIR=/home/admin/logs/experiment/
mkdir -p ${LOCAL_WORKER_DIR}
LOCAL_WORKER_MONITOR_LOG_FILE=${LOCAL_WORKER_DIR}/${WORKER_TYPE}-${WORKER_INDEX}-monitor.log
touch ${LOCAL_WORKER_MONITOR_LOG_FILE}

WORKER_LOG_DIR=${WORKER_LOG_DIR}/${UUID}
mkdir -p ${WORKER_LOG_DIR}

WORKER_LOG_FILE=${WORKER_LOG_DIR}/${WORKER_TYPE}-${WORKER_INDEX}.log

mkdir -p ${LOCAL_WORKER_DIR}/${EXPR_NAME}/${TRIAL_NAME}
WORKER_ANTLOGS_LOG_FILE=${LOCAL_WORKER_DIR}/${EXPR_NAME}/${TRIAL_NAME}/${WORKER_TYPE}-${WORKER_INDEX}.log

log_to_both_files() {
    echo "$@" | tee -a ${WORKER_ANTLOGS_LOG_FILE} >> ${WORKER_LOG_FILE}
}

log_to_both_files "ASystem Entrypoint Env:"
env >> ${WORKER_LOG_FILE}
env >> ${WORKER_ANTLOGS_LOG_FILE}
log_to_both_files "ASystem Worker Command: ${WORKER_COMMAND}"

cd ${REAL_PACKAGE_PATH}

log_to_both_files `uname -a`

#systemctl stop nanovisor
systemctl stop nanovisor

# 定义同步使用的文件和路径
LOCK_FILE="/tmp/copy.lock"
DONE_FILE="/tmp/copy_done.flag"

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
            if cp ${WORKER_IMAGE} /tmp/image.sif; then
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

sync_copy
WORKER_IMAGE="/tmp/image.sif"

singularity --debug exec --pid --nv --no-home --writable-tmpfs --bind /storage:/storage --workdir "${REAL_PACKAGE_PATH}" "${WORKER_IMAGE}" bash -c "${WORKER_COMMAND}" 2>&1 | tee -a ${WORKER_LOG_FILE} | tee -a ${WORKER_ANTLOGS_LOG_FILE} | grep --line-buffered -F "monitor_logger" >> ${LOCAL_WORKER_MONITOR_LOG_FILE}




