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

WORKER_COMMAND="/usr/bin/python -u -m arealite.scheduler.rpc.rpc_server"

#log output to local worker dir
LOCAL_WORKER_DIR=/home/admin/logs/experiment/
mkdir -p ${LOCAL_WORKER_DIR}
LOCAL_WORKER_MONITOR_LOG_FILE=${LOCAL_WORKER_DIR}/${WORKER_TYPE}-${WORKER_INDEX}-monitor.log
touch ${LOCAL_WORKER_MONITOR_LOG_FILE}

mkdir -p ${WORKER_LOG_DIR}

WORKER_LOG_FILE=${WORKER_LOG_DIR}/${WORKER_TYPE}-${WORKER_INDEX}.log

echo "ASystem Entrypoint Env:" >> ${WORKER_LOG_FILE}

env >> ${WORKER_LOG_FILE}

echo "ASystem Worker Command: ${WORKER_COMMAND}" >> ${WORKER_LOG_FILE}

cd ${REAL_PACKAGE_PATH}

uname -a >> ${WORKER_LOG_FILE}

#systemctl stop nanovisor
systemctl stop nanovisor


# 定义同步使用的文件和路径
LOCK_FILE="/tmp/copy.lock"
DONE_FILE="/tmp/copy_done.flag"

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
            if cp ${WORKER_IMAGE} /tmp/image.sif; then
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

sync_copy
WORKER_IMAGE="/tmp/image.sif"

singularity --debug exec --pid --nv --no-home --writable-tmpfs --bind /storage:/storage --workdir "${REAL_PACKAGE_PATH}" "${WORKER_IMAGE}" bash -c "${WORKER_COMMAND}" 2>&1 | tee -a ${WORKER_LOG_FILE} | grep --line-buffered -F 'monitor_logger' >> ${LOCAL_WORKER_MONITOR_LOG_FILE}




