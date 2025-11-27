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
#   IMAGE: singularity image
#   WORKER_COMMAND: worker startup command
#   LOG_DIR: worker log dir
#   WORKER_INDEX_OFFSET: offset from WORKER_INDEX
#   WORKER_TOTAL_COUNT: total worker count of worker groups which have same worker type
#   REAL_PACKAGE_PATH: package path in shared storage
#   PORT_LIST: comma-separated list of ports (use first port for RPC server)

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
WORKER_TYPE=${ROLE}-${TYPE}

# 从 PORT_LIST 环境变量获取端口号，使用第一个端口作为 RPC server 端口
if [[ -n "${PORT_LIST}" ]]; then
    # 将逗号分隔的端口列表转换为数组
    IFS=',' read -ra PORTS <<< "${PORT_LIST}"
    # 获取第一个端口
    FIRST_PORT="${PORTS[0]}"
    # 添加到 WORKER_COMMAND
    WORKER_COMMAND="/usr/bin/python -u -m areal.scheduler.rpc.rpc_server --worker-type ${WORKER_TYPE} --worker-index ${WORKER_INDEX} --port ${FIRST_PORT}"
else
    WORKER_COMMAND="/usr/bin/python -u -m areal.scheduler.rpc.rpc_server --worker-type ${WORKER_TYPE} --worker-index ${WORKER_INDEX}"
fi

#log output to local worker dir
LOCAL_WORKER_DIR=/home/admin/logs/experiment/
mkdir -p ${LOCAL_WORKER_DIR}
LOCAL_WORKER_MONITOR_LOG_FILE=${LOCAL_WORKER_DIR}/${WORKER_TYPE}-${WORKER_INDEX}-monitor.log
touch ${LOCAL_WORKER_MONITOR_LOG_FILE}

LOG_DIR=${LOG_DIR}/${UUID}
mkdir -p ${LOG_DIR}

WORKER_LOG_FILE=${LOG_DIR}/${WORKER_TYPE}-${WORKER_INDEX}.log

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

singularity --debug exec --pid --nv --no-home --writable-tmpfs --bind /storage:/storage "${IMAGE}" bash -c "pip uninstall -q -y huggingface-hub && pip install -q huggingface-hub==0.34.0 tensordict==0.10.0 gymnasium; cd ${REAL_PACKAGE_PATH}; export PYTHONPATH=${EXTRA_PYTHONPATH}:$PYTHONPATH; ${WORKER_COMMAND}" 2>&1 | tee -a ${WORKER_LOG_FILE} | tee -a ${WORKER_ANTLOGS_LOG_FILE} | grep --line-buffered -F "monitor_logger" >> ${LOCAL_WORKER_MONITOR_LOG_FILE}

