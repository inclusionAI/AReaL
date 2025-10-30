# 用来在 trainer 退出时清理相关 Job，适用于 AIStudio
# AIStudio 侧需要把 aistudio_pre_stop.py 复制到 /workspace/bin，作为 pre stop hook 执行。

import logging

import requests
from requests import RequestException

logger = logging.getLogger("cleanupjobs")

ENDPOINT = "http://asystem-scheduler.asystem-cluster-prod-1.svc:8081"
API_BASE_PATH = "/api/v1"

with open("/tmp/job_uids.txt") as f:
    uids = [line.strip() for line in f if line.strip()]
for uid in uids:
    logger.info(f"Stopping job with UID: {uid}")
    url = f"{ENDPOINT}/{API_BASE_PATH}/AsystemJobs/{uid}/cancel"
    try:
        response = requests.post(url, timeout=10)
        response.raise_for_status()
        if response.status_code == 200:
            logger.info(f"succeeded to stop job with UID: {uid}")
    except RequestException as e:
        logger.info(f"failed to issue request {url}: {e}")
