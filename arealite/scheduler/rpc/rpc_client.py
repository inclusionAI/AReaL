import gzip
import inspect
import os
import pickle
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import cloudpickle
import requests

from realhf.base import logging

logger = logging.getLogger("RPCClient")


class RPCClient:
    def __init__(self):
        self._addrs = {}

    def register(self, worker_id, ip, port):
        self._addrs[worker_id] = (ip, port)
        logger.info(f"Registered worker {worker_id} at {ip}:{port}")

    def create_engine(self, worker_id, engine_obj, init_config):
        ip, port = self._addrs[worker_id]
        url = f"http://{ip}:{port}/create_engine"
        logger.info(f"send create_engine to {worker_id} ({ip}:{port})")
        payload = (engine_obj, init_config)
        serialized_data = cloudpickle.dumps(payload)
        serialized_obj = gzip.compress(serialized_data)
        resp = requests.post(url, data=serialized_obj)
        logger.info(
            f"send create_engine to {worker_id} ({ip}:{port}), status={resp.status_code}"
        )
        if resp.status_code == 200:
            logger.info(f"create engine success.")
            return cloudpickle.loads(resp.content)
        else:
            logger.error(f"Failed to create engine, {resp.status_code}, {resp.content}")
            raise RuntimeError(
                f"Failed to create engine, {resp.status_code}, {resp.content}"
            )

    def call_engine(self, worker_id, method, *args, **kwargs):
        ip, port = self._addrs[worker_id]
        url = f"http://{ip}:{port}/call"
        # 支持变长参数
        req = (method, args, kwargs)

        serialized_data = cloudpickle.dumps(req)
        resp = requests.post(url, data=serialized_data, timeout=7200)
        logger.info(
            f"Sent call '{method}' to {worker_id} ({ip}:{port}), status={resp.status_code}"
        )
        if resp.status_code == 200:
            return cloudpickle.loads(resp.content)
        elif resp.status_code == 503:
            raise TimeoutError(f"RPC call timeout: {resp.content}")
        else:
            raise RuntimeError(f"RPC call failed: {resp.content}")

    def call_engine_with_serialized_data(self, worker_id: str, serialized_data: bytes):
        """
        数据面调用（带序列化数据）
        """
        ip, port = self._addrs[worker_id]
        url = f"http://{ip}:{port}/call"

        resp = requests.post(url, data=serialized_data, timeout=7200)
        logger.info(
            f"Sent call to {worker_id} ({ip}:{port}), status={resp.status_code}"
        )
        if resp.status_code == 200:
            return cloudpickle.loads(resp.content)
        elif resp.status_code == 503:
            raise TimeoutError(f"RPC call timeout: {resp.content}")
        else:
            raise RuntimeError(f"RPC call failed: {resp.content}")
