import gzip
import os
import requests
import pickle
import logging
import inspect
from http.server import BaseHTTPRequestHandler, HTTPServer
from arealite.scheduler.utils import serialize_with_metadata
import cloudpickle


class RPCClient:
    def __init__(self):
        self._addrs = {}

    def register(self, worker_id, ip, port):
        self._addrs[worker_id] = (ip, port)
        logging.info(f"Registered worker {worker_id} at {ip}:{port}")

    def create_engine(self, worker_id, engine_obj, init_config):
        ip, port = self._addrs[worker_id]
        url = f"http://{ip}:{port}/create_engine"
        print(f"[RPCClient] Sent create_engine to {worker_id} ({ip}:{port})")
        payload = (engine_obj, init_config)
        serialized_data = cloudpickle.dumps(payload)
        serialized_obj = gzip.compress(serialized_data)
        resp = requests.post(url, data=serialized_obj)
        print(
            f"[RPCClient] Sent create_engine to {worker_id} ({ip}:{port}), status={resp.status_code}"
        )
        return resp.status_code == 200

    def call_engine(self, worker_id, method, *args, **kwargs):
        ip, port = self._addrs[worker_id]
        url = f"http://{ip}:{port}/call"
        # 支持变长参数
        req = (method, args, kwargs)
        serialized_data = cloudpickle.dumps(req)
        serialized_req = gzip.compress(serialized_data)

        resp = requests.post(url, data=serialized_req, timeout=7200)
        logging.error(
           f"Sent call '{method}' to {worker_id} ({ip}:{port}), status={resp.status_code}"
        )
        if resp.status_code == 200:
            return cloudpickle.loads(resp.content)
        else:
            raise RuntimeError(f"RPC call failed: {resp.content}")

