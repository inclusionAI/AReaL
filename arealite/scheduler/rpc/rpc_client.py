import gzip
import os
import resource
import sys
import time

import requests
import pickle
import logging
import inspect
from http.server import BaseHTTPRequestHandler, HTTPServer
import cloudpickle
from pympler import asizeof

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
        print(f"call engine0: {method}, args: {args} MB, kwargs: {kwargs}",flush=True)
        mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f"call engine1: {method}, 内存使用: {mem_usage / 1024:.2f} MB, req size: {asizeof.asizeof(req)}", flush=True)

        serialized_data = cloudpickle.dumps(req)
        mem_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print(f"call engine2: {method}, 内存使用: {mem_usage / 1024:.2f} MB， serialized_data: {sys.getsizeof(serialized_data)}", flush=True)

        resp = requests.post(url, data=serialized_data, timeout=7200)
        logging.error(
           f"Sent call '{method}' to {worker_id} ({ip}:{port}), status={resp.status_code}"
        )
        if resp.status_code == 200:
            return cloudpickle.loads(resp.content)
        else:
            raise RuntimeError(f"RPC call failed: {resp.content}")


