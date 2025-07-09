import os
import requests
import pickle
import logging
import inspect
from http.server import BaseHTTPRequestHandler, HTTPServer
from areallite.scheduler.utils import serialize_with_metadata


class RPCClient:
    def __init__(self):
        self._addrs = {}

    def register(self, worker_id, ip, port):
        self._addrs[worker_id] = (ip, port)
        logging.info(f"Registered worker {worker_id} at {ip}:{port}")

    def create_engine(self, worker_id, engine_obj, init_config):
        ip, port = self._addrs[worker_id]
        url = f"http://{ip}:{port}/create_engine"

        serialized_obj = serialize_with_metadata(engine_obj, init_config)
        resp = requests.post(url, data=serialized_obj)
        logging.info(
            f"Sent create_engine to {worker_id} ({ip}:{port}), status={resp.status_code}"
        )
        return resp.status_code == 200

    def call(self, worker_id, method, arg):
        ip, port = self._addrs[worker_id]
        url = f"http://{ip}:{port}/call"
        serialized_obj = serialize_with_metadata(arg)
        req = {"method": method, "arg": serialized_obj}
        resp = requests.post(url, data=pickle.dumps(req))
        logging.info(
            f"Sent call '{method}' to {worker_id} ({ip}:{port}), status={resp.status_code}"
        )
        if resp.status_code == 200:
            return pickle.loads(resp.content)
        else:
            raise RuntimeError(f"RPC call failed: {resp.content}")
