# Copyright 2025 Ant Group Inc.
# Copyright 2024 Wei Fu & Zhiyu Mei
# Licensed under the Apache License, Version 2.0 (the "License").

import pickle
import socket
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import ray.util.queue as rq
import zmq

import realhf.system.worker_base as worker_base
from realhf.base import logging
from realhf.system.worker_base import WorkerServerStatus

logger = logging.getLogger("worker-control")
WORKER_WAIT_FOR_CONTROLLER_SECONDS = 3600
WORKER_JOB_STATUS_LINGER_SECONDS = 60


class ZmqTaskQueue(worker_base.WorkerServerTaskQueue):

    def __init__(self, port=0):
        self.__context = zmq.Context()
        self.__socket = self.__context.socket(zmq.REP)
        host_ip = socket.gethostbyname(socket.gethostname())
        if port == 0:
            self.__port = self.__socket.bind_to_random_port(f"tcp://{host_ip}")
        else:
            self.__socket.bind(f"tcp://{host_ip}:{port}")
            self.__port = port

    def __del__(self):
        self.__socket.close()

    @property
    def port(self):
        return self.__port

    def try_get_request(self) -> Tuple[str, Dict[str, Any]]:
        try:
            data = self.__socket.recv(zmq.NOBLOCK)
        except zmq.ZMQError:
            # Currently no request in the queue.
            raise worker_base.NoRequstForWorker()
        return pickle.loads(data)

    def respond(self, response):
        self.__socket.send(pickle.dumps(response))


class RayTaskQueue(worker_base.WorkerServerTaskQueue):

    def __init__(self, comm: Tuple[rq.Queue, rq.Queue]):
        recv_queue, send_queue = comm
        self.__recv_queue = recv_queue
        self.__send_queue = send_queue

    def try_get_request(self) -> Tuple[str, Dict[str, Any]]:
        try:
            command, kwargs = self.__recv_queue.get_nowait()
        except rq.Empty:
            # Currently no request in the queue.
            raise worker_base.NoRequstForWorker()
        return command, kwargs

    def respond(self, response):
        self.__send_queue.put(response)


class ZmqRequester(worker_base.WorkerControlPanelRequester):

    class ZmqFuture(worker_base.WorkerControlPanelRequester.Future):
        # Every ZmqFuture connect one socket, close after returning results.
        def __init__(
            self,
            payload,
            context: zmq.Context,
            address,
            worker_name,
            wait_response=True,
        ):
            self.__worker_name = worker_name
            self.__socket = context.socket(zmq.REQ)
            self.__socket.setsockopt(zmq.LINGER, 0)
            self.__socket.connect(f"tcp://{address}")
            self.__socket.send(payload, flags=zmq.NOBLOCK)
            if not wait_response:
                self.__socket.close()

        def result(self, timeout=None):
            if timeout is not None:
                self.__socket.RCVTIMEO = int(timeout * 1000)
            else:
                self.__socket.RCVTIMEO = int(1e9)
            try:
                r = pickle.loads(self.__socket.recv())
            except zmq.error.Again as e:
                raise TimeoutError(f"Waiting for RPC server response timeout: {e}")
            if isinstance(r, Exception):
                logger.error(f"Error configuring worker {self.__worker_name}")
                raise r
            self.__socket.close()
            return r

    def __init__(self):
        self.__context = zmq.Context()
        self.__context.set(zmq.MAX_SOCKETS, 20480)

    def async_request(
        self, worker_name, address, command, wait_response=True, **kwargs
    ):
        payload = pickle.dumps((command, kwargs))
        r = self.ZmqFuture(
            payload,
            self.__context,
            address,
            worker_name,
            wait_response=wait_response,
        )
        return r


class RayRequester(worker_base.WorkerControlPanelRequester):

    class RayQueueFuture(worker_base.WorkerControlPanelRequester.Future):

        def __init__(self, worker_name: str, queue: rq.Queue):
            self.__queue = queue
            self.__worker_name = worker_name

        def result(self, timeout=None):
            try:
                return self.__queue.get(timeout=timeout)
            except rq.Empty:
                raise TimeoutError(
                    f"Waiting for Ray worker {self.__worker_name} response timeout."
                )
            except Exception as e:
                raise RuntimeError(
                    f"Error waiting for Ray queue future {self.__worker_name}."
                ) from e

    def __init__(
        self,
        request_comms: Dict[str, rq.Queue],
        reply_comms: Dict[str, rq.Queue],
    ):
        self.__request_comms: Dict[str, rq.Queue] = request_comms
        self.__reply_comms: Dict[str, rq.Queue] = reply_comms

    def async_request(self, worker_name, _, command, __, **kwargs):
        request_queue = self.__request_comms[worker_name]
        request_queue.put((command, kwargs))
        reply_queue = self.__reply_comms[worker_name]
        return self.RayQueueFuture(worker_name, reply_queue)


def make_server(type_, worker_name, experiment_name, trial_name, **kwargs):
    if type_ == "zmq":
        q = ZmqTaskQueue(**kwargs)
    elif type_ == "ray":
        q = RayTaskQueue(**kwargs)
    else:
        raise NotImplementedError(type_)
    return worker_base.WorkerServer(worker_name, experiment_name, trial_name, q)


def make_control(type_, experiment_name, trial_name, **kwargs):
    if type_ == "zmq":
        requester = ZmqRequester(**kwargs)
    elif type_ == "ray":
        requester = RayRequester(**kwargs)
    else:
        raise NotImplementedError(type_)
    return worker_base.WorkerControlPanel(experiment_name, trial_name, requester)
