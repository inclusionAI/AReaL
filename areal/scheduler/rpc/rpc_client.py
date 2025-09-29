import gzip
import time
from http import HTTPStatus
from typing import Any, Union

import cloudpickle
import requests

from areal.api.cli_args import InferenceEngineConfig, TrainEngineConfig
from areal.api.engine_api import InferenceEngine, TrainEngine
from areal.utils import logging
from areal.utils.http import response_ok, response_retryable

logger = logging.getLogger("RPCClient")


class RPCClient:
    def __init__(self):
        self._addrs = {}

    def register(self, worker_id: str, ip: str, port: int) -> None:
        self._addrs[worker_id] = (ip, port)
        logger.info(f"Registered worker {worker_id} at {ip}:{port}")

    def create_engine(
        self,
        worker_id: str,
        engine_obj: Union[InferenceEngine, TrainEngine],
        init_config: Union[InferenceEngineConfig, TrainEngineConfig],
    ) -> None:
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
        if resp.status_code == HTTPStatus.OK:
            logger.info(f"create engine success.")
            return cloudpickle.loads(resp.content)
        else:
            logger.error(f"Failed to create engine, {resp.status_code}, {resp.content}")
            raise RuntimeError(
                f"Failed to create engine, {resp.status_code}, {resp.content}"
            )

    def call_engine(
        self, worker_id: str, method: str, max_retries: int = 3, *args, **kwargs
    ) -> Any:
        """
        call the rpc server with method name and args, retry on failure

        Parameters
        ----------
        worker_id: str
            the id of the worker to call
        method: str
            the method name to call
        max_retries: int
            max retries on failure
        *args:
            args to pass to the method
        **kwargs:
            kwargs to pass to the method

        Returns
        -------
            the deserialized result from the rpc server
        """
        req = (method, args, kwargs)
        serialized_data = cloudpickle.dumps(req)

        return self._call_engine_with_serialized_data(
            worker_id, serialized_data, max_retries
        )

    def _call_engine_with_serialized_data(
        self, worker_id: str, serialized_data: bytes, max_retries=3
    ) -> Any:
        """
        call the rpc server with serialized data, retry on failure

        Parameters
        ----------
        worker_id: str
            the id of the worker to call
        serialized_data: bytes
            the serialized data to send
        max_retries: int
            max retries on failure

        Returns
        -------
            the deserialized result from the rpc server
        """
        ip, port = self._addrs[worker_id]
        url = f"http://{ip}:{port}/call"
        last_exception = None

        for attempt in range(max_retries):
            try:
                resp = requests.post(url, data=serialized_data, timeout=7200)
                logger.info(
                    f"Sent call to {worker_id} ({ip}:{port}), status={resp.status_code}, attempt {attempt + 1}/{max_retries}"
                )

                if response_ok(resp.status_code):
                    return cloudpickle.loads(resp.content)
                elif response_retryable(resp.status_code):
                    last_exception = RuntimeError(
                        f"Retryable HTTP status {resp.status_code}: {resp.content}"
                    )
                else:
                    raise RuntimeError(
                        f"Non-retryable HTTP error: {resp.status_code} - {resp.content}"
                    )

            except (RuntimeError, TimeoutError) as e:
                logger.error(f"stop retrying, error on attempt {attempt + 1}: {e}")
                raise e
            except Exception as e:
                last_exception = e
                logger.error(f"error on attempt {attempt + 1}: {e}")

            if last_exception is not None:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Retrying in 1 second... ({attempt + 1}/{max_retries})"
                    )
                    time.sleep(1)
                    continue
                else:
                    logger.error(f"Max retries exceeded for {url}")
                    raise last_exception
