import asyncio
import gzip
from http import HTTPStatus

import aiohttp
import cloudpickle
import requests

from areal.utils import logging
from areal.utils.http import response_ok, response_retryable

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
        if resp.status_code == HTTPStatus.OK:
            logger.info("create engine success.")
            return cloudpickle.loads(resp.content)
        else:
            logger.error(f"Failed to create engine, {resp.status_code}, {resp.content}")
            raise RuntimeError(
                f"Failed to create engine, {resp.status_code}, {resp.content}"
            )

    def call_engine(self, worker_id, method, max_retries=3, *args, **kwargs):
        ip, port = self._addrs[worker_id]
        # 支持变长参数
        req = (method, args, kwargs)
        serialized_data = cloudpickle.dumps(req)

        return self.call_engine_with_serialized_data(
            worker_id, serialized_data, max_retries
        )

    async def async_create_engine(self, worker_id, engine, *args, **kwargs):
        """异步创建engine实例"""
        ip, port = self._addrs[worker_id]
        url = f"http://{ip}:{port}/create_engine"
        logger.info(f"Async send create_engine to {worker_id} ({ip}:{port})")

        # 构建payload：engine类路径和初始化参数
        payload = (engine, args, kwargs)
        serialized_data = cloudpickle.dumps(payload)
        serialized_obj = gzip.compress(serialized_data)

        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=serialized_obj) as resp:
                logger.info(
                    f"Async send create_engine to {worker_id} ({ip}:{port}), status={resp.status}"
                )
                if resp.status == HTTPStatus.OK:
                    content = await resp.read()
                    logger.info("Async create engine success.")
                    return cloudpickle.loads(content)
                else:
                    content = await resp.text()
                    logger.error(
                        f"Failed to async create engine, {resp.status}, {content}"
                    )
                    raise RuntimeError(
                        f"Failed to async create engine, {resp.status}, {content}"
                    )

    async def async_call_engine(
        self, worker_id, method, max_retries=3, *args, **kwargs
    ):
        """异步调用engine方法"""
        ip, port = self._addrs[worker_id]
        # 支持变长参数
        req = (method, args, kwargs)
        serialized_data = cloudpickle.dumps(req)

        return await self.async_call_engine_with_serialized_data(
            worker_id, serialized_data, max_retries
        )

    async def async_call_engine_with_serialized_data(
        self, worker_id: str, serialized_data: bytes, max_retries=3
    ):
        """
        异步数据面调用（带序列化数据），支持异常和503状态码重试
        """
        ip, port = self._addrs[worker_id]
        url = f"http://{ip}:{port}/call"
        last_exception = None

        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url, data=serialized_data, timeout=7200
                    ) as resp:
                        logger.info(
                            f"Async sent call to {worker_id} ({ip}:{port}), status={resp.status}, attempt {attempt + 1}/{max_retries}"
                        )

                        if response_ok(resp.status):
                            content = await resp.read()
                            return cloudpickle.loads(content)
                        elif response_retryable(resp.status):
                            last_exception = RuntimeError(
                                f"Retryable HTTP status {resp.status}: {await resp.text()}"
                            )
                        else:
                            content = await resp.text()
                            raise RuntimeError(
                                f"Non-retryable HTTP error: {resp.status} - {content}"
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
                    await asyncio.sleep(1)
                    continue
                else:
                    logger.error(f"Max retries exceeded for {url}")
                    raise last_exception
