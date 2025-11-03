import asyncio
from typing import Any

import httpx
import orjson

from areal.scheduler.exceptions import (
    EngineCallError,
    EngineCreationError,
    EngineImportError,
    RPCConnectionError,
)
from areal.scheduler.rpc.serialization import deserialize_value, serialize_value
from areal.utils import logging
from areal.utils.http import response_retryable

logger = logging.getLogger("RPCClient")


class RPCClient:
    def __init__(self):
        self._addrs = {}
        self._async_http_client = httpx.AsyncClient(timeout=7200.0)  # Async client

    def register(self, worker_id, ip, port):
        self._addrs[worker_id] = (ip, port)
        logger.info(f"Registered worker {worker_id} at {ip}:{port}")

    async def async_create_engine(self, worker_id, engine, *args, **kwargs):
        """
        Create an engine instance(we called it worker in Asystem) on a remote worker.

        The engine parameter is a string import path (e.g., "areal.engine.ppo.actor.FSDPPPOActor")
        that will be dynamically imported and instantiated on the worker.

        Args:
            worker_id: Worker ID in format "role/index"
            engine: Import path to the engine class (e.g., "areal.engine.ppo.actor.FSDPPPOActor")
            *args: Initialization arguments
            *kwargs: Initialization keyword arguments

        Returns:
            Result from engine initialization

        Raises:
            WorkerNotFoundError: If worker doesn't exist
            WorkerFailedError: If worker process has failed
            EngineCreationError: If engine creation fails
        """
        # Validate engine is a string import path
        if not isinstance(engine, str):
            raise EngineCreationError(
                worker_id,
                f"Engine must be a string import path, got {type(engine)}",
            )
        # Build JSON payload with serialized args and kwargs
        payload = {
            "engine": engine,
            "init_args": serialize_value(list(args)),
            "init_kwargs": serialize_value(kwargs),
        }

        ip, port = self._addrs[worker_id]
        url = f"http://{ip}:{port}/create_engine"
        logger.info(f"Async send create_engine to {worker_id} ({ip}:{port})")

        max_retries = 30
        last_exception = None

        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Creating engine '{engine}' on worker '{worker_id}' (attempt {attempt + 1}/{max_retries})"
                )

                response = await self._async_http_client.post(
                    url,
                    content=orjson.dumps(payload),
                    headers={"Content-Type": "application/json"},
                    timeout=300.0,
                )

                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"Engine created successfully on worker '{worker_id}'")
                    return result.get("result")
                elif response.status_code == 400:
                    # Import error or bad request
                    error_detail = response.json().get("detail", "Unknown error")
                    if "Failed to import" in error_detail:
                        raise EngineImportError(engine, error_detail)
                    else:
                        raise EngineCreationError(worker_id, error_detail, 400)
                elif response.status_code == 500:
                    # Engine initialization failed
                    error_detail = response.json().get("detail", "Unknown error")
                    raise EngineCreationError(worker_id, error_detail, 500)
                elif response_retryable(response.status_code):
                    # Retryable HTTP status codes
                    error_detail = response.json().get("detail", "Unknown error")
                    last_exception = EngineCreationError(
                        worker_id,
                        f"Retryable HTTP status {response.status_code}: {error_detail}",
                        response.status_code,
                    )
                else:
                    raise EngineCreationError(
                        worker_id,
                        f"Unexpected status code: {response.status_code}",
                        response.status_code,
                    )

            except httpx.ConnectError as e:
                # Network connection errors are retryable
                last_exception = EngineCreationError(
                    worker_id, f"Connection error: {str(e)}"
                )
                logger.warning(f"Connection error on attempt {attempt + 1}: {e}")

            except httpx.TimeoutException as e:
                # Timeout errors are retryable
                last_exception = EngineCreationError(
                    worker_id, f"Request timed out: {e}"
                )
                logger.error(f"Timeout error on attempt {attempt + 1}: {e}")

            except (EngineCreationError, EngineImportError, RPCConnectionError) as e:
                # Non-retryable errors
                logger.error(f"Non-retryable error on attempt {attempt + 1}: {e}")
                raise e

            except Exception as e:
                # Other unexpected errors
                last_exception = EngineCreationError(
                    worker_id, f"Unexpected error: {str(e)}"
                )
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")

            # Check if we should retry
            if last_exception is not None:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Retrying create_engine in 5 second... ({attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(5)
                    continue
                else:
                    logger.error(f"Max retries exceeded for create_engine at {url}")
                    raise last_exception

        # This should never be reached, but for safety
        raise EngineCreationError(
            worker_id, "Unexpected error: retry loop completed without result"
        )

    def call_engine(self, worker_id, method, max_retries=3, *args, **kwargs):
        """
        Call a method on an engine instance on a remote worker.

        Args:
            worker_id: Worker ID in format "role/index"
            method: Method name to call
            *args: Method arguments
            max_retries: Maximum number of retry attempts
            **kwargs: Method keyword arguments

        Returns:
            Result from method call

        Raises:
            WorkerNotFoundError: If worker doesn't exist
            WorkerFailedError: If worker process has failed
            EngineCallError: If method call fails
        """
        return asyncio.run(
            self.async_call_engine(worker_id, method, max_retries, *args, **kwargs)
        )

    async def async_call_engine(
        self, worker_id, method, max_retries=3, *args, **kwargs
    ):
        if method == "run_workflow":
            # Serialize kwargs for workflow execution
            payload = serialize_value(kwargs)
        elif method == "export_stats":
            payload = None
        else:
            # Serialize args and kwargs
            serialized_args = serialize_value(list(args))
            serialized_kwargs = serialize_value(kwargs)
            payload = {
                "method": method,
                "args": serialized_args,
                "kwargs": serialized_kwargs,
            }

        return await self.async_call_engine_with_serialized_data(
            worker_id, method, payload, max_retries
        )

    async def async_call_engine_with_serialized_data(
        self,
        worker_id: str,
        method: str,
        payload: Any,
        max_retries: int = 3,
        retry_delay: float = 10.0,
    ) -> Any:
        """
        Async version of call_engine for calling engine methods asynchronously.

        Args:
            worker_id: Worker ID in format "role/index"
            method: Method name to call
            *args: Method arguments
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (exponential backoff)
            **kwargs: Method keyword arguments

        Returns:
            Result from method call

        Raises:
            WorkerNotFoundError: If worker doesn't exist
            WorkerFailedError: If worker process has failed
            EngineCallError: If method call fails
        """
        ip, port = self._addrs[worker_id]
        base_url = f"http://{ip}:{port}"
        if method == "run_workflow":
            # Special routing for workflow execution
            url = f"{base_url}/run_workflow"
        elif method == "export_stats":
            url = f"{base_url}/export_stats"
        else:
            # Standard engine method call
            url = f"{base_url}/call"

        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    f"Async calling method '{method}' on worker '{worker_id}' (attempt {attempt})"
                )

                response = await self._async_http_client.post(
                    url,
                    content=orjson.dumps(payload),
                    headers={"Content-Type": "application/json"},
                    timeout=7200.0,  # 2 hours for long-running operations
                )

                result, should_retry, error_msg = self._handle_call_response(
                    response, worker_id, method, attempt
                )
                if not should_retry:
                    if attempt > 1:
                        logger.info(
                            f"Method '{method}' succeeded on worker '{worker_id}' "
                            f"after {attempt} attempts"
                        )
                    return result
                last_error = error_msg

            except Exception as e:
                last_error = self._handle_call_exception(e, worker_id)

            # Retry with exponential backoff
            if attempt < max_retries:
                delay = retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"Method '{method}' failed on worker '{worker_id}' "
                    f"(attempt {attempt}/{max_retries}): {last_error}. "
                    f"Retrying in {delay:.1f}s..."
                )
                import asyncio

                await asyncio.sleep(delay)

        # All retries exhausted
        raise EngineCallError(
            worker_id,
            method,
            last_error or "Max retries exceeded",
            attempt=max_retries,
        )

    def _handle_call_response(
        self, response, worker_id: str, method: str, attempt: int
    ):
        """
        Handle HTTP response from engine call.

        Args:
            response: HTTP response object
            worker_id: Worker ID
            method: Method name being called
            attempt: Current retry attempt number

        Returns:
            Tuple of (result, should_retry, error_message)
            - result: The result from the call if successful, None otherwise
            - should_retry: Whether to retry the request
            - error_message: Error message if failed, None if successful
        """
        if response.status_code == 200:
            result = response.json().get("result")
            # Deserialize result (convert SerializedTensor dicts back to tensors)
            deserialized_result = deserialize_value(result)
            return deserialized_result, False, None
        elif response.status_code == 400:
            # Bad request (e.g., method doesn't exist) - don't retry
            error_detail = response.json().get("detail", "Unknown error")
            raise EngineCallError(worker_id, method, error_detail, attempt)
        elif response.status_code == 500:
            # Engine method failed - don't retry
            error_detail = response.json().get("detail", "Unknown error")
            raise EngineCallError(worker_id, method, error_detail, attempt)
        elif response.status_code == 503:
            # Service unavailable - retry
            return None, True, "Service unavailable"
        else:
            # Other errors - retry
            return None, True, f"HTTP {response.status_code}: {response.text}"

    def _handle_call_exception(self, e: Exception, worker_id: str) -> str:
        """
        Handle exceptions during engine calls and return error message.

        Args:
            e: The exception that occurred
            worker_info: Worker information
            worker_id: Worker ID

        Returns:
            Error message string

        Raises:
            WorkerFailedError: If worker has died
            EngineCallError: If non-retryable error
        """
        if isinstance(e, httpx.ConnectError):
            return f"Connection error: {e}"
        elif isinstance(e, httpx.TimeoutException):
            raise f"Timeout: {e}"
        elif isinstance(e, EngineCallError):
            raise
        else:
            return f"Unexpected error: {e}"
