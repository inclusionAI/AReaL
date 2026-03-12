"""Internal agent for self-distillation mode.

``_SelfDistillAgent`` uses the admin API key to pull buffered completions
from the proxy worker's self-distill queue.  Unlike ``_OnlineAgent``, it
does NOT go through the session lifecycle (start/end session, set reward).
Instead, user code calls ``/chat/completions`` with the admin key, and
those completions are buffered separately.  This agent pulls them via
``/internal/get_self_distill_completion``.
"""

from __future__ import annotations

from typing import Any

import aiohttp

from areal.infra import workflow_context

from .server import INTERNAL_GET_SELF_DISTILL_COMPLETION_PATHNAME


class _SelfDistillAgent:
    """Internal agent that pulls buffered self-distill completions.

    Parameters
    ----------
    proxy_gateway_addr : str
        HTTP address of the proxy gateway server.
    admin_api_key : str
        Admin API key for authenticating with proxy workers.
    timeout : float
        Maximum seconds to wait for a buffered completion.
    """

    def __init__(
        self,
        proxy_gateway_addr: str,
        admin_api_key: str,
        timeout: float = 300.0,
    ):
        self.proxy_gateway_addr = proxy_gateway_addr
        self.admin_api_key = admin_api_key
        self.timeout = timeout

    async def run(
        self,
        data: dict[str, Any],
        **extra_kwargs: Any,
    ) -> dict[str, Any]:
        """Pull one buffered completion from the worker's distillation queue.

        Parameters
        ----------
        data : dict
            Ignored (dataloader yields empty dicts in online mode).
        extra_kwargs : dict
            Provided by ``OpenAIProxyWorkflow``:

            - ``base_url``: proxy worker address
            - ``api_key``: admin API key
            - ``http_client``: ``httpx.AsyncClient`` (unused here)

        Returns
        -------
        dict
            Contains ``tensor_dict`` (with ``input_ids``, ``loss_mask``,
            ``logprobs``, ``attention_mask``) and ``interaction_id``.
        """
        base_url = extra_kwargs["base_url"]  # proxy worker addr

        url = f"{base_url}/{INTERNAL_GET_SELF_DISTILL_COMPLETION_PATHNAME}"
        headers = {"Authorization": f"Bearer {self.admin_api_key}"}
        payload = {"timeout": self.timeout}

        timeout = aiohttp.ClientTimeout(total=self.timeout + 30.0)
        session = await workflow_context.get_aiohttp_session()
        async with session.post(
            url,
            json=payload,
            headers=headers,
            timeout=timeout,
        ) as resp:
            resp.raise_for_status()
            result = await resp.json()

        if result.get("status") == "timeout":
            raise TimeoutError("No self-distill completion available within timeout")

        return result
