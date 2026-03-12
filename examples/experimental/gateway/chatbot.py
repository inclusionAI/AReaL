"""Standalone streaming chatbot using the GatewayRolloutController.

This example launches a gateway stack (Gateway → Router → Data Proxy →
SGLang) on a single GPU and exposes an OpenAI-compatible ``/chat/completions``
endpoint.  It then runs an interactive REPL that sends user messages to the
gateway and prints the assistant's reply token-by-token as it streams back.

No training is involved — the example demonstrates the gateway inference
path in isolation.

Usage (single GPU)::

    python examples/experimental/gateway/chatbot.py \
        --model Qwen/Qwen3-1.7B \
        --tp 1
"""

from __future__ import annotations

import argparse
import asyncio

from openai import AsyncOpenAI


# ---------------------------------------------------------------------------
# Gateway bootstrap (runs in a background thread)
# ---------------------------------------------------------------------------


def _launch_gateway(
    model_path: str,
    tp_size: int = 1,
    gateway_port: int = 8080,
):
    """Launch SGLang + gateway micro-services and block until ready.

    This mirrors what ``GatewayRolloutController.initialize()`` does, but
    without a scheduler — we start SGLang as a subprocess and spin up the
    micro-services in-process.
    """
    import subprocess
    import time

    from areal.api.cli_args import SGLangConfig
    from areal.experimental.gateway.controller.config import GatewayControllerConfig
    from areal.experimental.gateway.data_proxy.app import (
        create_app as create_data_proxy_app,
    )
    from areal.experimental.gateway.data_proxy.config import DataProxyConfig
    from areal.experimental.gateway.gateway.app import (
        create_app as create_gateway_app,
    )
    from areal.experimental.gateway.gateway.config import GatewayConfig
    from areal.experimental.gateway.router.app import (
        create_app as create_router_app,
    )
    from areal.experimental.gateway.router.config import RouterConfig
    from areal.utils.network import find_free_ports

    cfg = GatewayControllerConfig(
        tokenizer_path=model_path,
        model_path=model_path,
        gateway_port=gateway_port,
    )

    # -- 1. Launch SGLang server as a subprocess ----------------------------
    sglang_port = find_free_ports(1)[0]
    sglang_cfg = SGLangConfig(
        model_path=model_path,
        skip_tokenizer_init=True,
        dtype="bfloat16",
        mem_fraction_static=0.85,
        context_length=32768,
    )
    sglang_args = SGLangConfig.build_args(
        sglang_config=sglang_cfg,
        tp_size=tp_size,
        base_gpu_id=0,
        port=sglang_port,
        host="127.0.0.1",
    )
    sglang_cmd = SGLangConfig.build_cmd_from_args(sglang_args)
    print(f"[chatbot] Starting SGLang server: {' '.join(sglang_cmd)}")
    sglang_proc = subprocess.Popen(sglang_cmd)  # noqa: S603

    # Wait for SGLang to be ready
    import httpx

    sglang_url = f"http://127.0.0.1:{sglang_port}"
    deadline = time.monotonic() + 300.0
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{sglang_url}/health", timeout=2.0)
            if r.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1.0)
    else:
        sglang_proc.kill()
        raise RuntimeError("SGLang server did not become healthy within 300s")

    print(f"[chatbot] SGLang server ready at {sglang_url}")

    # -- 2. Start micro-services in background threads ----------------------
    import threading

    import uvicorn

    def _run_uvicorn(app, host, port, name):
        uvi_cfg = uvicorn.Config(app, host=host, port=port, log_level="warning")
        server = uvicorn.Server(uvi_cfg)
        server.run()

    # Router
    router_cfg = RouterConfig(
        host=cfg.router_host,
        port=cfg.router_port,
        admin_api_key=cfg.admin_api_key,
        log_level="warning",
    )
    router_app = create_router_app(router_cfg)
    threading.Thread(
        target=_run_uvicorn,
        args=(router_app, cfg.router_host, cfg.router_port, "Router"),
        daemon=True,
    ).start()
    _wait_for(f"http://127.0.0.1:{cfg.router_port}/health", "Router")

    # Data Proxy (single worker)
    dp_port = cfg.data_proxy_base_port
    dp_cfg = DataProxyConfig(
        host=cfg.data_proxy_host,
        port=dp_port,
        backend_addr=sglang_url,
        tokenizer_path=model_path,
        log_level="warning",
        admin_api_key=cfg.admin_api_key,
    )
    dp_app = create_data_proxy_app(dp_cfg)
    threading.Thread(
        target=_run_uvicorn,
        args=(dp_app, cfg.data_proxy_host, dp_port, "DataProxy"),
        daemon=True,
    ).start()
    _wait_for(f"http://127.0.0.1:{dp_port}/health", "DataProxy")

    # Register data proxy in router
    import httpx as _httpx

    _httpx.post(
        f"http://127.0.0.1:{cfg.router_port}/admin/register",
        json={"worker_addr": f"http://127.0.0.1:{dp_port}"},
        headers={"Authorization": f"Bearer {cfg.admin_api_key}"},
        timeout=5.0,
    ).raise_for_status()

    # Gateway
    gw_cfg = GatewayConfig(
        host=cfg.gateway_host,
        port=cfg.gateway_port,
        admin_api_key=cfg.admin_api_key,
        router_addr=f"http://127.0.0.1:{cfg.router_port}",
        log_level="warning",
    )
    gw_app = create_gateway_app(gw_cfg)
    threading.Thread(
        target=_run_uvicorn,
        args=(gw_app, cfg.gateway_host, cfg.gateway_port, "Gateway"),
        daemon=True,
    ).start()
    _wait_for(f"http://127.0.0.1:{cfg.gateway_port}/health", "Gateway")

    print(f"[chatbot] Gateway ready at http://127.0.0.1:{cfg.gateway_port}")
    return sglang_proc


def _wait_for(url: str, name: str, timeout: float = 30.0) -> None:
    import time

    import httpx

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.3)
    raise RuntimeError(f"{name} at {url} did not become healthy within {timeout}s")


# ---------------------------------------------------------------------------
# Interactive chat REPL
# ---------------------------------------------------------------------------


async def chat_loop(gateway_url: str, admin_api_key: str) -> None:
    """Run an interactive chat loop using streaming completions."""
    client = AsyncOpenAI(
        base_url=f"{gateway_url}/",
        api_key=admin_api_key,
        max_retries=0,
    )

    messages: list[dict] = []
    print("\n=== Qwen3 Chatbot (streaming) ===")
    print("Type your message and press Enter.  Ctrl-C or 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if user_input.strip().lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        messages.append({"role": "user", "content": user_input})

        print("Assistant: ", end="", flush=True)
        assistant_text = ""
        try:
            stream = await client.chat.completions.create(
                model="sglang",
                messages=messages,  # type: ignore[arg-type]
                stream=True,
                temperature=0.7,
                max_completion_tokens=2048,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    print(delta.content, end="", flush=True)
                    assistant_text += delta.content
        except Exception as e:
            print(f"\n[Error] {e}")
            messages.pop()  # Remove the failed user message
            continue

        print()  # newline after streamed response
        messages.append({"role": "assistant", "content": assistant_text})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Streaming chatbot using the AReaL gateway stack"
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-1.7B",
        help="HuggingFace model path (default: Qwen/Qwen3-1.7B)",
    )
    parser.add_argument(
        "--tp", type=int, default=1, help="Tensor parallel size (default: 1)"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="Gateway port (default: 8080)"
    )
    args = parser.parse_args()

    admin_api_key = "areal-admin-key"
    gateway_url = f"http://127.0.0.1:{args.port}"

    # Launch SGLang + gateway services
    sglang_proc = _launch_gateway(
        model_path=args.model,
        tp_size=args.tp,
        gateway_port=args.port,
    )

    try:
        asyncio.run(chat_loop(gateway_url, admin_api_key))
    finally:
        print("[chatbot] Shutting down SGLang server...")
        sglang_proc.terminate()
        sglang_proc.wait(timeout=10)


if __name__ == "__main__":
    main()
