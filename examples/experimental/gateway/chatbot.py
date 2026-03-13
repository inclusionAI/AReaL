"""Standalone streaming chatbot using the GatewayRolloutController.

Launches a full gateway stack (SGLang + Router + Data Proxy + Gateway)
via ``GatewayRolloutController`` and runs an interactive REPL that
streams chat completions token-by-token through the OpenAI-compatible
``/chat/completions`` endpoint.

No training is involved — the example demonstrates the gateway inference
path in isolation.

Usage::

    python examples/experimental/gateway/chatbot.py \
        --config examples/experimental/gateway/chatbot.yaml
"""

from __future__ import annotations

import asyncio
import sys

from openai import AsyncOpenAI

from areal.api.alloc_mode import AllocationMode
from areal.api.cli_args import (
    BaseExperimentConfig,
    SGLangConfig,
    load_expr_config,
)
from areal.experimental.gateway.controller.config import GatewayControllerConfig
from areal.experimental.gateway.controller.controller import (
    GatewayRolloutController,
)
from areal.infra import LocalScheduler


# ---------------------------------------------------------------------------
# Interactive streaming chat REPL
# ---------------------------------------------------------------------------


async def chat_loop(gateway_url: str, admin_api_key: str) -> None:
    """Run an interactive chat loop using streaming completions."""
    client = AsyncOpenAI(
        base_url=f"{gateway_url}/",
        api_key=admin_api_key,
        max_retries=0,
    )

    messages: list[dict] = []
    print("\n=== Qwen3 Chatbot (streaming via GatewayRolloutController) ===")
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


def main(args):
    config, _ = load_expr_config(args, BaseExperimentConfig)

    # Parse allocation mode to derive TP/DP sizes
    alloc_mode = AllocationMode.from_str(config.allocation_mode)

    # Build SGLang server args
    server_args = SGLangConfig.build_args(
        sglang_config=config.sglang,
        tp_size=alloc_mode.gen.tp_size,
        base_gpu_id=0,
    )

    # Build GatewayControllerConfig from the experiment config
    from areal.api.cli_args import SchedulingSpec

    gw_cfg = GatewayControllerConfig(
        tokenizer_path=config.tokenizer_path,
        model_path=config.tokenizer_path,  # same as tokenizer for chatbot
        consumer_batch_size=1,
        scheduling_spec=(SchedulingSpec(gpu=1, cmd="python3 -m areal.infra.rpc.rpc_server"),),
    )

    # Create scheduler and controller
    scheduler = LocalScheduler(exp_config=config)
    controller = GatewayRolloutController(gw_cfg, scheduler)

    # Initialize — this launches SGLang servers and starts gateway services
    controller.initialize(
        role="rollout",
        alloc_mode=alloc_mode,
        server_args=server_args,
    )

    gateway_url = controller._gateway_addr
    admin_api_key = gw_cfg.admin_api_key
    print(f"[chatbot] Gateway ready at {gateway_url}")

    try:
        asyncio.run(chat_loop(gateway_url, admin_api_key))
    finally:
        print("[chatbot] Shutting down...")
        controller.destroy()


if __name__ == "__main__":
    main(sys.argv[1:])
