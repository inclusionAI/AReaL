import asyncio
import contextlib
from functools import wraps
from typing import Any, Callable

import litellm
import openai

DEBUG = True


def is_async_callable(obj: Any) -> bool:
    while hasattr(obj, "__wrapped__"):
        obj = obj.__wrapped__
    return asyncio.iscoroutinefunction(obj)


def patched_create(original_create_fn: Callable, session_id: str):
    if hasattr(original_create_fn, "__areal_patched__"):
        return original_create_fn

    # print("???DEBUG", is_async_callable(original_create_fn))
    if is_async_callable(original_create_fn):

        @wraps(original_create_fn)
        async def acreate_fn(*args, **kwargs):
            # Get the current extra_body or create a new dict
            extra_body = kwargs.get("extra_body", {})
            if extra_body is None:
                extra_body = {}

            # Ensure extra_body is a dict and add session_id
            if not isinstance(extra_body, dict):
                extra_body = {}

            # Add session_id to extra_body
            extra_body["session_id"] = session_id
            kwargs["extra_body"] = extra_body

            if DEBUG:
                print(f"Debug kwargs: {kwargs}")
                return "DEBUG"

            # Call the original method with modified kwargs
            return await original_create_fn(*args, **kwargs)

        acreate_fn.__wrapped__ = original_create_fn
        acreate_fn.__areal_patched__ = True
        return acreate_fn
    else:

        @wraps(original_create_fn)
        def create_fn(*args, **kwargs):
            # Get the current extra_body or create a new dict
            extra_body = kwargs.get("extra_body", {})
            if extra_body is None:
                extra_body = {}

            # Ensure extra_body is a dict and add session_id
            if not isinstance(extra_body, dict):
                extra_body = {}

            # Add session_id to extra_body
            extra_body["session_id"] = session_id
            kwargs["extra_body"] = extra_body

            if DEBUG:
                print(f"Debug kwargs: {kwargs}")
                return "DEBUG"

            # Call the original method with modified kwargs
            return original_create_fn(*args, **kwargs)

        create_fn.__wrapped__ = original_create_fn
        create_fn.__areal_patched__ = True
        return create_fn


# monkey patching
@contextlib.contextmanager
def patch_clients(
    session_id: str,
    clients: (
        list[openai.OpenAI | openai.AsyncOpenAI]
        | openai.OpenAI
        | openai.AsyncOpenAI
        | None
    ) = None,
    patch_openai_completion: bool = True,
    patch_litellm_completion: bool = True,
):
    """
    More comprehensive context manager that patches OpenAI client methods
    to automatically include session_id in extra_body.

    This version patches multiple OpenAI client methods and handles
    both direct client calls and module-level calls.
    """
    # Store original methods
    patch_openai_client_completion = (
        hasattr(openai, "_client") and openai._client is not None
    )
    if patch_openai_completion:
        original_openai_create_fn = openai.chat.completions.create
        if patch_openai_client_completion:
            original_openai_client_create_fn = openai._client.chat.completions.create

    if patch_litellm_completion:
        original_litellm_completion_fn = litellm.completion
        original_litellm_acompletion_fn = litellm.acompletion

    if clients is None:
        clients = []
    elif not isinstance(clients, list):
        clients = [clients]
    original_clients_create_fn = []
    for client in clients:
        original_clients_create_fn.append(client.chat.completions.create)

    try:
        if patch_openai_completion:
            # Patch the module-level method
            openai.chat.completions.create = patched_create(
                openai.chat.completions.create, session_id=session_id
            )

            # Patch the client method if it exists
            if patch_openai_client_completion:
                openai._client.chat.completions.create = patched_create(
                    openai._client.chat.completions.create, session_id=session_id
                )

        if patch_litellm_completion:
            litellm.completion = patched_create(
                litellm.completion, session_id=session_id
            )
            litellm.acompletion = patched_create(
                litellm.acompletion, session_id=session_id
            )

        for client in clients:
            client.chat.completions.create = patched_create(
                client.chat.completions.create, session_id=session_id
            )

        yield clients

    finally:
        # Restore original methods
        if patch_openai_completion:
            openai.chat.completions.create = original_openai_create_fn
            if patch_openai_client_completion:
                openai._client.chat.completions.create = (
                    original_openai_client_create_fn
                )

        if patch_litellm_completion:
            litellm.completion = original_litellm_completion_fn
            litellm.acompletion = original_litellm_acompletion_fn

        for client, original_fn in zip(clients, original_clients_create_fn):
            client.chat.completions.create = original_fn


client = openai.AsyncOpenAI(api_key="none")


async def main():
    # session_id = start_session()
    session_id = "my_session_123"
    client = openai.AsyncOpenAI(api_key="none")
    with patch_clients(session_id, clients=[client]):
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
        )
        print(response)
        # outputs:
        # >>> Debug kwargs: {'model': 'gpt-3.5-turbo', 'messages': [{'role': 'user', 'content': 'Hello'}], 'extra_body': {'session_id': 'my_session_123'}}
        # >>> DEBUG

    # end_session(session_id)


if __name__ == "__main__":
    asyncio.run(main())
