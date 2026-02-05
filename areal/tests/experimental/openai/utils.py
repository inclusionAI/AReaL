from openai import AsyncOpenAI


class SimpleAgent:
    async def run(self, data: dict, **extra_kwargs) -> float:
        http_client = extra_kwargs.get("http_client", None)
        base_url = extra_kwargs.get("base_url", None)
        client = AsyncOpenAI(base_url=base_url, http_client=http_client, max_retries=0)
        _ = await client.chat.completions.create(
            messages=data["messages"], model="default"
        )
        return 1.0
