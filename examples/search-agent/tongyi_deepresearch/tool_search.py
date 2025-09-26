import asyncio
import os
from typing import List, Optional, Union

import aiohttp
from qwen_agent.tools.base import BaseTool, register_tool

from areal.utils.http import get_default_connector

SERPER_KEY = os.environ.get("SERPER_KEY_ID", "")


@register_tool("search", allow_overwrite=True)
class Search(BaseTool):
    name = "search"
    description = "Performs batched web searches: supply an array 'query'; the tool retrieves the top 10 results for each query in one call."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Array of query strings. Include multiple complementary search queries in a single call.",
            },
        },
        "required": ["query"],
    }

    def __init__(self, cfg: Optional[dict] = None):
        super().__init__(cfg)

    async def google_search_with_serp(self, session: aiohttp.ClientSession, query: str):
        def contains_chinese_basic(text: str) -> bool:
            return any("\u4e00" <= char <= "\u9fff" for char in text)

        payload = (
            {"q": query, "location": "China", "gl": "cn", "hl": "zh-cn"}
            if contains_chinese_basic(query)
            else {
                "q": query,
                "location": "United States",
                "gl": "us",
                "hl": "en",
            }
        )
        headers = {"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"}

        url = "https://google.serper.dev/search"
        last_exc: Exception | None = None
        for attempt in range(5):
            try:
                async with session.post(
                    url, json=payload, headers=headers, timeout=30
                ) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise RuntimeError(f"HTTP {resp.status}: {text[:200]}")
                    results = await resp.json()
                break
            except Exception as e:
                last_exc = e
                if attempt == 4:
                    return "Google search Timeout, return None, Please try again later."
                await asyncio.sleep(1 + attempt * 0.5)
        else:  # pragma: no cover (safety)
            return f"Search failed: {last_exc}"

        try:
            organic = results.get("organic", [])
            if not organic:
                raise ValueError("no organic results")
            web_snippets: List[str] = []
            for idx, page in enumerate(organic, 1):
                date_published = (
                    ("\nDate published: " + page["date"]) if page.get("date") else ""
                )
                source = ("\nSource: " + page["source"]) if page.get("source") else ""
                snippet = ("\n" + page["snippet"]) if page.get("snippet") else ""
                title = page.get("title", "(no title)")
                link = page.get("link", "#")
                redacted_version = (
                    f"{idx}. [{title}]({link}){date_published}{source}\n{snippet}"
                )
                redacted_version = redacted_version.replace(
                    "Your browser can't play this video.", ""
                )
                web_snippets.append(redacted_version)
            content = (
                f"A Google search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n"
                + "\n\n".join(web_snippets)
            )
            return content
        except Exception:
            return f"No results found for '{query}'. Try with a more general query."

    async def search_with_serp(self, session: aiohttp.ClientSession, query: str):
        return await self.google_search_with_serp(session, query)

    async def call(self, params: Union[str, dict], **kwargs) -> str:  # type: ignore[override]
        try:
            query = params["query"]
        except Exception:
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field"

        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=30,
                sock_connect=10,
                connect=10,
            ),
            read_bufsize=1024 * 1024 * 10,
            connector=get_default_connector(),
        ) as session:
            if isinstance(query, str):
                return await self.search_with_serp(session, query)
            assert isinstance(query, List)
            tasks = [self.search_with_serp(session, q) for q in query]
            results = await asyncio.gather(*tasks)
            return "\n=======\n".join(results)
