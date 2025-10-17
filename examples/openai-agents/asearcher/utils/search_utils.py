import asyncio
import json
import os
import random
from typing import Any, Dict, List

import aiohttp

try:
    from tools.web_browser import WebPageCache

    WEBPAGECACHE_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] import web browser error: {e}")
    WEBPAGECACHE_AVAILABLE = False
    WebPageCache = None


SERPER_STATS = dict(num_requests=0)


class AsyncSearchBrowserClient:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session = None
        self.server_list = self.get_server_list()
        self.server_addr = random.choice(self.server_list)
        # print(self.server_list)

    def get_server_list(self):
        import glob

        rag_server_addr_dir = os.environ.get("RAG_SERVER_ADDR_DIR", "")

        server_list = []

        filenames = glob.glob(rag_server_addr_dir + "/Host*_IP*.txt")
        for filename in filenames:
            try:
                server_list.extend(open(filename).readlines())
            except:
                continue
        return server_list

    async def query_async(self, req_meta: Dict[str, Any]) -> List[Dict]:
        cnt = 0
        last_exception = None
        while cnt < 10:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://{self.server_addr}/retrieve",
                        json=req_meta,
                        timeout=aiohttp.ClientTimeout(total=120, sock_connect=120),
                    ) as response:
                        response.raise_for_status()
                        res = await response.json()
                        return [
                            dict(
                                documents=[r["contents"] for r in result],
                                urls=[r["url"] for r in result],
                                server_type="async-search-browser",
                            )
                            for result in res["result"]
                        ]
            except Exception as e:
                print(
                    "query_async", self.server_list, e.__class__.__name__, e.__cause__
                )
                last_exception = e
                self.server_list = self.get_server_list()
                self.server_addr = random.choice(self.server_list)
                print(f"Search Engine switched to {self.server_addr}")
                cnt += 1
                await asyncio.sleep(10)

        raise RuntimeError(
            "Fail to post search query to RAG server"
        ) from last_exception

    async def access_async(self, urls: List[str]) -> List[Dict]:
        cnt = 0
        last_exception = None
        while cnt < 10:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://{self.server_addr}/access",
                        json={"urls": urls},
                        timeout=aiohttp.ClientTimeout(total=120, sock_connect=120),
                    ) as response:
                        response.raise_for_status()
                        res = await response.json()
                        return [
                            dict(
                                page=result["contents"] if result is not None else "",
                                type="access",
                                server_type="async-search-browser",
                            )
                            for result in res["result"]
                        ]
            except Exception as e:
                print("access_async", self.server_list, e)
                last_exception = e
                self.server_list = self.get_server_list()
                self.server_addr = random.choice(self.server_list)
                print(f"Search Engine switched to {self.server_addr}")
                cnt += 1
                await asyncio.sleep(10)

        raise RuntimeError(
            "Fail to post access request to RAG server"
        ) from last_exception


class AsyncOnlineSearchClient:

    _search_semaphore = None
    _access_semaphore = None

    @classmethod
    def _get_search_semaphore(cls):

        if cls._search_semaphore is None:
            cls._search_semaphore = asyncio.Semaphore(20)
        return cls._search_semaphore

    @classmethod
    def _get_access_semaphore(cls):

        if cls._access_semaphore is None:
            cls._access_semaphore = asyncio.Semaphore(10)
        return cls._access_semaphore

    def __init__(
        self,
        enable_cache: bool = True,
        cache_size: int = 10000,
        cache_file: str = "../webpage_cache.json",
        use_jina: bool = False,
        jina_api_key: str = None,
        wrapper_format: bool = True,
    ):
        # Serper API
        self.serper_server_addr = "https://google.serper.dev"
        self.serper_api_key = os.environ.get("SERPER_API_KEY", "")
        if not self.serper_api_key:
            raise RuntimeError(
                "Serper API key is not set. Please configure it in config.yaml or set the SERPER_API_KEY environment variable."
            )
        self.serper_headers = {
            "X-API-KEY": self.serper_api_key,
            "Content-Type": "application/json",
        }
        self.max_workers = 10

        self.max_retries = 3
        self.retry_delay = 1.0
        self.backoff_factor = 2.0

        self.wrapper_format = wrapper_format

        self.use_jina = use_jina

        self.jina_api_key = jina_api_key or os.environ.get("JINA_API_KEY", "")
        if self.use_jina and not self.jina_api_key:
            raise RuntimeError(
                "Jina is enabled but the API key is not set. Please configure it in config.yaml or set the JINA_API_KEY environment variable."
            )

        if enable_cache and WEBPAGECACHE_AVAILABLE:
            self.webpage_cache = WebPageCache(cache_size, cache_file, save_interval=5)
        else:
            self.webpage_cache = None

    async def _jina_readpage_async(self, session, url: str) -> str:
        """
        Read webpage content using Jina service asynchronously.

        Args:
            session: aiohttp ClientSession
            url: The URL to read

        Returns:
            str: The webpage content or error message
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.jina_api_key}",
                "Content-Type": "application/json",
            }

            async with session.get(
                f"https://r.jina.ai/{url}",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status == 200:
                    content = await response.text()
                    return content
                else:
                    return (
                        f"[visit] Failed to read page. Status code: {response.status}"
                    )

        except Exception as e:
            return f"[visit] Failed to read page. Error: {str(e)}"

    async def query_async(self, req_meta):

        import aiohttp

        queries = req_meta.get("queries", [])
        topk = req_meta.get("topk", 5)

        if not queries:
            return []

        async def single_serper_query_async(session, query: str, topk: int) -> dict:

            query = query[:2000]
            async with self._get_search_semaphore():
                payload = {"q": query, "num": topk}

                for attempt in range(4):
                    try:
                        if attempt > 0:
                            delay = 1.0 * (2 ** (attempt - 1))  # 1s, 2s, 4s
                            await asyncio.sleep(delay)

                        await asyncio.sleep(0.1)

                        SERPER_STATS["num_requests"] += 1
                        print("Serper Stats: ", json.dumps(SERPER_STATS))

                        async with session.post(
                            f"{self.serper_server_addr}/search",
                            headers=self.serper_headers,
                            json=payload,
                            timeout=aiohttp.ClientTimeout(total=30),
                        ) as response:
                            if response.status == 200:
                                data = await response.json()
                                if attempt > 0:
                                    print(
                                        f"[INFO] AsyncOnlineSearchClient: Query succeeded on retry {attempt}"
                                    )
                                return {"success": True, "data": data}
                            else:

                                response_text = await response.text()
                                error_msg = (
                                    f"HTTP {response.status}: {response_text[:100]}"
                                )
                                print(
                                    f"[WARNING] AsyncOnlineSearchClient: HTTP error (attempt {attempt + 1}): {error_msg}"
                                )
                                if attempt == 3:
                                    return {"success": False, "error": error_msg}

                    except Exception as e:
                        error_msg = f"{type(e).__name__}: {str(e)[:100]}"
                        print(
                            f"[WARNING] AsyncOnlineSearchClient: Error (attempt {attempt + 1}): {error_msg}"
                        )
                        if attempt == 3:
                            return {"success": False, "error": error_msg}

                return {"success": False, "error": "Unknown error after all retries"}

        async with aiohttp.ClientSession() as session:
            tasks = [
                single_serper_query_async(session, query, topk) for query in queries
            ]
            serper_results = await asyncio.gather(*tasks)

        formatted_results = []
        for query, serper_result in zip(queries, serper_results):
            query_results = []

            if serper_result and serper_result.get("success", False):
                data = serper_result.get("data", {})
                organic_results = data.get("organic", [])[:topk]

                for result in organic_results:
                    query_results.append(
                        {
                            "title": result.get("title", ""),
                            "url": result.get("link", ""),
                            "snippet": result.get("snippet", ""),
                            "server_type": "async-online-search",
                        }
                    )
            else:
                error = (
                    serper_result.get("error", "Unknown error")
                    if serper_result
                    else "No response"
                )
                print(
                    f"[ERROR] AsyncOnlineSearchClient: Search failed for '{query}': {error}"
                )

            formatted_results.append(query_results)

        if self.wrapper_format:
            first_query_results = formatted_results[0] if formatted_results else []
            return [
                {
                    "documents": [
                        result.get("title", "") + " " + result.get("snippet", "")
                        for result in first_query_results
                    ],
                    "urls": [result.get("url", "") for result in first_query_results],
                    "server_type": "async-online-search",
                }
            ]
        else:
            if len(queries) == 1:
                return formatted_results[
                    0
                ]  # return [{...}, {...}] rather than [[{...}, {...}]]
            else:
                return formatted_results  # return [[...], [...]]

    async def access_async(self, urls):

        if not urls:
            return []

        results = []
        urls_to_fetch = []

        for url in urls:
            if self.webpage_cache and self.webpage_cache.has(url):
                cached_content = self.webpage_cache.get(url)
                if cached_content:
                    results.append(dict(page=cached_content, type="access"))
                else:
                    urls_to_fetch.append(url)
                    results.append(None)
            else:
                urls_to_fetch.append(url)
                results.append(None)

        if urls_to_fetch:
            if self.use_jina and self.jina_api_key:
                try:
                    async with self._get_access_semaphore():
                        fetched_results = await self._access_urls_jina_async(
                            urls_to_fetch
                        )

                    fetch_index = 0
                    for i, result in enumerate(results):
                        if result is None:
                            if fetch_index < len(fetched_results):
                                fetched_result = fetched_results[fetch_index]
                                results[i] = fetched_result

                                if self.webpage_cache and fetched_result.get("page"):
                                    self.webpage_cache.put(
                                        urls[i], fetched_result["page"]
                                    )

                                fetch_index += 1
                            else:
                                results[i] = dict(page="", type="access")

                except Exception:
                    for i, result in enumerate(results):
                        if result is None:
                            results[i] = dict(page="", type="access")
            else:
                for i, result in enumerate(results):
                    if result is None:
                        results[i] = dict(page="", type="access")

        for result in results:
            if result is not None:
                result["server_type"] = "async-online-search"
        return results

    async def _access_urls_jina_async(self, urls):
        results = []

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                for url in urls:
                    content = await self._jina_readpage_async(session, url)
                    if content and not content.startswith("[visit] Failed"):
                        results.append(dict(page=content, type="access"))
                    else:
                        results.append(dict(page="", type="access"))

        except Exception:
            results = [dict(page="", type="access") for _ in urls]

        for r in results:
            if len(r["page"]) > 0:
                r["type"] = "jina"

        return results

    def get_cache_stats(self):
        if self.webpage_cache:
            return self.webpage_cache.get_stats()
        else:
            return {"cache_disabled": True}

    def clear_cache(self):
        if self.webpage_cache:
            self.webpage_cache.clear()

    def force_save_cache(self):
        if self.webpage_cache:
            self.webpage_cache.force_save()


SEARCH_CLIENTS = {
    "async-search-access": AsyncSearchBrowserClient,
    "async-online-search-access": AsyncOnlineSearchClient,
}


def make_search_client(
    search_client_type: str, use_jina: bool = False, jina_api_key: str = None
):
    if search_client_type == "async-online-search":
        return SEARCH_CLIENTS[search_client_type](
            use_jina=use_jina, jina_api_key=jina_api_key
        )
    elif search_client_type == "async-online-search-access":
        return SEARCH_CLIENTS[search_client_type](
            use_jina=use_jina, jina_api_key=jina_api_key, wrapper_format=True
        )
    else:
        return SEARCH_CLIENTS[search_client_type]()


if __name__ == "__main__":
    search_client = AsyncOnlineSearchClient()
    url_response = asyncio.run(
        search_client.access_async(
            ["https://en.wikipedia.org/w/index.php?title=Beveridge,%20Victoria"]
        )
    )
    print(url_response[0]["page"][:1000])

    exit(0)
