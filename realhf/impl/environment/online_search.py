import requests
import random
import time
import json
import asyncio
import html
from pprint import pprint
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor

try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    from contextlib import AsyncExitStack
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("[WARNING] MCP client not available. access method will be limited.")

from pprint import pprint
import aiohttp
import asyncio
from typing import Dict, List, Any
from threading import Lock

from realhf.impl.environment.online_web_browser import WebPageCache

class MCPToolCaller:
    """MCP工具调用器，用于访问URLs"""
    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.session = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_sse_server(self):
        """连接到运行SSE传输的MCP服务器"""
        print("sse server streams context", flush=True)
        self._streams_context = sse_client(url=self.server_url)
        print("sse server streams context aenter", flush=True)
        streams = await self._streams_context.__aenter__()

        print("sse server client session", flush=True)
        self._session_context = ClientSession(*streams)
        
        print("sse server session context aenter", flush=True )
        self.session = await self._session_context.__aenter__()
        print("sse server session context initialize", flush=True)
        await self.session.initialize()
        return self.session

    async def call_tool(self, url: str):
        """调用open_url工具访问指定URL"""
        if not self.session:
            raise RuntimeError(
                "Session not initialized. Call connect_to_sse_server first."
            )

        print(f"[DEBUG] MCPToolCaller: 调用 open_url 工具，URL: '{url}'")

        try:
            response = await self.session.call_tool("open_url", {"url": url})
            
            # 处理响应
            if response.isError:
                error = response.content[0].text if response.content else "未知错误"
                print(f"[ERROR] MCPToolCaller: 工具执行错误: {error}")
            else:
                result = response.content[0].text if response.content else ""
                print(f"[DEBUG] MCPToolCaller: 工具执行成功，内容长度: {len(result)}")

            return response
        except Exception as e:
            print(f"[ERROR] MCPToolCaller: 工具调用异常: {str(e)}")
            raise

    async def list_tools(self):
        """列出可用的工具"""
        if not self.session:
            raise RuntimeError(
                "Session not initialized. Call connect_to_sse_server first."
            )
            
        response = await self.session.list_tools()
        tools = response.tools
        print(f"[DEBUG] MCPToolCaller: 可用工具: {[tool.name for tool in tools]}")
        return tools

    async def cleanup(self):
        """正确清理会话和流"""
        try:
            if hasattr(self, "_session_context") and self._session_context:
                await self._session_context.__aexit__(None, None, None)
            if hasattr(self, "_streams_context") and self._streams_context:
                await self._streams_context.__aexit__(None, None, None)
        except Exception as e:
            print(f"[WARNING] MCPToolCaller: 清理资源时出现警告: {e}")

class AsyncOnlineSearchServer:
    """专门为异步流程设计的OnlineSearchServer"""
    
    def __init__(self, enable_cache: bool = True, cache_size: int = 10000, cache_file: str = "./webpage_cache.json"):
        # Serper API配置
        self.serper_server_addr = "https://google.serper.dev"
        self.serper_api_key = "SERPER_API_KEY"
        self.serper_headers = {
            'X-API-KEY': self.serper_api_key,
            'Content-Type': 'application/json'
        }
        self.max_workers = 10
        
        # 重试配置
        self.max_retries = 3
        self.retry_delay = 1.0
        self.backoff_factor = 2.0
        
        # MCP服务器地址
        self.mcp_server_url = "MCP_SERVER_URL"
        
        # 网页缓存
        self.webpage_cache = WebPageCache(cache_size, cache_file) if enable_cache else None
        print(f"[DEBUG] AsyncOnlineSearchServer: Initialized with Serper API, cache={'enabled' if enable_cache else 'disabled'}, size={cache_size}")
        
        self.serper_query_stats_lock = Lock()
        self.serper_query_stats = dict(total=0, succes=0)

    
    async def query_async(self, req_meta):
        """异步搜索方法，使用aiohttp进行Serper API查询"""
        import aiohttp
        
        queries = req_meta.get("queries", [])
        topk = req_meta.get("topk", 5)
        
        if not queries:
            return []

        async def single_serper_query_async(session, query: str, topk: int) -> dict:
            if len(query) == 0 :
                print(f"[DEBUG] AsyncOnlineSearchServer: query is empty")
                return {
                    "success": False,
                    "error": "query is empty"
                }
            if len(query) > 1000 :
                print(f"[DEBUG] AsyncOnlineSearchServer: query is too long")
                return {
                    "success": False,
                    "error": "query is too long"
                }
            """异步执行单个Serper API查询"""
            payload = {
                "q": query,
                "num": topk
            }
            
            # 简化的重试机制：3次重试
            for attempt in range(4):  # 1次初始 + 3次重试
                try:
                    if attempt > 0:
                        delay = 30.0 * (2 ** (attempt - 1))  # 1s, 2s, 4s
                        print(f"[DEBUG] AsyncOnlineSearchServer: Retry {attempt}/3 for query '{query}' after {delay:.1f}s delay")
                        await asyncio.sleep(delay)
                    
                    print(f"[DEBUG] AsyncOnlineSearchServer: Sending query to Serper (attempt {attempt + 1}): {query}")
                    
                    with self.serper_query_stats_lock:
                        self.serper_query_stats["total"] += 1

                    async with session.post(
                        f"{self.serper_server_addr}/search",
                        headers=self.serper_headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            if attempt > 0:
                                print(f"[INFO] AsyncOnlineSearchServer: Query succeeded on retry {attempt}")
                            with self.serper_query_stats_lock:
                                self.serper_query_stats["succes"] += 1
                            return {
                                "success": True,
                                "data": data
                            }
                        else:
                            # 任何HTTP错误都记录并重试
                            response_text = await response.text()
                            error_msg = f"HTTP {response.status}: {response_text[:100]}"
                            print(f"[WARNING] AsyncOnlineSearchServer: HTTP error (attempt {attempt + 1}): {error_msg}")
                            if attempt == 3:  # 最后一次尝试
                                return {
                                    "success": False,
                                    "error": error_msg
                                }
                        
                except Exception as e:
                    # 任何异常都记录并重试
                    error_msg = f"{type(e).__name__}: {str(e)[:100]}"
                    print(f"[WARNING] AsyncOnlineSearchServer: Error (attempt {attempt + 1}): {error_msg}")
                    time.sleep(3)
                    if attempt == 3:  # 最后一次尝试
                        return {
                            "success": False,
                            "error": error_msg
                        }
            
            return {
                "success": False,
                "error": "Unknown error after all retries"
            }
        
        # 使用aiohttp Session并发执行所有查询
        async with aiohttp.ClientSession() as session:
            tasks = [single_serper_query_async(session, query, topk) for query in queries]
            serper_results = await asyncio.gather(*tasks)
        
        # 解析结果并转换为目标格式
        formatted_results = []
        for query, serper_result in zip(queries, serper_results):
            query_results = []
            
            # 检查查询是否成功
            if serper_result and serper_result.get("success", False):
                data = serper_result.get("data", {})
                organic_results = data.get("organic", [])[:topk]
                
                for result in organic_results:
                    query_results.append({
                        "title": result.get("title", ""),
                        "link": result.get("link", ""),
                        "snippet": result.get("snippet", "")
                    })
                
                print(f"[DEBUG] AsyncOnlineSearchServer: Found {len(query_results)} results for: {query}. serper usage: {self.serper_query_stats}")
            else:
                error = serper_result.get("error", "Unknown error") if serper_result else "No response"
                if error in ["query is empty", "query is too long"]:
                    print(f"[WARNING] AsyncOnlineSearchServer: Search failed for '{query}': {error} (Known Error)")
                else:
                    raise RuntimeError(f"[ERROR] AsyncOnlineSearchServer: Search failed for '{query}': {error}")
            
            formatted_results.append(query_results)
        
        return formatted_results

    async def access_async(self, urls):
        """异步访问URLs获取页面内容，支持缓存"""
        if not urls:
            return []
            
        print(f"[DEBUG] AsyncOnlineSearchServer: Processing {len(urls)} URLs with cache support")
        
        results = []
        urls_to_fetch = []
        
        # 检查缓存
        for url in urls:
            if self.webpage_cache and self.webpage_cache.has(url):
                cached_content = self.webpage_cache.get(url)
                if cached_content:
                    results.append(dict(page=cached_content, type="access"))
                    print(f"[DEBUG] AsyncOnlineSearchServer: Using cached content for: {url}")
                else:
                    urls_to_fetch.append(url)
                    results.append(None)  # 占位符
            else:
                urls_to_fetch.append(url)
                results.append(None)  # 占位符
        
        # 如果有需要获取的URLs，使用MCP客户端
        if urls_to_fetch:
            print(f"[DEBUG] AsyncOnlineSearchServer: Fetching {len(urls_to_fetch)} URLs via MCP client")
            
            if not MCP_AVAILABLE:
                raise RuntimeError("[ERROR] MCP client not available. Cannot fetch URLs.")
                # 为无法获取的URLs返回空结果
                for i, result in enumerate(results):
                    if result is None:
                        results[i] = dict(page="", type="access")
            else:
                try:
                    fetched_results = await self._access_urls_async(urls_to_fetch)
                    
                    # 填充结果并缓存
                    fetch_index = 0
                    for i, result in enumerate(results):
                        if result is None:  # 需要填充的位置
                            if fetch_index < len(fetched_results):
                                fetched_result = fetched_results[fetch_index]
                                results[i] = fetched_result
                                
                                # 缓存新获取的内容
                                if self.webpage_cache and fetched_result.get("page"):
                                    self.webpage_cache.put(urls[i], fetched_result["page"])
                                    print(f"[DEBUG] AsyncOnlineSearchServer: Cached new content for: {urls[i]}")
                                
                                fetch_index += 1
                            else:
                                results[i] = dict(page="", type="access")
                                
                except Exception as e:
                    raise RuntimeError(f"[ERROR] AsyncOnlineSearchServer: MCP access failed: {e}")
                    # MCP失败时，为所有未获取的URLs返回空结果
                    for i, result in enumerate(results):
                        if result is None:
                            results[i] = dict(page="", type="access")
        
        # 打印缓存统计
        if self.webpage_cache:
            stats = self.webpage_cache.get_stats()
            print(f"[DEBUG] AsyncOnlineSearchServer: Cache stats - hits: {stats['hits']}, misses: {stats['misses']}, hit_rate: {stats['hit_rate']:.2f}")
        
        return results

    async def _access_urls_async(self, urls):
        """异步访问URLs的实现"""
        caller = MCPToolCaller(self.mcp_server_url)
        results = []
        
        try:
            print(f"[DEBUG] AsyncOnlineSearchServer: Connecting to MCP server at {self.mcp_server_url}")
            await caller.connect_to_sse_server()
            print(f"[DEBUG] AsyncOnlineSearchServer: Successfully connected to MCP server", flush=True)
            
            # 处理每个URL
            for url in urls:
                result = await self._single_url_access(caller, url)
                results.append(result)
                    
        except Exception as e:
            print(f"[ERROR] AsyncOnlineSearchServer: MCP server connection failed: {e}")
            # 连接失败时，为所有URL返回空结果
            results = [dict(page="", type="access") for _ in urls]
            
        finally:
            try:
                await caller.cleanup()
                print(f"[DEBUG] AsyncOnlineSearchServer: Cleaned up MCP client")
            except:
                pass
            
        return results
    
    async def _single_url_access(self, caller, url):
        """访问单个URL，包含重试机制"""
        print(f"[DEBUG] AsyncOnlineSearchServer: Accessing URL: {url}")

        if url == "":
            print(f"[WARNING] AsyncOnlineSearchServer: url is empty")
            return dict(page="", type="access")
        
        # 简化的重试：3次重试，任何错误都重试
        for attempt in range(4):  # 1次初始 + 3次重试
            try:
                if attempt > 0:
                    delay = 1.0
                    print(f"[DEBUG] AsyncOnlineSearchServer: Retry {attempt}/3 for {url}")
                    await asyncio.sleep(delay)
                
                response = await caller.call_tool(url)
                
                # 检查响应是否有错误
                if response.isError:
                    error = response.content[0].text if response.content else "Unknown error"
                    print(f"[WARNING] AsyncOnlineSearchServer: Error response for {url}: {error[:100]}")
                    if attempt == 3:  # 最后一次尝试
                        return dict(page="", type="access")
                    continue
                
                # 获取响应内容
                raw_content = response.content[0].text if response.content else ""
                if not raw_content.strip():
                    print(f"[WARNING] AsyncOnlineSearchServer: Empty response for {url}")
                    if attempt == 3:  # 最后一次尝试
                        return dict(page="", type="access")
                    continue
                
                # 尝试解析JSON，失败则使用原始内容
                try:
                    json_response = json.loads(raw_content)
                    if json_response.get("status") == "success" and "data" in json_response:
                        content = json_response["data"]
                    else:
                        print(f"[WARNING] AsyncOnlineSearchServer: Invalid JSON status for {url}")
                        if attempt == 3:  # 最后一次尝试
                            return dict(page="", type="access")
                        continue
                except json.JSONDecodeError:
                    # 非JSON响应，直接使用原始内容
                    content = raw_content
                
                # 检查内容是否有效
                if content and content.strip():
                    print(f"[DEBUG] AsyncOnlineSearchServer: ✅ Successfully accessed {url} (length: {len(content)})")
                    content = html.unescape(content)
                    return dict(page=content, type="access")
                else:
                    print(f"[WARNING] AsyncOnlineSearchServer: Empty content for {url}")
                    if attempt == 3:  # 最后一次尝试
                        return dict(page="", type="access")
                    continue
                    
            except Exception as e:
                print(f"[WARNING] AsyncOnlineSearchServer: Error accessing {url}: {type(e).__name__}: {str(e)[:100]}")
                if attempt == 3:  # 最后一次尝试
                    return dict(page="", type="access")
                continue
        
        print(f"[WARNING] AsyncOnlineSearchServer: Failed to access {url} after all retries")
        return dict(page="", type="access")

    def get_cache_stats(self):
        """获取缓存统计信息"""
        if self.webpage_cache:
            return self.webpage_cache.get_stats()
        else:
            return {"cache_disabled": True}
    
    def clear_cache(self):
        """清空缓存"""
        if self.webpage_cache:
            self.webpage_cache.clear()
            print("[DEBUG] AsyncOnlineSearchServer: Webpage cache cleared")
