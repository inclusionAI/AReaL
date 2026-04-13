import os
import json
import time
import queue
import atexit
import pathlib
import threading
import aiohttp
import asyncio
from typing import Optional, Union, Dict, List, Any
from urllib.parse import urlencode
import regex as re

import langid
from .base import BaseTool, register_tool

class BingSearchEngine():
    """
    Async Bing search engine that provides web search capability with caching.
    
    This tool interfaces with the Brightdata API to perform Bing searches.
    It includes robust caching to minimize redundant API calls and supports
    asynchronous operations with connection pooling.
    """

    def __init__(
        self,
        api_key: str,
        zone: str = "serp_api1",
        max_results: int = 10,
        result_length: int = 1000,
        location: str = "us",
        cache_file: Optional[str] = None,
        cache_refresh_interval: float = 15.0
    ):
        """
        Initialize the Bing search engine.
        
        Args:
            api_key: Brightdata API key
            zone: Brightdata zone name
            max_results: Maximum number of search results to return
            result_length: Maximum length of each result snippet
            location: Country code for search localization
            cache_file: Path to cache file (if None, uses ~/.verl_cache/bing_search_cache.jsonl)
            cache_refresh_interval: Minimum seconds between cache file checks
        """
        # API configuration
        self._api_key = api_key
        self._zone = zone
        self._max_results = max_results
        self._result_length = result_length
        self._location = location
        
        # Cache and synchronization
        self._cache = {}
        self._cache_lock = threading.Lock()
        self._lang_id_lock = threading.Lock()
        self._cache_refresh_interval = cache_refresh_interval
        self._last_cache_check = 0.0
        self._cache_mod_time = 0.0
        
        # Setup cache file paths
        self._setup_cache_paths(cache_file)
        
        # Load existing cache
        self._load_cache()
        
        # HTTP session for connection pooling
        self._session = None
    
    def _setup_cache_paths(self, cache_file: Optional[str]) -> None:
        """
        Set up cache file path.
        
        Args:
            cache_file: Path to cache file or None for default
        """
        if cache_file is None:
            cache_dir = pathlib.Path.home() / ".verl_cache"
            cache_dir.mkdir(exist_ok=True)
            self._cache_file = cache_dir / "bing_search_cache.jsonl"
        else:
            self._cache_file = pathlib.Path(cache_file)
            self._cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    def _load_cache(self) -> None:
        """Load the cache from JSONL file."""
        if not self._cache_file.exists():
            return
            
        try:
            # Record file modification time
            self._cache_mod_time = os.path.getmtime(self._cache_file)
            
            # Load JSONL file line by line
            cache_data = {}
            with open(self._cache_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if 'query' in entry and 'result' in entry:
                            cache_data[entry['query']] = entry['result']
                        else:
                            print(f"Invalid cache entry format at line {line_num}")
                    except json.JSONDecodeError as e:
                        print(f"Invalid JSON at line {line_num}: {e}")
                        continue
            
            # Update in-memory cache
            with self._cache_lock:
                self._cache = cache_data
            
            self._last_cache_check = time.time()
            print(f"Loaded {len(self._cache)} cache entries from {self._cache_file}")
            
        except Exception as e:
            print(f"Failed to load cache file: {str(e)}")
            self._cache = {}

    async def _save_cache_async(self, query: str, result: str) -> None:
        """Save a single cache entry to JSONL file asynchronously."""
        if query is None or result is None:
            return
            
        def _write_cache():
            try:
                # Create cache entry
                cache_entry = {
                    "query": query,
                    "result": result,
                    "timestamp": time.time()
                }
                
                # Append to JSONL file
                with open(self._cache_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(cache_entry, ensure_ascii=False) + "\n")
                
                # Update modification time record
                self._cache_mod_time = os.path.getmtime(self._cache_file)
                    
            except Exception as e:
                print(f"Failed to save cache entry: {str(e)}")
        
        # Run cache write in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _write_cache)

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with connection pooling."""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=100,  # Total connection pool size
                limit_per_host=30,  # Max connections per host
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={'User-Agent': 'AsyncBingSearchEngine/1.0'}
            )
        return self._session

    @property
    def name(self) -> str:
        """Tool name identifier."""
        return "bing_search"

    @property
    def trigger_tag(self) -> str:
        """Tag used to trigger this tool."""
        return "search"

    async def _make_request(self, query: str, timeout: int) -> Dict:
        """
        Send async request to Brightdata API.

        Args:
            query: Search query
            timeout: Request timeout in seconds

        Returns:
            API response data as dict
        """
        # Determine language settings based on query language
        with self._lang_id_lock:
            lang_code, lang_confidence = langid.classify(query)
        if lang_code == 'zh':
            mkt, setLang = "zh-CN", "zh"
        else:
            mkt, setLang = "en-US", "en"
        
        # Prepare URL with query parameters
        encoded_query = urlencode({
            "q": query, 
            "mkt": mkt, 
            "setLang": setLang
        })
        target_url = f"https://www.bing.com/search?{encoded_query}&brd_json=1&cc={self._location}"

        # Prepare headers and payload
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "zone": self._zone,
            "url": target_url,
            "format": "raw"
        }

        # Get session and make async request
        session = await self._get_session()
        
        async with session.post(
            "https://api.brightdata.com/request",
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as response:
            if response.status != 200:
                text = await response.text()
                raise Exception(f"HTTP {response.status}: {text}")
            
            response_text = await response.text()
            return json.loads(response_text)

    async def execute(self, query: str, timeout: int = 60) -> str:
        """
        Execute Bing search query asynchronously.

        Args:
            query: Search query string
            timeout: API request timeout in seconds

        Returns:
            Formatted search results as string
        """
        # Clean query
        query = query.replace('"', '')
        
        # Check cache for existing results
        with self._cache_lock:
            if query in self._cache:
                print(f"Cache hit for query: {query}")
                return self._cache[query]

        try:
            # Make async API request
            data = await self._make_request(query, timeout)

            # Extract search results
            result = self._extract_and_format_results(data)
            
            # Update cache
            with self._cache_lock:
                self._cache[query] = result
            
            # Save cache asynchronously
            await self._save_cache_async(query, result)
                
            return result

        except asyncio.TimeoutError:
            error_msg = f"Bing search request timed out after {timeout} seconds"
            print(error_msg)
            return f"Search failed: {error_msg}"
        except Exception as e:
            error_msg = f"Bing search failed: {str(e)}"
            print(error_msg)
            return f"Search failed: {error_msg}"
    
    def _extract_and_format_results(self, data: Dict) -> str:
        """
        Extract and format search results from API response.
        
        Args:
            data: API response data
            
        Returns:
            Formatted search results as string
        """
        # If no organic results, return empty response
        if 'organic' not in data:
            data['chunk_content'] = []
            return self._format_results(data)

        # Extract unique snippets
        chunk_content_list = []
        seen_snippets = set()
        for result in data['organic']:
            snippet = result.get('description', '').strip()
            if len(snippet) > 0 and snippet not in seen_snippets:
                chunk_content_list.append(snippet)
                seen_snippets.add(snippet)

        data['chunk_content'] = chunk_content_list
        return self._format_results(data)

    def _format_results(self, results: Dict) -> str:
        """
        Format search results into readable text.
        
        Args:
            results: Dictionary containing search results
            
        Returns:
            Formatted string of search results
        """
        if not results.get("chunk_content"):
            return "No search results found."

        formatted = []
        for idx, snippet in enumerate(results["chunk_content"][:self._max_results], 1):
            snippet = snippet[:self._result_length]
            formatted.append(f"Page {idx}: {snippet}")
        
        return "\n".join(formatted)

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


@register_tool
class BingSearchTool(BaseTool):
    """
    Async Bing search tool that follows the BaseTool interface.
    
    This tool wraps the BingSearchEngine to provide search functionality
    while adhering to the standard tool interface.
    """
    
    tool_type = "bing_search"
    
    def __init__(
        self,
        num_workers=1,
        api_key: str = None,
        zone: str = "serp_api1",
        max_results: int = 10,
        result_length: int = 1000,
        location: str = "cn",
        cache_file: Optional[str] = None,
        cache_refresh_interval: float = 15.0,
        timeout: int = 60
    ):
        """
        Initialize the Bing search tool.
        
        Args:
            num_workers: Number of workers (inherited from BaseTool)
            api_key: Brightdata API key
            zone: Brightdata zone name
            max_results: Maximum number of search results to return
            result_length: Maximum length of each result snippet
            location: Country code for search localization
            cache_file: Path to cache file
            cache_refresh_interval: Minimum seconds between cache file checks
            timeout: Default timeout for search requests
        """
        super().__init__(num_workers)
        
        # Get API key from environment if not provided
        if api_key is None:
            api_key = os.getenv('BRIGHTDATA_API_KEY')
            if api_key is None:
                raise ValueError("API key must be provided either as parameter or BRIGHTDATA_API_KEY environment variable")
        
        # Initialize the search engine
        self.search_engine = BingSearchEngine(
            api_key=api_key,
            zone=zone,
            max_results=max_results,
            result_length=result_length,
            location=location,
            cache_file=cache_file,
            cache_refresh_interval=cache_refresh_interval
        )
        
        self.timeout = timeout
    
    def get_usage_inst(self):
        """
        Get the usage instructions for the tool
        """
        return "Use this tool to search the web using Bing. Provide search queries in <search>query</search> tags or ```search\\nquery\\n``` code blocks."
    
    def parse_action(self, action: str):
        """
        Parse the raw action string to extract the search query.
        
        Args:
            action: The raw action string
            
        Returns:
            tuple: (search_query, is_valid)
        """
        # Try to find search query in various formats
        search_queries = re.findall(r"<search>(.*?)</search>", action, re.DOTALL)
        
        if not search_queries:
            search_queries = re.findall(r"```\n?search\n(.*?)\n```", action, re.DOTALL)
        
        if not search_queries:
            # Try to find any search-like patterns
            search_queries = re.findall(r"search:\s*(.*?)(?:\n|$)", action, re.IGNORECASE)
        
        if len(search_queries) == 0:
            return "", False
        
        # Use the first search query found and clean it
        search_query = search_queries[0].strip()
        
        # Basic validation - ensure query is not empty and reasonable length
        if not search_query:
            return "", False
        
        return search_query, True
    
    def get_action_priority(self, action: str, extra_field: dict) -> int:
        """
        Get the priority for handling this action.
        
        Args:
            action: The raw action string
            extra_field: Extra fields associated with the action
            
        Returns:
            priority: Integer priority (-1 means cannot handle, 0 = default, positive = higher priority)
        """
        _, valid = self.parse_action(action)
        if not valid:
            return -1
        
        # Give higher priority if the action explicitly mentions search
        if any(keyword in action.lower() for keyword in ['<search>', 'search:', '```search']):
            return 2
        
        return 0
    
    def postprocess_observation(self, observation: Union[str, Dict[str, Any]]) -> Union[str, Dict[str, Any]]:
        """
            add <result> tags to the observation
        """
        if isinstance(observation, str):
            # Wrap the observation in <result> tags
            return f"<result>{observation}</result>"
        elif isinstance(observation, dict):
            # If it's a dict, wrap the 'observation' field
            observation['obs'] = f"<result>{observation.get('observation', '')}</result>"
            return observation
        else:
            # If it's neither, return as is
            return observation

    async def aget_observations(self, trajectory_ids: List[str], actions: List[str], extra_fields: List[Dict[str, Any]]):
        """
        Async version of get_observations for better performance.
        """
        observations = []
        dones = []
        valids = []
        
        # Process all actions concurrently
        tasks = []
        for i, (trajectory_id, action, extra_field) in enumerate(zip(trajectory_ids, actions, extra_fields)):
            task = self._conduct_action_async(trajectory_id, action, extra_field)
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                observations.append(f"Search error: {str(result)}")
                dones.append(False)
                valids.append(False)
            else:
                obs, done, valid = result
                observations.append(obs)
                dones.append(done)
                valids.append(valid)
        
        return observations, dones, valids

    async def _conduct_action_async(self, trajectory_id: str, action: str, extra_field: Dict[str, Any]):
        """
        Conduct the search action asynchronously and return the observation.
        
        Args:
            trajectory_id: The trajectory ID
            action: The action to conduct (should contain search query)
            extra_field: Extra data to include in the request
            
        Returns:
            tuple: (observation, done, valid)
        """
        parsed_query, is_valid = self.parse_action(action)
        if len(parsed_query) > 500:
            observation = "Search query is too long. Please shorten your query."
            return observation, False, False
        
        env = self.load_env(trajectory_id)
        
        if not is_valid:
            observation = "No valid search query found. Please provide a search query in <search>query</search> tags or ```search\\nquery\\n``` code blocks."
            done = False
            valid = False
        else:
            try:
                # Get timeout from extra_field if provided
                timeout = extra_field.get('timeout', self.timeout) if extra_field else self.timeout
                
                # Execute the async search
                search_results = await self.search_engine.execute(parsed_query, timeout)
                
                if search_results and not search_results.startswith("Search failed:"):
                    observation = f"Search results for '{parsed_query}':\n\n{search_results}"
                    valid = True
                else:
                    observation = f"Search for '{parsed_query}' returned no results or failed."
                    valid = False
                
                # Search action is typically always done after one execution
                done = False
                
            except Exception as e:
                observation = f"Search failed with error: {str(e)}"
                done = False
                valid = False
        
        observation = self.postprocess_observation(observation)

        # Update environment
        self.update_env(trajectory_id, env, parsed_query, is_valid, extra_field, observation)
        self.save_env(trajectory_id, env)
        return observation, done, valid

    def conduct_action(self, trajectory_id, action, extra_field):
        """
        Synchronous wrapper for backward compatibility.
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._conduct_action_async(trajectory_id, action, extra_field))
        finally:
            loop.close()
    
    def __del__(self):
        """Cleanup when tool is destroyed."""
        if hasattr(self, 'search_engine') and hasattr(self.search_engine, '_session'):
            if self.search_engine._session and not self.search_engine._session.closed:
                # Try to close session gracefully
                try:
                    loop = asyncio.get_event_loop()
                    if not loop.is_closed():
                        loop.create_task(self.search_engine.close())
                except:
                    pass  # Best effort cleanup