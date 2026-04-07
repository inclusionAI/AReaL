## Tool Server Usage
We provide a tool server starting command to start any tool server that is supported by verl-tool (see full list in [verl_tool/servers/tools](verl_tool/servers/tools)). To start the tool server, you can use the following command:
```bash
# Start the tool server
host=localhost
port=5500
tool_type=python_code # separate by comma if you want to start multiple tool servers. 
workers_per_tool=4 # number of workers for the tool server, meaning how many threads will be used to handle a single tool request with multiple trajectories
python -m verl_tool.servers.serve --host $host --port $port --tool_type $tool_type --workers_per_tool $workers_per_tool & # run in background
```
After running, you should see the following output. Those marked with 🟢 are active tools, while those marked with ⚪ are inactive tools. `finish` as a tool will always be added to manage the end of each trajectory (e.g. delete env)
```
2025-06-06 02:19:04,764 - __main__ - INFO - Initializing tools: ('python_code',)
2025-06-06 02:19:04,772 - __main__ - INFO - Initialized tool: python_code
2025-06-06 02:19:04,772 - __main__ - INFO - Available Tools:
2025-06-06 02:19:04,773 - __main__ - INFO -   - base: inactive ⚪
2025-06-06 02:19:04,773 - __main__ - INFO -   - text_browser: inactive ⚪
2025-06-06 02:19:04,773 - __main__ - INFO -   - finish: active 🟢
2025-06-06 02:19:04,773 - __main__ - INFO -   - piston: inactive ⚪
2025-06-06 02:19:04,773 - __main__ - INFO -   - ipython_code: inactive ⚪
2025-06-06 02:19:04,773 - __main__ - INFO -   - python_code: active 🟢
2025-06-06 02:19:04,773 - __main__ - INFO -   - bash_terminal: inactive ⚪
2025-06-06 02:19:04,773 - __main__ - INFO -   - sandbox_fusion: inactive ⚪
2025-06-06 02:19:04,773 - __main__ - INFO -   - python_oj: inactive ⚪
2025-06-06 02:19:04,774 - __main__ - INFO - Starting async server on localhost:5500
2025-06-06 02:19:04,774 - __main__ - INFO - Server configured for up to 128 concurrent requests
INFO:     Started server process [493613]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://localhost:5500 (Press CTRL+C to quit)
```
To test the tool server, we provide a list of corresponding test scripts in the `verl_tool/servers/tests` directory. For example, to test the `python_code` tool server, you can run the following command:
```bash
# Test the python_code tool server
python -m verl_tool.servers.tests.test_python_code_tool python --url=http://localhost:$port/get_observation
```

## Available Tools
|Tool          |Type            |
|--------------|----------------|
|[Python Interpreter](https://github.com/TIGER-AI-Lab/verl-tool/blob/main/verl_tool/servers/tools/python_code.py) (recommend)|Code Interpreter|
|[Python OJ](https://github.com/TIGER-AI-Lab/verl-tool/blob/main/verl_tool/servers/tools/python_oj.py)|Python Online Judge with test cases|
|[Piston](https://github.com/TIGER-AI-Lab/verl-tool/blob/main/verl_tool/servers/tools/piston.py) (sandbox)|Code Interpreter|
|[Text Browser](https://github.com/TIGER-AI-Lab/verl-tool/blob/main/verl_tool/servers/tools/text_browser.py)  |Web Browser     |
|[Base Terminal](https://github.com/TIGER-AI-Lab/verl-tool/blob/main/verl_tool/servers/tools/bash_terminal.py) | Base Terminal |
|[Google Search](https://github.com/TIGER-AI-Lab/verl-tool/blob/main/verl_tool/servers/tools/google_search.py) | Web Search (Google Custom Search API) |
|[SERP Search](https://github.com/TIGER-AI-Lab/verl-tool/blob/main/verl_tool/servers/tools/serp_search.py) | Web Search (SerpAPI - Multiple Engines) |
|Image Processing (Coming soon) | Image Processing |

---

## Google Search Tool

Web search functionality using Google Custom Search API with full content retrieval.

### Features
- Google Custom Search API integration
- Full web content fetching with BeautifulSoup
- Snippet-only or full content modes
- Query sanitization and link filtering
- Support for `<search>` tags and ````search` code blocks

### Setup

1. **Get Google API Credentials**:
   - Create project at [Google Cloud Console](https://console.cloud.google.com/)
   - Enable "Custom Search API" in APIs & Services > Library
   - Create API key in APIs & Services > Credentials
   - Create Custom Search Engine at [Google CSE](https://cse.google.com/cse/)

2. **Set Environment Variables**:
   ```bash
   export GOOGLE_API_KEY="your_google_api_key"
   export GOOGLE_CSE_ID="your_custom_search_engine_id"
   ```

### Usage

```bash
# Start server
python -m verl_tool.servers.serve --host localhost --port 5500 --tool_type google_search --workers_per_tool 4 &

# Test tool
python -m verl_tool.servers.tests.test_google_search_tool google_search --url=http://localhost:5500/get_observation
```

### Query Formats
- `<search>machine learning algorithms</search>`
- ````search\nPython tutorial\n```
- Mixed: `What is AI? <search>artificial intelligence basics</search>`

### Limitations
- 100 searches/day (free tier)
- Requires Google Cloud setup
- Single search engine (Google only)

---

## SERP Search Tool

Web search using SerpAPI with support for multiple search engines and structured results.

### Features
- **Multiple Search Engines**: Google, Bing, Yahoo, DuckDuckGo, Yandex, Baidu
- **Rich Result Types**: Answer boxes (📋), organic results (🔍), related questions (❓)
- **No Rate Limiting**: SerpAPI handles rate limiting automatically
- **Structured Data**: Clean JSON results without web scraping
- **Visual Indicators**: Emoji-coded result types

### Setup

1. **Get SerpAPI Key**:
   - Sign up at [SerpAPI](https://serpapi.com/) (100 searches/month free)
   - Get API key from [Dashboard](https://serpapi.com/dashboard)

2. **Set Environment Variable**:
   ```bash
   export SERP_API_KEY="your_serpapi_key"
   ```

### Usage

```bash
# Start server  
python -m verl_tool.servers.serve --host localhost --port 5500 --tool_type serp_search --workers_per_tool 4 &

# Test tool
python -m verl_tool.servers.tests.test_serp_search_tool serp_search --url=http://localhost:5500/get_observation
```

### Configuration Options
- `serp_engine`: "google" (default), "bing", "yahoo", "duckduckgo", etc.
- `topk`: Number of results per type (default: 3)
- `search_url`: SerpAPI endpoint (default: "https://serpapi.com/search")

### Output Example
```
📋 Answer Box 1:
URL: https://example.com
Content: "Direct Answer Title"
Direct answer content...

🔍 Web Result 2:  
URL: https://website.com
Content: "Page Title"
Relevant webpage content...
```

### Advantages over Google Search
- **Simpler Setup**: Only API key needed (no CSE configuration)
- **Multiple Engines**: Choose from 10+ search engines
- **Better Structure**: Answer boxes, related questions included
- **Rate Limiting**: Automatically handled by SerpAPI

### Limitations
- 100 searches/month (free tier)
- Requires internet connection
- Monthly quotas (paid plans start at $50/month)

---

## Search Tool Comparison

| Feature | Google Search Tool | SERP Search Tool |
|---------|-------------------|------------------|
| **Setup** | Complex (API + CSE) | Simple (API key only) |
| **Engines** | Google only | 10+ engines available |
| **Content** | Full web scraping | Structured snippets |
| **Free Tier** | 100/day | 100/month |
| **Result Types** | Basic results | Rich (answer boxes, etc.) |
| **Rate Limits** | Manual handling | Auto-handled |
| **Best For** | Detailed content | Quick structured results |