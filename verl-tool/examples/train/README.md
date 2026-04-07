## Supported Training Recipes:

| Recipe | Description | Related Tool | Results |
|--------|-------------|--------------|---------|
| [Math Tool-Integrated RL](./math_tir/README.md) | Training a model that can call the code interpreter to solve mathematical problems | `python_code.py` | [link](../../assets/docs/math_results.md) |
| [AceCoder](./acecoder/README.md) | Training a model that can call the code interpreter to solve coding problems | `ipython_code.py` / - | TBD |
| [Search R1](./search_r1/README.md) | Training a model that can call the search tool to retrieve relevant information | `search_retrieval.py` | TBD |
| [DeepSearch](./deepsearch/README.md) | Training a model that can call the web search tool (google search via serper) to retrieve relevant information | `google_search.py` | TBD |
| [SQL](./skysql/README.md) | Training a model that can interact with a SQL database to retrieve relevant information | `sql.py` | TBD |
| [Pixel-Reasoner](./pixel_reasoner/README.md) | Training a model that can solve visual reasoning tasks with image/video processing tools (e.g., resizing, selecting, and cropping images) | `pixel_reasoner.py` | TBD |
| [SWE](./swe/README.md) | Training a model that can solve software engineering tasks with tool-calling capabilities | [`sandbox_r2e.py`](https://github.com/cogito233/verl-tool/blob/r2e_zhiheng/verl_tool/servers/tools/sandbox_r2e.py) | TBD |
| [MCP—Tutorial](./mcp_tut/mcp_agent_tutorial.md) | A tutorial for MCP agent training/evaluation on verltool | N/A| [instruction](./mcp_tut/mcp_agent_tutorial.md) |
| [MCP—Universe](./mcp_universe/README.md) | Evaluate the agent on MCP-Universe bench with verltool | [`mcp_interface.py`]| [instruction](./mcp_universe/README.md) |
