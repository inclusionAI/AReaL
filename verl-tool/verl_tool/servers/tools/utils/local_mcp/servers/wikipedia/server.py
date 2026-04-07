"""
An MCP server for Wikipedia search
"""
# pylint: disable=broad-exception-caught
import json
import click
import wikipediaapi
from mcp.server.fastmcp import FastMCP
from mcpuniverse.common.logger import get_logger


def build_server(port: int) -> FastMCP:
    """
    Initializes the MCP server.

    :param port: Port for SSE.
    :return: The MCP server.
    """
    mcp = FastMCP("wikipedia", port=port)

    @mcp.tool()
    async def search(query: str) -> str:
        """
        Fetch Wikipedia information for a given search query.

        Args:
            query: The search query string.
        """
        wiki = wikipediaapi.Wikipedia(
            user_agent="MCPUniverse Agents (mcpuniverse@salesforce.com)",
            language="en"
        )
        try:
            page = wiki.page(query)
            if page.exists():
                result = {
                    "query": query,
                    "title": page.title,
                    "summary": page.summary
                }
                return json.dumps(result, ensure_ascii=False, indent=2)
            return f"No results found for query: {query}"
        except Exception as e:
            return f"An error occurred while processing query: {e}"

    return mcp


@click.command()
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    help="Transport type",
)
@click.option("--port", default="8000", help="Port to listen on for SSE")
def main(transport: str, port: str):
    """
    Starts the initialized MCP server.

    :param port: Port for SSE.
    :param transport: The transport type, e.g., `stdio` or `sse`.
    """
    assert transport.lower() in ["stdio", "sse"], \
        "Transport should be `stdio` or `sse`"
    logger = get_logger("Service:wikipedia")
    logger.info("Starting the MCP server")
    mcp = build_server(int(port))
    mcp.run(transport=transport.lower())
