"""
An MCP server for fetching date information
"""
# pylint: disable=broad-exception-caught
from datetime import datetime, timezone
import pytz
import click
from mcp.server.fastmcp import FastMCP
from mcpuniverse.common.logger import get_logger


def build_server(port: int) -> FastMCP:
    """
    Initializes the MCP server.

    :param port: Port for SSE.
    :return: The MCP server.
    """
    mcp = FastMCP("date", port=port)

    @mcp.tool()
    async def get_today_date() -> str:
        """
        Get today's date in YYYY-MM-DD format.
        
        Returns:
            Today's date as a string in YYYY-MM-DD format
        """
        today = datetime.now().date()
        return today.strftime("%Y-%m-%d")

    @mcp.tool()
    async def get_current_datetime() -> str:
        """
        Get the current date and time in ISO format.
        
        Returns:
            Current datetime as a string in ISO format
        """
        now = datetime.now()
        return now.isoformat()

    @mcp.tool()
    async def get_current_datetime_utc() -> str:
        """
        Get the current date and time in UTC timezone.
        
        Returns:
            Current UTC datetime as a string in ISO format
        """
        now_utc = datetime.now(timezone.utc)
        return now_utc.isoformat()

    @mcp.tool()
    async def get_date_in_timezone(timezone_name: str) -> str:
        """
        Get the current date and time in a specific timezone.
        
        Args:
            timezone_name: Timezone name (e.g., 'America/New_York', 'Europe/London', 'Asia/Tokyo')
            
        Returns:
            Current datetime in the specified timezone as a string
        """
        try:
            tz = pytz.timezone(timezone_name)
            now_in_tz = datetime.now(tz)
            return f"{now_in_tz.strftime('%Y-%m-%d %H:%M:%S %Z')} ({timezone_name})"
        except pytz.exceptions.UnknownTimeZoneError:
            return (f"Error: Unknown timezone '{timezone_name}'. "
                    f"Please use a valid timezone name like 'America/New_York' or 'Europe/London'.")

    @mcp.tool()
    async def get_timestamp() -> str:
        """
        Get the current Unix timestamp.
        
        Returns:
            Current Unix timestamp as a string
        """
        timestamp = datetime.now().timestamp()
        return str(int(timestamp))

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
    :return:
    """
    assert transport.lower() in ["stdio", "sse"], \
        "Transport should be `stdio` or `sse`"
    logger = get_logger("Service:date")
    logger.info("Starting the MCP date server")
    mcp = build_server(int(port))
    mcp.run(transport=transport.lower())
