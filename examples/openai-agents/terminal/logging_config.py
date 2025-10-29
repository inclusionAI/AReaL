import logging
import sys


def setup_logging():
    """Setup logging configuration for the MCP server."""
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,  # Changed from DEBUG to INFO to reduce noise
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Suppress noisy third-party loggers
    logging.getLogger("mcp.server.lowlevel.server").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("litellm").setLevel(logging.WARNING)

    # Ensure terminal_bench loggers also output to stdout
    terminal_logger = logging.getLogger("terminal_bench_server")
    if not terminal_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        terminal_logger.addHandler(handler)
        terminal_logger.setLevel(logging.DEBUG)
        terminal_logger.propagate = False  # Prevent duplicate logging to root logger
