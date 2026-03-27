#!/usr/bin/env python3
"""
Catalyst Center stdio entrypoint.
Do not add tool logic here; keep logic in catalyst_center_core.py.
"""

import asyncio
import logging

from mcp.server import NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server

from catalyst_center_core import EnhancedDeclarativeCatalystServer

WRAPPER_LOGGER = logging.getLogger("catalyst_center_stdio_wrapper")


async def _run_stdio() -> None:
    server_impl = EnhancedDeclarativeCatalystServer()
    async with stdio_server() as (read_stream, write_stream):
        await server_impl.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="catalyst-center-mcp",
                server_version="2.1.0",
                capabilities=server_impl.server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    try:
        WRAPPER_LOGGER.info("Starting wrapper: catalyst_center_stdio.py -> catalyst_center_mcp (stdio)")
        asyncio.run(_run_stdio())
    except Exception as e:
        WRAPPER_LOGGER.error("Server crashed: %s", str(e))
        raise
