#!/usr/bin/env python3
"""
Meraki stdio entrypoint.
Do not add tool logic here; keep logic in meraki_core.py.
"""

import asyncio
import logging

from mcp.server import NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server

from meraki_core import EnhancedMultiOrgMerakiServer

WRAPPER_LOGGER = logging.getLogger("meraki_stdio_wrapper")


async def _run_stdio() -> None:
    server_impl = EnhancedMultiOrgMerakiServer()
    async with stdio_server() as (read_stream, write_stream):
        await server_impl.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="meraki-mcp",
                server_version="3.0.0",
                capabilities=server_impl.server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    try:
        WRAPPER_LOGGER.info("Starting wrapper: meraki_stdio.py -> meraki_mcp (stdio)")
        asyncio.run(_run_stdio())
    except Exception as e:
        WRAPPER_LOGGER.error("Server crashed: %s", str(e))
        raise
