#!/usr/bin/env python3
"""
Catalyst Center stdio entrypoint.
Do not add tool logic here; keep logic in catalyst_center_core.py.
"""

import asyncio
import logging

from catalyst_center_core import EnhancedDeclarativeCatalystServer

WRAPPER_LOGGER = logging.getLogger("catalyst_center_stdio_wrapper")

if __name__ == "__main__":
    try:
        WRAPPER_LOGGER.info("Starting wrapper: catalyst_center_stdio.py -> catalyst_center_mcp (stdio)")
        asyncio.run(EnhancedDeclarativeCatalystServer().run())
    except Exception as e:
        WRAPPER_LOGGER.error("Server crashed: %s", str(e))
        raise
