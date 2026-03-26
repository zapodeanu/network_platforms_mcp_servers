#!/usr/bin/env python3
"""
Meraki stdio entrypoint.
Do not add tool logic here; keep logic in meraki_core.py.
"""

import asyncio
import logging

from meraki_core import EnhancedMultiOrgMerakiServer

WRAPPER_LOGGER = logging.getLogger("meraki_stdio_wrapper")

if __name__ == "__main__":
    try:
        WRAPPER_LOGGER.info("Starting wrapper: meraki_stdio.py -> meraki_mcp (stdio)")
        asyncio.run(EnhancedMultiOrgMerakiServer().run())
    except Exception as e:
        WRAPPER_LOGGER.error("Server crashed: %s", str(e))
        raise
