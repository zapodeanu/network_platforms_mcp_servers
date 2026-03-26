#!/usr/bin/env python3
"""
Legacy Catalyst Center stdio shim (deprecated).

Do not add tool logic here; keep logic in catalyst_center_core.py.
"""

import asyncio
import logging

from catalyst_center_core import EnhancedDeclarativeCatalystServer

shim_logger = logging.getLogger("catalyst_center_stdio_legacy_shim")


if __name__ == "__main__":
    try:
        shim_logger.info(
            "Starting legacy shim: enhanced_declarative_catalyst.py -> "
            "catalyst_center_stdio.py (catalyst_center_mcp stdio wrapper)"
        )
        shim_logger.warning(
            "DEPRECATION: 'enhanced_declarative_catalyst.py' is a legacy stdio entrypoint. "
            "Use 'catalyst_center_stdio.py' going forward."
        )
        asyncio.run(EnhancedDeclarativeCatalystServer().run())
    except Exception as e:
        shim_logger.error("Server crashed: %s", str(e))
        raise
