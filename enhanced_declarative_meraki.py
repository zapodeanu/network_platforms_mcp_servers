#!/usr/bin/env python3
"""
Legacy Meraki stdio shim (deprecated).

Do not add tool logic here; keep logic in meraki_core.py.
"""

import asyncio
import logging

from meraki_stdio import _run_stdio

shim_logger = logging.getLogger("meraki_stdio_legacy_shim")


if __name__ == "__main__":
    try:
        shim_logger.info(
            "Starting legacy shim: enhanced_declarative_meraki.py -> "
            "meraki_stdio.py (meraki_mcp stdio wrapper)"
        )
        shim_logger.warning(
            "DEPRECATION: 'enhanced_declarative_meraki.py' is a legacy stdio entrypoint. "
            "Use 'meraki_stdio.py' going forward."
        )
        asyncio.run(_run_stdio())
    except Exception as e:
        shim_logger.error("Server crashed: %s", str(e))
        raise
