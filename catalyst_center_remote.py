#!/usr/bin/env python3
"""
Catalyst Center streamable HTTP entrypoint.
Do not add tool logic here; keep logic in catalyst_center_core.py.
"""

import argparse
import asyncio
import contextlib
import json
import logging
import os
from collections.abc import AsyncIterator

import uvicorn
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Mount

from catalyst_center_core import EnhancedDeclarativeCatalystServer, PATH

os.makedirs(os.path.join(PATH, "logs"), exist_ok=True)
WRAPPER_LOGGER = logging.getLogger("catalyst_center_remote_wrapper")


class CatalystCenterRemoteServer(EnhancedDeclarativeCatalystServer):
    """Remote streamable HTTP transport wrapper."""

    async def _handle_execute_explored_endpoint(self, arguments: dict):
        method = arguments.get("method", "GET").upper()
        if method != "GET":
            error_response = {
                "error": "Security restriction: Only GET methods allowed",
                "provided_method": method,
                "allowed_methods": ["GET"],
                "reason": "Write operations (POST/PUT/DELETE) are disabled for safety",
                "suggestion": "Use existing YAML tools for configuration changes",
            }
            return {"content": [{"type": "text", "text": json.dumps(error_response, indent=2)}]}
        return await super()._handle_execute_explored_endpoint(arguments)

    async def run(self, host: str = "0.0.0.0", port: int = 8000):
        session_manager = StreamableHTTPSessionManager(
            app=self.server,
            json_response=False,
            stateless=False,
        )

        async def handle_streamable_http(scope, receive, send):
            await session_manager.handle_request(scope, receive, send)

        @contextlib.asynccontextmanager
        async def lifespan(app: Starlette) -> AsyncIterator[None]:
            async with session_manager.run():
                yield

        app = Starlette(
            debug=False,
            routes=[Mount("/mcp", app=handle_streamable_http)],
            lifespan=lifespan,
        )

        async def app_no_redirect(scope, receive, send):
            if scope["type"] == "http" and scope.get("path") == "/mcp":
                scope = dict(scope)
                scope["path"] = "/mcp/"
                scope["raw_path"] = b"/mcp/"
            await app(scope, receive, send)

        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["Mcp-Session-Id"],
        )

        @app.route("/health")
        async def health(request):
            return JSONResponse({"status": "ok", "server": "catalyst_center_mcp_remote"})

        config = uvicorn.Config(app_no_redirect, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Catalyst Center Remote MCP Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    args = parser.parse_args()

    try:
        WRAPPER_LOGGER.info("Starting wrapper: catalyst_center_remote.py -> catalyst_center_mcp (streamable_http)")
        asyncio.run(CatalystCenterRemoteServer().run(host=args.host, port=args.port))
    except Exception as e:
        WRAPPER_LOGGER.error("Server crashed: %s", str(e))
        raise
