#!/usr/bin/env python3
"""
Meraki streamable HTTP entrypoint.
Do not add tool logic here; keep logic in meraki_core.py.
"""

import argparse
import asyncio
import contextlib
import logging
import os
from collections.abc import AsyncIterator

import uvicorn
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Mount

from meraki_core import EnhancedMultiOrgMerakiServer, PATH

os.makedirs(os.path.join(PATH, "logs"), exist_ok=True)
WRAPPER_LOGGER = logging.getLogger("meraki_remote_wrapper")


class MerakiRemoteServer(EnhancedMultiOrgMerakiServer):
    """Remote streamable HTTP transport wrapper."""

    async def run(self, host: str = "0.0.0.0", port: int = 8001):
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
            return JSONResponse({"status": "ok", "server": "meraki_mcp_remote"})

        config = uvicorn.Config(app_no_redirect, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Meraki Remote MCP Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8001, help="Port to listen on (default: 8001)")
    args = parser.parse_args()

    try:
        WRAPPER_LOGGER.info("Starting wrapper: meraki_remote.py -> meraki_mcp (streamable_http)")
        asyncio.run(MerakiRemoteServer().run(host=args.host, port=args.port))
    except Exception as e:
        WRAPPER_LOGGER.error("Server crashed: %s", str(e))
        raise
