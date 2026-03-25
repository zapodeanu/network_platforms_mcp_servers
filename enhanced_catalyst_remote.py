#!/usr/bin/env python3
"""
Enhanced Declarative Catalyst Center remote MCP server
Streamable HTTP transport wrapper around enhanced_declarative_catalyst.py
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

from enhanced_declarative_catalyst import EnhancedDeclarativeCatalystServer, PATH

# Ensure logs directory exists
os.makedirs(os.path.join(PATH, "logs"), exist_ok=True)

# Configure logging for remote server
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - PID:%(process)d - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(PATH, "logs/enhanced_catalyst_remote.log"), mode="a"),
    ],
    force=True,
)

logging.info("=" * 80)
logging.info("Enhanced Catalyst Remote Server Started")
logging.info("Process ID: %s", os.getpid())
logging.info("=" * 80)


class EnhancedCatalystRemoteServer(EnhancedDeclarativeCatalystServer):
    """Remote streamable HTTP version of the enhanced Catalyst server."""

    def __init__(self):
        super().__init__()

    async def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the remote MCP server over streamable HTTP."""
        logging.info("Starting enhanced remote Catalyst MCP server on http://%s:%s/mcp", host, port)

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
                logging.info("MCP streamable HTTP session manager started")
                try:
                    yield
                finally:
                    logging.info("MCP streamable HTTP session manager stopped")

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
            return JSONResponse({"status": "ok", "server": "enhanced_catalyst_remote"})

        config = uvicorn.Config(app_no_redirect, host=host, port=port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Catalyst Remote MCP Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    args = parser.parse_args()

    try:
        asyncio.run(EnhancedCatalystRemoteServer().run(host=args.host, port=args.port))
    except Exception as e:
        logging.error("Server crashed: %s", str(e))
        raise
