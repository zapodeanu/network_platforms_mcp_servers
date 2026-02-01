#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2026 Cisco and/or its affiliates.
This software is licensed to you under the terms of the Cisco Sample
Code License, Version 1.1 (the "License"). You may obtain a copy of the
License at
               https://developer.cisco.com/docs/licenses
All use of the material herein must be in accordance with the terms of
the License. All rights not expressly granted by the License are
reserved. Unless required by applicable law or agreed to separately in
writing, software distributed under the License is distributed on an "AS
IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied.

Developed with assistance from Anthropic Claude
"""

import asyncio
import json
import os
import httpx

from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool


# Get the directory where this script is located
PATH = os.path.dirname(os.path.abspath(__file__))

# Load environment file from the same directory as the script
load_dotenv(os.path.join(PATH, 'environment.env'))

# Meraki API Key
MERAKI_API_KEY = os.getenv('MERAKI_API_KEY')
MERAKI_BASE_URL = "https://api.meraki.com/api/v1"


# Create new foundation MCP server, with the provided name
server = Server('foundation_meraki_mcp_server')

@server.list_tools()
async def list_tools():
    """List available tools that clients may call.
    Returns a list of Tool objects that define the available
    functionality this MCP server provides.
    """
    return [
        Tool(
            name='list_organizations_details',
            description='📡 MERAKI API TOOL: Calls Meraki API to get organization details (IDs, names, URLs)',
            inputSchema={
                'type': 'object',
                'properties': {},
                'additionalProperties': False
            }
        ),
        Tool(
            name='list_networks',
            description='📋 List all networks in a Meraki organization. REQUIRES organization id parameter, obtained from'
                        'list_organization_details tool',
            inputSchema={
                'type': 'object',
                'properties': {
                    'organization': {
                        'type': 'string',
                        'description': 'REQUIRED: Organization Id. Networks are organization-specific.'
                    }
                },
                'required': ['organization'],
                'additionalProperties': False
            }
        )
    ]


# noinspection PyUnusedLocal

@server.call_tool()
async def call_tool(tool_name: str, arguments: dict):
    """Execute the requested tool with given name and arguments.
    Makes authenticated API call to Meraki retrieve organization
    details and returns the response data, formatted as JSON
    """
    headers = {
        'X-Cisco-Meraki-API-Key': MERAKI_API_KEY,
        'Content-Type': 'application/json'
    }

    async with httpx.AsyncClient() as client:
        if tool_name == 'list_organizations_details':
            response = await client.get(f"{MERAKI_BASE_URL}/organizations", headers=headers)
            orgs = response.json()
            return {'content': [orgs]}

        elif tool_name == 'list_networks':
            org_id = arguments['organization']
            response = await client.get(f"{MERAKI_BASE_URL}/organizations/{org_id}/networks", headers=headers)
            networks = response.json()
            return {'content': [networks]}


async def main():
    """Initialize and run the MCP server.
    Sets up stdio communication streams and starts the server
    to listen for client requests.
    """
    async with stdio_server() as streams:
        await server.run(streams[0], streams[1], server.create_initialization_options())


if __name__ == '__main__':
    """Run the MCP server"""
    asyncio.run(main())