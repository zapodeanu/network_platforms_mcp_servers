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
import yaml
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


# Create new multi-org MCP server, with the provided name
server = Server('multiorg_meraki_mcp_server')

# Load organization configuration
with open(PATH + '/Resources/meraki_organizations.yaml', 'r') as f:
    org_config = yaml.safe_load(f)

# Create organization mapping
ORGANIZATIONS = {}
for org in org_config['meraki_organizations']:
    if org['enabled']:
        ORGANIZATIONS[org['name']] = {
            'api_key': os.getenv(org['api_key_env']),
            'description': org['description']
        }

@server.list_tools()
async def list_tools():
    """List available tools that clients may call."""
    return [
        Tool(
            name='get_configured_organizations',
            description='🔍 DISCOVERY TOOL: Lists YOUR configured organizations (Prod, Lab, etc). Use this FIRST to discover available options.',
            inputSchema={
                'type': 'object',
                'properties': {},
                'additionalProperties': False
            }
        ),
        Tool(
            name='list_organizations_details',
            description='🏢 MERAKI API TOOL: Calls Meraki API to get organization details. REQUIRES organization parameter (Prod, Lab, or all).',
            inputSchema={
                'type': 'object',
                'properties': {
                    'organization': {
                        'type': 'string',
                        'description': 'REQUIRED: Organization name (Prod, Lab, or all). Get names from get_configured_organizations first.'
                    }
                },
                'required': ['organization'],
                'additionalProperties': False
            }
        ),
        Tool(
            name='list_networks',
            description='📋 List networks in Meraki organization. REQUIRES organization parameter (Prod, Lab, or all).',
            inputSchema={
                'type': 'object',
                'properties': {
                    'organization': {
                        'type': 'string',
                        'description': 'REQUIRED: Organization name (Prod, Lab, or all). Networks are organization-specific.'
                    }
                },
                'required': ['organization'],
                'additionalProperties': False
            }
        )
    ]


@server.call_tool()
async def call_tool(tool_name: str, arguments: dict):
    """Execute the requested tool with given name and arguments."""

    if tool_name == 'get_configured_organizations':
        return {
            'content': [list(ORGANIZATIONS.keys())]
        }

    elif tool_name == 'list_organizations_details':
        org_name = arguments['organization']

        if org_name == 'all':
            # Get all organizations
            all_orgs = []
            for name, config in ORGANIZATIONS.items():
                headers = {
                    'X-Cisco-Meraki-API-Key': config['api_key'],
                    'Content-Type': 'application/json'
                }
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{MERAKI_BASE_URL}/organizations", headers=headers)
                    orgs = response.json()
                    all_orgs.extend(orgs)
            return {'content': [all_orgs]}
        else:
            # Single organization
            api_key = ORGANIZATIONS[org_name]['api_key']
            headers = {
                'X-Cisco-Meraki-API-Key': api_key,
                'Content-Type': 'application/json'
            }
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{MERAKI_BASE_URL}/organizations", headers=headers)
                orgs = response.json()
                return {'content': [orgs]}

    elif tool_name == 'list_networks':
        org_name = arguments['organization']
        api_key = ORGANIZATIONS[org_name]['api_key']

        headers = {
            'X-Cisco-Meraki-API-Key': api_key,
            'Content-Type': 'application/json'
        }

        # First get org details to find org_id
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{MERAKI_BASE_URL}/organizations", headers=headers)
            orgs = response.json()
            org_id = orgs[0]['id']  # Assume first org

            # Get networks
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