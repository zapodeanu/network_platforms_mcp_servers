#!/usr/bin/env python3
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

Enhanced Declarative Catalyst Center MCP Server
Combines YAML-configured declarative tools with cosine similarity-based API exploration
Focus: Exploration of complete API coverage beyond existing YAML tools
Security: API Explorer restricted to read-only (GET) operations for safety

Developed with assistance from Anthropic Claude
"""

import asyncio
import json
import os
import yaml
import httpx
import logging
import sys
import base64
import pickle
import numpy as np

from datetime import datetime, timedelta
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, ListToolsResult, Tool, TextContent

# Get the directory where this script is located
PATH = os.path.dirname(os.path.abspath(__file__))

# Load environment file from the same directory as the script
load_dotenv(os.path.join(PATH, 'environment.env'))

# Catalyst Center access
CC_URL = os.getenv('CC_URL')
CC_USER = os.getenv('CC_USER')
CC_PASS = os.getenv('CC_PASS')

# Ensure logs directory exists
os.makedirs(os.path.join(PATH, "logs"), exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - PID:%(process)d - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PATH + '/logs/enhanced_declarative_catalyst.log', mode='a'),
    ],
    force=True
)

logging.info("=" * 80)
logging.info(f"Enhanced Catalyst Center Server with Complete API Explorer Started")
logging.info(f"Process ID: {os.getpid()}")
logging.info(f"CC URL: {CC_URL}")
logging.info("=" * 80)


class CatalystCenterTokenManager:
    """Manages authentication tokens for Catalyst Center"""

    def __init__(self, base_url: str, username: str, password: str):
        self.base_url = base_url
        self.username = username
        self.password = password
        self.token = None
        self.token_expires = None

    async def get_valid_token(self) -> str:
        """Get a valid token, refreshing if necessary"""
        if self.token and self.token_expires and datetime.now() < self.token_expires:
            return self.token
        return await self._get_new_token()

    async def _get_new_token(self) -> str:
        """Get a new authentication token from Catalyst Center"""
        try:
            url = f"{self.base_url}/dna/system/api/v1/auth/token"
            credentials = base64.b64encode(f"{self.username}:{self.password}".encode()).decode()

            headers = {
                "Authorization": f"Basic {credentials}",
                "Content-Type": "application/json"
            }

            async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
                response = await client.post(url, headers=headers)

                if response.status_code == 200:
                    result = response.json()
                    self.token = result.get('Token')
                    if self.token:
                        self.token_expires = datetime.now() + timedelta(hours=1)
                        return self.token
                    else:
                        raise Exception("No token found in authentication response")
                else:
                    raise Exception(f"Authentication failed: {response.status_code} - {response.text}")

        except Exception as e:
            error_msg = f"Token generation failed: {str(e)}"
            logging.error(error_msg)
            raise Exception(error_msg)


class CatalystAPIExplorer:
    """Cosine similarity-based API exploration using sentence transformers"""

    def __init__(self, swagger_file_path: str, cache_dir: str = None):
        self.swagger_file_path = swagger_file_path
        self.cache_dir = cache_dir or f"{PATH}/embedding_cache"
        self.model = None
        self.embeddings_data = []
        self._initialize()

    def _initialize(self):
        """Initialize the sentence transformer model and load/build embeddings"""
        logging.info("Initializing cosine similarity API explorer...")

        # Load sentence transformer model
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logging.info("Sentence transformer model loaded")
        except Exception as e:
            logging.error(f"Failed to load sentence transformer: {e}")
            return

        # Load or build embeddings
        self._load_or_build_embeddings()

    def _load_or_build_embeddings(self):
        """Load cached embeddings or build new ones"""
        os.makedirs(self.cache_dir, exist_ok=True)

        # Check if swagger file exists
        if not os.path.exists(self.swagger_file_path):
            logging.error(f"Swagger file not found: {self.swagger_file_path}")
            return

        # Create cache filename based on swagger file modification time
        swagger_mtime = os.path.getmtime(self.swagger_file_path)
        cache_file = f"{self.cache_dir}/catalyst_embeddings_{int(swagger_mtime)}.pkl"

        if os.path.exists(cache_file):
            logging.info(f"Loading cached embeddings from: {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    self.embeddings_data = pickle.load(f)
                logging.info(f"Loaded {len(self.embeddings_data)} cached embeddings")
                return
            except Exception as e:
                logging.warning(f"Failed to load cached embeddings: {e}")

        # Build new embeddings
        logging.info("Building new embeddings from swagger file...")
        self._build_embeddings(cache_file)

    def _build_embeddings(self, cache_file: str):
        """Build embeddings from swagger specification"""
        try:
            with open(self.swagger_file_path, 'r') as f:
                swagger_spec = json.load(f)
            paths = swagger_spec.get('paths', {})
            self.embeddings_data = []

            for path, path_spec in paths.items():
                for method, operation_spec in path_spec.items():
                    # Only process read-only GET operations for security
                    if method.lower() in ['get']:
                        # Create rich text for embedding
                        summary = operation_spec.get('summary', '')
                        description = operation_spec.get('description', '')
                        tags = ' '.join(operation_spec.get('tags', []))

                        # Combine multiple text sources for richer embeddings
                        embedding_text = f"{summary} {description} {path} {tags}"

                        # Generate embedding
                        embedding = self.model.encode(embedding_text)

                        # Store with metadata
                        self.embeddings_data.append({
                            'embedding': embedding,
                            'metadata': {
                                'path': path,
                                'method': method.upper(),
                                'endpoint_id': f"{method.upper()}:{path}",
                                'summary': summary,
                                'description': description,
                                'tags': operation_spec.get('tags', []),
                                'operation_id': operation_spec.get('operationId', ''),
                                'parameters': operation_spec.get('parameters', []),
                                'category': self._categorize_endpoint(path, operation_spec),
                                'embedding_text': embedding_text
                            }
                        })

            # Cache the embeddings
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(self.embeddings_data, f)
                logging.info(f"Cached {len(self.embeddings_data)} embeddings to: {cache_file}")
            except Exception as e:
                logging.warning(f"Failed to cache embeddings: {e}")

            logging.info(f"Built {len(self.embeddings_data)} embeddings")

        except Exception as e:
            logging.error(f"Failed to build embeddings: {e}")

    def _categorize_endpoint(self, path: str, operation_spec: dict):
        """Categorize API endpoint based on path and operation"""
        path_lower = path.lower()

        if any(keyword in path_lower for keyword in ['site', 'device', 'topology', 'compliance', 'inventory']):
            return 'inventory'
        elif any(keyword in path_lower for keyword in ['issue', 'client', 'interface', 'health', 'assurance']):
            return 'assurance'
        elif any(keyword in path_lower for keyword in ['task', 'command', 'execute', 'job', 'file']):
            return 'operations'
        elif any(keyword in path_lower for keyword in ['system', 'auth', 'user', 'setting']):
            return 'administration'
        else:
            return 'general'

    def search_endpoints(self, query: str, category: str = None, method: str = None, limit: int = 20):
        """Search for API endpoints using cosine similarity"""
        if not self.model or not self.embeddings_data:
            logging.error("Embeddings not initialized")
            return []

        try:
            # Generate query embedding
            query_embedding = self.model.encode(query)

            # Calculate similarities
            similarities = []
            for item in self.embeddings_data:
                # Apply filters
                metadata = item['metadata']
                if category and metadata['category'] != category.lower():
                    continue
                if method and metadata['method'] != method.upper():
                    continue

                # Calculate cosine similarity
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    item['embedding'].reshape(1, -1)
                )[0][0]

                similarities.append({
                    'similarity_score': float(similarity),
                    'metadata': metadata
                })

            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
            return similarities[:limit]

        except Exception as e:
            logging.error(f"Search failed: {e}")
            return []

    def get_endpoint_info(self, path: str, method: str = 'GET'):
        """Get detailed information about a specific endpoint"""
        endpoint_id = f"{method.upper()}:{path}"

        for item in self.embeddings_data:
            if item['metadata']['endpoint_id'] == endpoint_id:
                return item['metadata']

        return None


class ExplorerAnalytics:
    """Analytics manager for API explorer tracking"""

    def __init__(self, analytics_file_path: str):
        self.analytics_file = analytics_file_path
        self.explorer_analytics = {
            "successful_calls": [],
            "failed_calls": [],
            "call_count": 0,
            "success_rate": 0.0
        }
        self._load_analytics()

    def _load_analytics(self):
        """Load existing analytics data from file"""
        try:
            if os.path.exists(self.analytics_file):
                with open(self.analytics_file, 'r') as f:
                    self.explorer_analytics = json.load(f)
                logging.info(
                    f"Loaded existing explorer analytics: {len(self.explorer_analytics.get('successful_calls', []))} successful calls")
        except Exception as e:
            logging.warning(f"Could not load explorer analytics file: {e}")

    def _save_analytics(self):
        """Save analytics data to file"""
        try:
            # Calculate success rate
            total_calls = len(self.explorer_analytics['successful_calls']) + len(
                self.explorer_analytics['failed_calls'])
            if total_calls > 0:
                self.explorer_analytics['success_rate'] = len(
                    self.explorer_analytics['successful_calls']) / total_calls

            self.explorer_analytics['call_count'] = total_calls

            with open(self.analytics_file, 'w') as f:
                json.dump(self.explorer_analytics, f, indent=2, default=str)
        except Exception as e:
            logging.error(f"Could not save explorer analytics: {e}")

    def _record_explorer_call(self, path: str, method: str, parameters: dict, success: bool, response_data=None,
                              error_message=None):
        """Record an explorer API call for analytics"""
        timestamp = datetime.now().isoformat()

        call_record = {
            "timestamp": timestamp,
            "path": path,
            "method": method,
            "parameters": parameters,
            "endpoint_id": f"{method}:{path}"
        }

        if success:
            call_record.update({
                "status": "success",
                "response_size": len(json.dumps(response_data)) if response_data else 0,
                "data_returned": bool(response_data)
            })
            self.explorer_analytics['successful_calls'].append(call_record)
            logging.info(f"Explorer analytics: {method} {path} - SUCCESS")
        else:
            call_record.update({
                "status": "failed",
                "error": error_message
            })
            self.explorer_analytics['failed_calls'].append(call_record)
            logging.info(f"Explorer analytics: {method} {path} - FAILED: {error_message}")

        # Save analytics after each call
        self._save_analytics()

    def _get_catalyst_explorer_analytics(self):
        """Get analytics summary for popular endpoints"""
        successful_calls = self.explorer_analytics.get('successful_calls', [])

        # Count calls by endpoint
        endpoint_stats = {}
        for call in successful_calls:
            endpoint_id = call.get('endpoint_id')
            if endpoint_id not in endpoint_stats:
                endpoint_stats[endpoint_id] = {
                    "path": call.get('path'),
                    "method": call.get('method'),
                    "call_count": 0,
                    "last_used": call.get('timestamp'),
                    "avg_response_size": 0
                }

            endpoint_stats[endpoint_id]["call_count"] += 1
            endpoint_stats[endpoint_id]["last_used"] = max(endpoint_stats[endpoint_id]["last_used"],
                                                           call.get('timestamp'))

        # Sort by popularity
        popular_endpoints = sorted(endpoint_stats.items(), key=lambda x: x[1]["call_count"], reverse=True)

        return {
            "total_successful_calls": len(successful_calls),
            "total_failed_calls": len(self.explorer_analytics.get('failed_calls', [])),
            "success_rate": self.explorer_analytics.get('success_rate', 0.0),
            "unique_successful_endpoints": len(endpoint_stats),
            "most_popular_endpoints": popular_endpoints[:10],
            "candidates_for_yaml_promotion": [ep for ep in popular_endpoints[:5] if ep[1]["call_count"] >= 3]
        }


class CatalystClusterManager:
    """Simple manager for reading Catalyst Center clusters"""

    def __init__(self, clusters_config_path: str, username: str, password: str):
        self.clusters_config_path = clusters_config_path
        self.username = username
        self.password = password
        self.clusters = {}
        self.token_managers = {}
        self._load_clusters()

    def _load_clusters(self):
        """Load cluster configuration from YAML file"""
        logging.info("Loading Catalyst Center clusters configuration...")

        if not os.path.exists(self.clusters_config_path):
            logging.error(f"Clusters config file not found: {self.clusters_config_path}")
            return

        try:
            with open(self.clusters_config_path, 'r') as f:
                config = yaml.safe_load(f)

            if 'catalyst_centers' not in config:
                logging.error("Clusters config must include 'catalyst_centers' section")
                return

            for cluster_config in config['catalyst_centers']:
                cluster_name = cluster_config.get('name')
                cluster_host = cluster_config.get('host')
                cluster_enabled = cluster_config.get('enabled', True)

                if not cluster_name or not cluster_host or not cluster_enabled:
                    continue

                self.clusters[cluster_name] = {
                    'name': cluster_name,
                    'host': cluster_host,
                    'base_url': f"https://{cluster_host}",
                    'enabled': cluster_enabled
                }

                self.token_managers[cluster_name] = CatalystCenterTokenManager(
                    f"https://{cluster_host}",
                    self.username,
                    self.password
                )

            logging.info(f"Successfully loaded {len(self.clusters)} clusters")

        except Exception as e:
            logging.error(f"Failed to load clusters config: {e}")

    def get_clusters(self) -> dict:
        return self.clusters

    def get_cluster_names(self) -> list:
        return list(self.clusters.keys())

    def get_token_manager(self, cluster_name: str) -> CatalystCenterTokenManager:
        return self.token_managers.get(cluster_name)


class EnhancedDeclarativeCatalystServer:
    def __init__(self):
        logging.info("Initializing Enhanced Declarative Catalyst Server...")

        self.server = Server("enhanced-declarative-catalyst")

        # Initialize credentials
        self.cc_url = CC_URL
        self.username = CC_USER
        self.password = CC_PASS

        if not all([self.cc_url, self.username, self.password]):
            raise ValueError("Missing required Catalyst Center credentials")

        # Initialize analytics tracking with new naming
        self.analytics_file = f"{PATH}/logs/enhanced_declarative_catalyst_explorer_analytics.json"
        self.analytics_manager = ExplorerAnalytics(self.analytics_file)

        # Initialize managers
        self.token_manager = CatalystCenterTokenManager(
            self.cc_url, self.username, self.password
        )

        clusters_config_path = PATH + "/Resources/catalyst_center_clusters.yaml"
        self.cluster_manager = CatalystClusterManager(clusters_config_path, self.username, self.password)

        # Load configuration
        self.config = self._load_config()

        # Initialize cosine similarity API explorer
        swagger_file_path = PATH + "/Resources/CC_swagger.json"
        self.api_explorer = CatalystAPIExplorer(swagger_file_path)

        # Build YAML endpoint lookup for filtering
        self.yaml_endpoints = self._build_yaml_endpoint_lookup()

        # Organize tools by category
        self.tool_categories = self._organize_tools_by_category()

        self._setup()
        logging.info("Enhanced Declarative Catalyst Server ready!")

    def _load_config(self):
        """Load YAML config"""
        config_path = PATH + "/Resources/catalyst_config.yaml"

        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logging.info(f"Loaded config with {len(config.get('tools', []))} declarative tools")
            return config
        else:
            logging.warning("YAML config not found, using minimal default")
            return {
                "base_url": self.cc_url,
                "tools": []
            }

    def _build_yaml_endpoint_lookup(self):
        """Build lookup of endpoints covered by YAML tools"""
        yaml_endpoints = set()
        for tool in self.config.get("tools", []):
            endpoint = tool.get("endpoint", "")
            method = tool.get("method", "GET").upper()
            # Normalize endpoint path
            normalized_path = f"/{endpoint.strip('/')}"
            yaml_endpoints.add(f"{method}:{normalized_path}")
        return yaml_endpoints

    def _organize_tools_by_category(self):
        """Organize tools by category"""
        categories = {}
        for tool in self.config.get("tools", []):
            category = tool.get("category", "uncategorized")
            if category not in categories:
                categories[category] = []
            categories[category].append(tool)
        return categories

    def _setup(self):
        """Setup MCP server handlers"""
        logging.info("Setting up enhanced MCP server handlers...")

        @self.server.list_tools()
        async def list_tools():
            """Generate tools from YAML config plus explorer tools"""
            tools = []

            # Add declarative tools from YAML config
            for tool_config in self.config.get("tools", []):
                try:
                    category = tool_config.get("category", "uncategorized")
                    emoji = {"inventory": "📊", "assurance": "🚨", "operations": "⚙️", "administration": "🔧"}.get(
                        category, "🔧")

                    # Build schema from parameters
                    properties = {}
                    required = []

                    for param_name, param_info in tool_config.get("parameters", {}).items():
                        properties[param_name] = {
                            "type": param_info["type"],
                            "description": param_info["description"]
                        }
                        if param_info.get("required", False):
                            required.append(param_name)

                    enhanced_description = f"{emoji} [{category.title()}] {tool_config['description']}"

                    tool = Tool(
                        name=tool_config["name"],
                        description=enhanced_description,
                        inputSchema={
                            "type": "object",
                            "properties": properties,
                            "required": required
                        }
                    )
                    tools.append(tool)

                except Exception as e:
                    logging.error(f"Failed to create tool {tool_config.get('name', 'unknown')}: {e}")

            # Add explorer tools
            explorer_tools = [
                Tool(
                    name="explore_catalyst_api_endpoints",
                    description="🔍 [Explorer] Search Catalyst Center API endpoints using natural language queries",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language description of what you want to do (e.g., 'get network devices', 'site information', 'client status')"
                            },
                            "category": {
                                "type": "string",
                                "description": "Optional category filter: inventory, assurance, operations, administration",
                                "enum": ["inventory", "assurance", "operations", "administration"]
                            },
                            "method": {
                                "type": "string",
                                "description": "Optional HTTP method filter (read-only operations only)",
                                "enum": ["GET"]
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results (default: 10)",
                                "default": 10,
                                "minimum": 1,
                                "maximum": 50
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="execute_catalyst_api_endpoint",
                    description="⚡ [Execute] Execute any Catalyst Center API endpoint dynamically",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "API endpoint path (e.g., '/dna/intent/api/v1/network-device')"
                            },
                            "method": {
                                "type": "string",
                                "description": "HTTP method (read-only operations only)",
                                "enum": ["GET"],
                                "default": "GET"
                            },
                            "parameters": {
                                "type": "object",
                                "description": "Query parameters, path parameters, or request body",
                                "additionalProperties": True
                            },
                            "cluster": {
                                "type": "string",
                                "description": "Cluster name or 'all' for network-wide results (optional)"
                            }
                        },
                        "required": ["path"]
                    }
                ),
                Tool(
                    name="get_catalyst_endpoint_info",
                    description="📋 [Info] Get detailed information about an API endpoint including parameters and usage",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "API endpoint path"
                            },
                            "method": {
                                "type": "string",
                                "description": "HTTP method (read-only operations only)",
                                "enum": ["GET"],
                                "default": "GET"
                            }
                        },
                        "required": ["path"]
                    }
                ),
                Tool(
                    name="get_catalyst_explorer_analytics",
                    description="📈 [Analytics] Get Catalyst explorer analytics data for explored Catalyst API endpoints showing usage patterns and success rates",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "detailed": {
                                "type": "boolean",
                                "description": "Include detailed call history (default: false)",
                                "default": False
                            }
                        },
                        "required": []
                    }
                )
            ]

            tools.extend(explorer_tools)

            logging.info(
                f"Created {len(tools)} total tools ({len(self.config.get('tools', []))} declarative + {len(explorer_tools)} explorer)")
            return tools

        @self.server.list_resources()
        async def list_resources():
            return []

        @self.server.list_prompts()
        async def list_prompts():
            return []

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict):
            """Handle tool calls for both declarative and explorer tools"""
            logging.info(f"Executing tool: {name}")

            try:
                # Handle explorer tools
                if name == "explore_catalyst_api_endpoints":
                    return await self._handle_cosine_search(arguments)
                elif name == "execute_catalyst_api_endpoint":
                    return await self._handle_execute_explored_endpoint(arguments)
                elif name == "get_catalyst_endpoint_info":
                    return await self._handle_get_endpoint_info(arguments)
                elif name == "get_catalyst_explorer_analytics":
                    return await self._handle_get_analytics(arguments)

                # Handle declarative tools
                return await self._handle_declarative_tool(name, arguments)

            except Exception as e:
                logging.error(f"Error executing {name}: {e}")
                return {'content': [{'type': 'text', 'text': f'Error: {str(e)}'}]}

        logging.info("Enhanced MCP server handlers registered")

    async def _handle_cosine_search(self, arguments: dict):
        """API explorer search for complete API coverage - endpoints not already exposed as YAML tools"""
        query = arguments.get('query', '')
        category = arguments.get('category')
        method = arguments.get('method')
        limit = arguments.get('limit', 10)
        min_similarity = arguments.get('min_similarity', 0.45)

        logging.info(f"API explorer search for complete endpoints: '{query}' (min_similarity: {min_similarity})")

        # Get all cosine similarity results
        all_explored_results = self.api_explorer.search_endpoints(query, category, method, limit * 2)

        # Filter OUT endpoints already covered by YAML tools AND below similarity threshold
        complete_endpoints = []
        yaml_covered_endpoints = []

        for result in all_explored_results:
            metadata = result['metadata']

            # Apply minimum similarity filter
            if result['similarity_score'] < min_similarity:
                continue

            # Security: Only allow read-only GET operations
            if metadata['method'].upper() != 'GET':
                continue

            # Check if this endpoint is already covered by a YAML tool
            yaml_equivalent = self._find_yaml_equivalent(metadata['path'], metadata['method'])

            if yaml_equivalent:
                yaml_covered_endpoints.append({
                    "explored_path": metadata['path'],
                    "explored_method": metadata['method'],
                    "yaml_tool": yaml_equivalent['tool_name'],
                    "similarity_score": round(result['similarity_score'], 3),
                    "note": f"Already available as tool: {yaml_equivalent['tool_name']}"
                })
            else:
                complete_endpoints.append({
                    "path": metadata['path'],
                    "method": metadata['method'],
                    "summary": metadata['summary'],
                    "description": metadata['description'],
                    "similarity_score": round(result['similarity_score'], 3),
                    "category": metadata['category'],
                    "tags": metadata['tags'],
                    "status": "explored",
                    "usage": f"execute_catalyst_api_endpoint with path: {metadata['path']}"
                })

        # Limit complete endpoints to requested limit
        complete_endpoints = complete_endpoints[:limit]

        search_results = {
            "query": query,
            "search_purpose": "explore_complete_api_coverage",
            "note": "YAML tools are already available to client via list_tools()",
            "filters_applied": {
                "category": category,
                "method": method,
                "limit": limit,
                "min_similarity": min_similarity
            },
            "summary": {
                "complete_endpoints_found": len(complete_endpoints),
                "yaml_covered_endpoints": len(yaml_covered_endpoints),
                "total_searched": len(all_explored_results)
            },
            "explored_endpoints": complete_endpoints,
            "already_available": {
                "description": "These endpoints are already exposed as YAML tools",
                "count": len(yaml_covered_endpoints),
                "examples": yaml_covered_endpoints[:5]  # Show first 5 as examples
            },
            "recommendations": self._generate_explorer_recommendations(complete_endpoints, yaml_covered_endpoints)
        }

        return {'content': [{'type': 'text', 'text': json.dumps(search_results, indent=2)}]}

    def _find_yaml_equivalent(self, path: str, method: str):
        """Find if an explored endpoint has a YAML tool equivalent"""
        normalized_path = f"/{path.strip('/')}"

        for tool in self.config.get("tools", []):
            tool_path = f"/{tool.get('endpoint', '').strip('/')}"
            tool_method = tool.get('method', 'GET').upper()

            if tool_path == normalized_path and tool_method == method.upper():
                return {
                    "tool_name": tool["name"],
                    "description": tool["description"]
                }

        return None

    def _generate_explorer_recommendations(self, complete_endpoints, yaml_covered):
        """Generate recommendations focused on exploration value"""
        recommendations = []

        if complete_endpoints:
            high_value_endpoints = [ep for ep in complete_endpoints if ep['similarity_score'] > 0.8]

            if high_value_endpoints:
                recommendations.append({
                    "type": "high_value_exploration",
                    "message": f"Found {len(high_value_endpoints)} highly relevant endpoints for complete coverage",
                    "suggestion": "These might be valuable additions to your YAML tool configuration",
                    "candidates": [ep['path'] for ep in high_value_endpoints[:3]]
                })

            if len(complete_endpoints) > 5:
                recommendations.append({
                    "type": "extensive_exploration",
                    "message": f"Explored {len(complete_endpoints)} endpoints for complete API coverage",
                    "suggestion": "Consider exploring these with execute_catalyst_api_endpoint to evaluate their utility"
                })

        if yaml_covered:
            recommendations.append({
                "type": "existing_coverage",
                "message": f"{len(yaml_covered)} relevant endpoints are already available as tools",
                "suggestion": "Use the existing YAML tools which have proper parameter validation and error handling"
            })

        if not complete_endpoints and not yaml_covered:
            recommendations.append({
                "type": "no_exploration",
                "message": "No matching endpoints found",
                "suggestion": "Try broader search terms or explore related functionality"
            })

        return recommendations

    async def _handle_get_endpoint_info(self, arguments: dict):
        """Handle endpoint info requests"""
        path = arguments.get('path', '')
        method = arguments.get('method', 'GET').upper()

        endpoint_info = self.api_explorer.get_endpoint_info(path, method)

        if not endpoint_info:
            return {'content': [{'type': 'text', 'text': f'Endpoint not found: {method} {path}'}]}

        return {'content': [{'type': 'text', 'text': json.dumps(endpoint_info, indent=2)}]}

    async def _handle_get_analytics(self, arguments: dict):
        """Handle explorer analytics reporting requests"""
        detailed = arguments.get('detailed', False)

        analytics_summary = self.analytics_manager._get_catalyst_explorer_analytics()

        if detailed:
            # Include full call history
            analytics_summary["recent_successful_calls"] = self.analytics_manager.explorer_analytics.get(
                'successful_calls', [])[-20:]  # Last 20
            analytics_summary["recent_failed_calls"] = self.analytics_manager.explorer_analytics.get('failed_calls',
                                                                                                     [])[
                                                       -10:]  # Last 10

        return {'content': [{'type': 'text', 'text': json.dumps(analytics_summary, indent=2)}]}

    async def _handle_execute_explored_endpoint(self, arguments: dict):
        """Execute an explored API endpoint with analytics tracking"""
        path = arguments.get('path', '')
        method = arguments.get('method', 'GET').upper()
        parameters = arguments.get('parameters', {})
        cluster = arguments.get('cluster')

        logging.info(f"Executing explored endpoint: {method} {path}")

        # Security: Only allow read-only GET operations
        if method.upper() != 'GET':
            error_response = {
                "error": "Security restriction: Only GET methods allowed",
                "provided_method": method,
                "allowed_methods": ["GET"],
                "reason": "Write operations (POST/PUT/DELETE) are disabled for safety",
                "suggestion": "Use existing YAML tools for configuration changes"
            }
            return {'content': [{'type': 'text', 'text': json.dumps(error_response, indent=2)}]}

        # Check if this endpoint is covered by an existing YAML tool
        yaml_equivalent = self._find_yaml_equivalent(path, method)

        if yaml_equivalent:
            warning_message = {
                "warning": "Endpoint covered by existing YAML tool",
                "recommendation": f"Use existing tool: {yaml_equivalent['tool_name']}",
                "yaml_tool_info": yaml_equivalent,
                "execution_result": "Proceeding with dynamic execution, but consider using the YAML tool instead"
            }

            # Still execute but include the warning
            execution_result = await self._execute_on_clusters_with_analytics(path, method, parameters, cluster)

            # Merge warning with execution result
            if isinstance(execution_result.get('content', [{}])[0].get('text'), str):
                result_data = json.loads(execution_result['content'][0]['text'])
                result_data['yaml_tool_warning'] = warning_message
                execution_result['content'][0]['text'] = json.dumps(result_data, indent=2)

            return execution_result

        # Proceed with normal dynamic execution with analytics
        return await self._execute_on_clusters_with_analytics(path, method, parameters, cluster)

    async def _execute_on_clusters_with_analytics(self, path: str, method: str, parameters: dict, cluster_param: str):
        """Execute API call on specified clusters with analytics tracking"""
        # Handle cluster selection
        available_clusters = self.cluster_manager.get_clusters()

        if not available_clusters:
            target_clusters = [{'name': 'Default', 'base_url': self.cc_url}]
        elif cluster_param and cluster_param.lower() == 'all':
            target_clusters = list(available_clusters.values())
        elif cluster_param and cluster_param.lower() in [c.lower() for c in available_clusters.keys()]:
            cluster_name = next(c for c in available_clusters.keys() if c.lower() == cluster_param.lower())
            target_clusters = [available_clusters[cluster_name]]
        else:
            target_clusters = [list(available_clusters.values())[0]] if available_clusters else [
                {'name': 'Default', 'base_url': self.cc_url}]

        all_results = []
        analytics_success = False
        analytics_error = None
        analytics_response_data = None

        for cluster in target_clusters:
            cluster_name = cluster['name']

            try:
                # Get authentication token
                if cluster_name in self.cluster_manager.token_managers:
                    token_manager = self.cluster_manager.get_token_manager(cluster_name)
                    token = await token_manager.get_valid_token()
                else:
                    token = await self.token_manager.get_valid_token()

                # Build URL
                url = f"{cluster['base_url']}/{path.lstrip('/')}"

                # Handle path parameters
                for param_name, param_value in parameters.items():
                    if f"{{{param_name}}}" in url:
                        url = url.replace(f"{{{param_name}}}", str(param_value))

                # Separate query and body parameters
                query_params = {k: v for k, v in parameters.items() if f"{{{k}}}" not in path}

                headers = {
                    "X-Auth-Token": token,
                    "Content-Type": "application/json"
                }

                async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
                    if method == "GET":
                        response = await client.get(url, headers=headers, params=query_params)
                    elif method == "POST":
                        response = await client.post(url, headers=headers, json=query_params)
                    elif method == "PUT":
                        response = await client.put(url, headers=headers, json=query_params)
                    elif method == "DELETE":
                        response = await client.delete(url, headers=headers, params=query_params)
                    elif method == "PATCH":
                        response = await client.patch(url, headers=headers, json=query_params)

                    if response.status_code in [200, 201, 202]:
                        result = response.json() if response.text else {"status": "success"}
                        cluster_result = {
                            "cluster_info": {
                                "name": cluster_name,
                                "host": cluster.get('host', 'localhost')
                            },
                            "data": result
                        }
                        all_results.append(cluster_result)

                        # Mark as successful for analytics
                        analytics_success = True
                        analytics_response_data = result
                    else:
                        error_msg = f"API request failed: {response.status_code} - {response.text}"
                        error_result = {
                            "cluster_info": {"name": cluster_name},
                            "error": error_msg
                        }
                        all_results.append(error_result)

                        # Record error for analytics
                        if not analytics_error:  # Only record first error
                            analytics_error = error_msg

            except Exception as e:
                error_msg = f"Execution error: {str(e)}"
                error_result = {
                    "cluster_info": {"name": cluster_name},
                    "error": error_msg
                }
                all_results.append(error_result)

                # Record error for analytics
                if not analytics_error:  # Only record first error
                    analytics_error = error_msg

        # Record analytics for this explorer call
        self.analytics_manager._record_explorer_call(
            path=path,
            method=method,
            parameters=parameters,
            success=analytics_success,
            response_data=analytics_response_data,
            error_message=analytics_error
        )

        # Format response
        if len(target_clusters) == 1:
            final_result = all_results[0] if all_results else {"error": "No results"}
        else:
            final_result = {
                "network_wide_query": True,
                "total_clusters_queried": len(target_clusters),
                "results": all_results
            }

        return {'content': [{'type': 'text', 'text': json.dumps(final_result, indent=2)}]}

    async def _handle_declarative_tool(self, name: str, arguments: dict):
        """Handle declarative tools from YAML config"""
        # Find tool config
        tool_config = None
        for tool in self.config.get("tools", []):
            if tool["name"] == name:
                tool_config = tool
                break

        if not tool_config:
            return {'content': [{'type': 'text', 'text': f'Unknown declarative tool: {name}'}]}

        # Handle static response tools
        if tool_config.get("static_response"):
            logging.info(f"Returning static response for tool: {name}")
            static_data = tool_config["static_response"]
            return {'content': [{'type': 'text', 'text': json.dumps(static_data, indent=2)}]}

        # Handle special dynamic tools
        if name == "get_clusters":
            logging.info("Returning dynamic cluster information")
            clusters_info = {
                "clusters": self.cluster_manager.get_clusters(),
                "total_clusters": len(self.cluster_manager.get_clusters()),
                "cluster_names": self.cluster_manager.get_cluster_names()
            }
            return {'content': [{'type': 'text', 'text': json.dumps(clusters_info, indent=2)}]}

        # Handle delegator tools (like get_config_changes)
        if tool_config.get("parameters", {}).get("delegate_to"):
            delegate_to_param = tool_config["parameters"]["delegate_to"]
            delegate_target = delegate_to_param.get("value")

            if delegate_target:
                logging.info(f"Delegating {name} to {delegate_target}")

                # Build arguments for the delegate tool
                delegate_args = {"cluster": arguments.get("cluster", "")}

                # Add predefined parameter values from the tool config
                for param_name, param_info in tool_config.get("parameters", {}).items():
                    if "value" in param_info and param_name != "delegate_to":
                        delegate_args[param_name] = param_info["value"]
                        logging.info(f"Using predefined parameter: {param_name}={param_info['value']}")

                # Only add user arguments that don't have predefined values
                predefined_params = {param_name for param_name, param_info in
                                     tool_config.get("parameters", {}).items()
                                     if "value" in param_info}

                for key, value in arguments.items():
                    if key not in ["cluster", "delegate_to"] and key not in predefined_params:
                        delegate_args[key] = value
                        logging.info(f"User parameter: {key}={value}")
                    elif key in predefined_params:
                        logging.info(f"Ignoring user override for: {key} (using predefined value)")

                logging.info(f"Calling {delegate_target} with args: {delegate_args}")
                return await self._handle_declarative_tool(delegate_target, delegate_args)

        # Handle cluster selection
        cluster_param = arguments.get('cluster', '').lower() if arguments.get('cluster') else None
        available_clusters = self.cluster_manager.get_clusters()

        if not available_clusters:
            target_clusters = [{'name': 'Default', 'base_url': self.cc_url}]
        elif cluster_param == 'all':
            target_clusters = list(available_clusters.values())
        elif cluster_param and cluster_param in [c.lower() for c in available_clusters.keys()]:
            cluster_name = next(c for c in available_clusters.keys() if c.lower() == cluster_param)
            target_clusters = [available_clusters[cluster_name]]
        else:
            target_clusters = [list(available_clusters.values())[0]] if available_clusters else [
                {'name': 'Default', 'base_url': self.cc_url}]

        # Remove special parameters from API arguments
        api_arguments = {k: v for k, v in arguments.items()
                         if k not in ['cluster'] and
                         tool_config.get("parameters", {}).get(k, {}).get("location") != "special"}

        # Execute on target clusters
        all_results = []

        for cluster in target_clusters:
            cluster_name = cluster['name']

            try:
                # Get authentication token
                if cluster_name in self.cluster_manager.token_managers:
                    token_manager = self.cluster_manager.get_token_manager(cluster_name)
                    token = await token_manager.get_valid_token()
                else:
                    token = await self.token_manager.get_valid_token()

                # Build URL
                url = f"{cluster['base_url']}/{tool_config['endpoint']}"

                # Replace path parameters
                for param_name, param_info in tool_config.get("parameters", {}).items():
                    if param_info.get("location") == "path" and param_name in api_arguments:
                        url = url.replace(f"{{{param_name}}}", str(api_arguments[param_name]))

                # Build query parameters, body parameters, and header parameters
                query_params = {}
                body_params = {}
                header_params = {}

                for param_name, param_info in tool_config.get("parameters", {}).items():
                    if param_info.get("location") == "query" and param_name in api_arguments:
                        query_params[param_name] = api_arguments[param_name]
                    elif param_info.get("location") == "body" and param_name in api_arguments:
                        body_params[param_name] = api_arguments[param_name]
                    elif param_info.get("location") == "header" and param_name in api_arguments:
                        header_params[param_name] = api_arguments[param_name]

                # Make API call
                headers = {
                    "X-Auth-Token": token,
                    "Content-Type": "application/json"
                }

                # Add custom header parameters
                headers.update(header_params)

                http_method = tool_config.get("method", "GET").upper()

                async with httpx.AsyncClient(verify=False, timeout=30.0) as client:
                    if http_method == "POST":
                        if body_params:
                            response = await client.post(url, headers=headers, json=body_params, params=query_params)
                        else:
                            response = await client.post(url, headers=headers, json=query_params)
                    else:
                        response = await client.get(url, headers=headers, params=query_params)

                    if response.status_code in [200, 202]:
                        result = response.json()
                        cluster_result = {
                            "cluster_info": {
                                "name": cluster_name,
                                "host": cluster.get('host', 'localhost')
                            },
                            "data": result
                        }
                        all_results.append(cluster_result)
                    else:
                        raise Exception(f"API request failed: {response.status_code} - {response.text}")

            except Exception as e:
                error_result = {
                    "cluster_info": {"name": cluster_name},
                    "error": f"Error executing {name} on {cluster_name}: {str(e)}"
                }
                all_results.append(error_result)

        # Return results
        if len(target_clusters) == 1:
            final_result = all_results[0] if all_results else {"error": "No results"}
        else:
            final_result = {
                "network_wide_query": True,
                "total_clusters_queried": len(target_clusters),
                "results": all_results
            }

        return {'content': [{'type': 'text', 'text': json.dumps(final_result, indent=2)}]}

    async def run(self):
        """Run the enhanced MCP server"""
        logging.info("Starting enhanced MCP server...")
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="enhanced-declarative-catalyst",
                    server_version="2.1.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={}
                    )
                )
            )


if __name__ == "__main__":
    try:
        asyncio.run(EnhancedDeclarativeCatalystServer().run())
    except Exception as e:
        logging.error(f"Server crashed: {str(e)}")
        raise