## Servers

This project is an open-source, published as is. It is not intended to be used in production.

### Learning Progression Servers
The following three servers demonstrate MCP development from simple to multi-organization patterns:
### Basic Meraki MCP Server (`basic_meraki_mcp_server.py`)
- **Purpose**: Minimal single-tool MCP server example for learning MCP fundamentals
- **Features**: Single tool (`list_organizations_details`) with one API key, no parameters required
- **Use Case**: Starting point for understanding MCP server structure, tool registration, and API integration

### Foundation Meraki MCP Server (`foundation_meraki_mcp_server.py`)
- **Purpose**: Introductory multi-tool MCP server with basic Meraki API coverage
- **Features**: Two tools (organizations and networks), single API key, parameter handling with required inputs
- **Use Case**: Demonstrates tool chaining — use organization details to discover org IDs, then pass them to list networks

### Multi-Org Meraki MCP Server (`multiorg_meraki_mcp_server.py`)
- **Purpose**: Multi-organization Meraki MCP server with YAML-driven org configuration
- **Features**: Three tools including a discovery tool, support for multiple orgs (Prod, Lab, or all), external org config via `meraki_organizations.yaml`
- **Use Case**: Shows how to scale MCP servers across multiple environments with per-org API keys and an "all" aggregation pattern

### Proof-of-Concept Servers
The following two proof-of-concept servers demonstrate auto-generation of MCP tools and management of multiple Catalyst Center clusters and multiple Meraki organizations.
### Meraki MCP Server (`enhanced_declarative_meraki.py`)
- **Purpose**: Cisco Meraki cloud-managed network automation
- **Features**: Multi-organization support, device management, client tracking, network configuration
- **API Coverage**: Organizations, networks, devices, clients, security policies
- **API Explorer**: Use Meraki API specs file for Cosine search of APIs. Identify APIs, call the APIs, provide API docs and API telemetry. Restricted to only call GET API endpoints.

### Catalyst Center MCP Server (`enhanced_declarative_catalyst.py`)  
- **Purpose**: Cisco Catalyst Center on-premises network automation
- **Features**: Device inventory, compliance checking, issue tracking, configuration management
- **API Coverage**: Sites, devices, clients, assurance data, operations
- **API Explorer**: Use Catalyst Center API specs file for Cosine search of APIs. Identify APIs, call the APIs, provide API docs and API telemetry. Restricted to only call GET API endpoints.

### Remote Streamable HTTP Servers
The following wrappers expose the same toolsets over streamable HTTP transport:

### Meraki Remote MCP Server (`enhanced_meraki_remote.py`)
- **Purpose**: Remote HTTP wrapper for `enhanced_declarative_meraki.py`
- **Transport**: Streamable HTTP on `/mcp` (default port `8001`)
- **API Explorer Safety**: Read-only explorer behavior inherited from declarative server (GET-only)

### Catalyst Remote MCP Server (`enhanced_catalyst_remote.py`)
- **Purpose**: Remote HTTP wrapper for `enhanced_declarative_catalyst.py`
- **Transport**: Streamable HTTP on `/mcp` (default port `8000`)
- **API Explorer Safety**: Read-only explorer behavior inherited from declarative server (GET-only; no POST/PUT/DELETE)

## Quick Start

1. **Environment Setup**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Credentials**

   Create `environment.env` with your API keys and endpoints

   If you use API explorer features, also configure local sentence-transformer model access:
   - Set `SENTENCE_TRANSFORMERS_MODEL_DIR` to a local `all-MiniLM-L6-v2` folder, or
   - Place the model at `embeddings_cache/model/all-MiniLM-L6-v2` in this repo.

3. **Run Servers**

   Validate the servers are running local. They can be started as subprocesses by an MCP client.
      ```bash
      python enhanced_declarative_meraki.py
      python enhanced_declarative_catalyst.py
      ```

4. **Run Remote Streamable HTTP Wrappers** (optional)

   Use these when your MCP client connects over HTTP instead of stdio:
   ```bash
   python enhanced_meraki_remote.py --host 0.0.0.0 --port 8001
   python enhanced_catalyst_remote.py --host 0.0.0.0 --port 8000
   ```

## Configuration

### Core Configuration Files

- `meraki_config.yaml` - Meraki API endpoints and parameters
- `catalyst_config.yaml` - Catalyst Center API endpoints and parameters
- `requirements.txt` - Python dependencies including MCP, API clients, ML libraries

### API Explorer Swagger Files

- `Resources/meraki_swagger.json` - Meraki OpenAPI/Swagger source used by `enhanced_declarative_meraki.py` explorer features
- `Resources/cc_swagger.json` - Catalyst Center OpenAPI/Swagger source used by `enhanced_declarative_catalyst.py` explorer features

These files are used for endpoint discovery, similarity search, endpoint metadata, and explorer analytics workflows.

### Multi-Environment Setup

**`meraki_organizations.yaml`** - Configure multiple Meraki organizations:
```yaml
meraki_organizations:
  - name: "Production"
    api_key_env: "MERAKI_PROD_API_KEY"
    description: "Production environment networks"
    enabled: true
  - name: "Lab"
    api_key_env: "MERAKI_LAB_API_KEY"
    description: "Development and testing networks"
    enabled: true
```

**`catalyst_center_clusters.yaml`** - Configure multiple Catalyst Center clusters:
```yaml
catalyst_centers:
  - name: "Portland"
    host: "Portland-center.domain.com"
    version: "2.3.7.10"
    location: "Portland"
    enabled: true
  - name: "San Jose"
    host: "SanJose-catalyst.domain.com"
    version: "2.3.7.9" 
    location: "San Jose"
    enabled: false
```

### Environment Variables

Create `environment.env` file:
```bash
# Meraki API Keys
MERAKI_PROD_API_KEY=your_production_api_key
MERAKI_LAB_API_KEY=your_lab_api_key

# Catalyst Center Credentials
CC_URL=https://your-catalyst-center.domain.com
CC_USER=your_username
CC_PASS=your_password
```

## Dependencies

Key packages from `requirements.txt`:
- `mcp` - Model Context Protocol server framework
- `httpx` - Modern HTTP client for API calls  
- `sentence-transformers` - AI embeddings for API exploration
- `catalystcentersdk` - Cisco Catalyst Center Python SDK
- `scikit-learn`, `numpy` - ML libraries for cosine similarity search

## Claude Desktop Configuration

Add to your Claude Desktop `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "enhanced_declarative_meraki": {
      "command": "/path/to/your/venv/bin/python3",
      "args": ["/path/to/your/enhanced_declarative_meraki.py"]
    },
    "enhanced_declarative_catalyst": {
      "command": "/path/to/your/venv/bin/python3", 
      "args": ["/path/to/your/enhanced_declarative_catalyst.py"]
    }
  }
}
```

**Note**: Update the paths to match your actual Python virtual environment and script locations.

Each server provides declarative tools plus AI-powered API exploration for complete network automation capabilities.
