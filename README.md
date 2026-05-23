## Servers

> Deprecation notice: `network_platforms_mcp_servers` is being deprecated.  
> Please use the new repository: `https://github.com/zapodeanu/platforms-mcp-servers-oss`.

This project is an open-source, published as is. It is not intended to be used in production.

## File Structure (Core + Transport Wrappers)

Legacy stdio shims (deprecated, still supported):

- `enhanced_declarative_catalyst.py` -> use `catalyst_center_stdio.py`
- `enhanced_declarative_meraki.py` -> use `meraki_stdio.py`

Current suite (preferred):

- `catalyst_center_core.py` / `meraki_core.py` - self-contained core logic
- `catalyst_center_stdio.py` / `meraki_stdio.py` - stdio wrappers
- `catalyst_center_remote.py` / `meraki_remote.py` - streamable HTTP wrappers

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
- **Features**: Three tools including a discovery tool, support for multiple orgs (Prod, Lab, or all), external org config via `Resources/meraki_organizations.yaml`
- **Use Case**: Shows how to scale MCP servers across multiple environments with per-org API keys and an "all" aggregation pattern

### Proof-of-Concept Servers
The following two proof-of-concept servers demonstrate auto-generation of MCP tools and management of multiple Catalyst Center clusters and multiple Meraki organizations.
### Meraki MCP Server (`meraki_stdio.py`)
- **Purpose**: Cisco Meraki cloud-managed network automation
- **Features**: Multi-organization support, device management, client tracking, network configuration
- **API Coverage**: Organizations, networks, devices, clients, security policies
- **API Explorer**: Use Meraki API specs file for Cosine search of APIs. Identify APIs, call the APIs, provide API docs and API telemetry. Restricted to only call GET API endpoints.

### Catalyst Center MCP Server (`catalyst_center_stdio.py`)  
- **Purpose**: Cisco Catalyst Center on-premises network automation
- **Features**: Device inventory, compliance checking, issue tracking, configuration management
- **API Coverage**: Sites, devices, clients, assurance data, operations
- **API Explorer**: Use Catalyst Center API specs file for Cosine search of APIs. Identify APIs, call the APIs, provide API docs and API telemetry. Restricted to only call GET API endpoints.

### Remote Streamable HTTP Servers
The following wrappers expose the same toolsets over streamable HTTP transport:

**Wrapper architecture note:** Remote servers in this repo are thin transport wrappers over the `_core` servers (`meraki_core.py` and `catalyst_center_core.py`).  
All tool logic (including API explorer safety rules and method restrictions) lives in `_core`, which keeps stdio and remote behavior in sync.  
Legacy shims are compatibility entrypoints only and route to the current stdio/core flow.

### Meraki Remote MCP Server (`meraki_remote.py`)
- **Purpose**: Remote HTTP wrapper for `meraki_core.py`
- **Transport**: Streamable HTTP on `/mcp` (default port `8001`)
- **API Explorer Safety**: Read-only explorer behavior inherited from declarative server (GET-only)
- **Sync Model**: Inherits declarative server logic (no duplicated execution engine)

### Catalyst Remote MCP Server (`catalyst_center_remote.py`)
- **Purpose**: Remote HTTP wrapper for `catalyst_center_core.py`
- **Transport**: Streamable HTTP on `/mcp` (default port `8000`)
- **API Explorer Safety**: Read-only explorer behavior inherited from declarative server (GET-only; no POST/PUT/DELETE)
- **Sync Model**: Inherits declarative server logic (no duplicated execution engine)

## Quick Start

1. **Environment Setup**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Credentials**

   Create `environment.env` with your API keys and endpoints

   If you use API explorer features, configure local sentence-transformer model access:
   - Set `SENTENCE_TRANSFORMERS_MODEL_DIR` to a local `all-MiniLM-L6-v2` folder, or
   - Place the model at `embeddings_cache/model/all-MiniLM-L6-v2` in this repo.

   You can populate the local model cache with:
   ```bash
   python download_model.py
   ```

3. **Run Servers**

   Validate the servers are running local. They can be started as subprocesses by an MCP client.
      ```bash
      python meraki_stdio.py
      python catalyst_center_stdio.py
      ```

4. **Run Remote Streamable HTTP Wrappers** (optional)

   Use these when your MCP client connects over HTTP instead of stdio:
   ```bash
   python meraki_remote.py --host 0.0.0.0 --port 8001
   python catalyst_center_remote.py --host 0.0.0.0 --port 8000
   ```

5. **Legacy Stdio Shims (Deprecated but Supported)**

   ```bash
   python enhanced_declarative_meraki.py
   python enhanced_declarative_catalyst.py
   ```

## Configuration

### Core Configuration Files

- `meraki_config.yaml` - Meraki API endpoints and parameters
- `catalyst_config.yaml` - Catalyst Center API endpoints and parameters
- `requirements.txt` - Python dependencies including MCP, API clients, ML libraries

**Location note:** Runtime organization/cluster config files are loaded from the `Resources/` directory:
- `Resources/meraki_organizations.yaml`
- `Resources/catalyst_center_clusters.yaml`

### API Explorer Swagger Files

- `Resources/meraki_swagger.json` - Meraki OpenAPI/Swagger source used by `meraki_core.py` explorer features
- `Resources/cc_swagger.json` - Catalyst Center OpenAPI/Swagger source used by `catalyst_center_core.py` explorer features

These files are used for endpoint discovery, similarity search, endpoint metadata, and explorer analytics workflows.

**Note:** API explorer auto-discovery and dynamic execution are intentionally restricted to `GET` endpoints only.  
`POST`/`PUT`/`DELETE` operations are not auto-generated by explorer tools and must be executed only through explicit declarative tools defined in YAML.

### Transformer Model Resource

- Local model cache base directory: `embeddings_cache/`
- Expected model path: `embeddings_cache/model/all-MiniLM-L6-v2`
- Model download helper: `download_model.py`

Example:
```bash
python download_model.py --model-name all-MiniLM-L6-v2
```

### Tool Catalog

- `Resources/MCP_Tools.md` - Markdown catalog of declarative and explorer tools for both Catalyst Center and Meraki servers

### Logging

- `logs/catalyst_center_mcp.log`
- `logs/meraki_mcp.log`

### Multi-Environment Setup

**`Resources/meraki_organizations.yaml`** - Configure multiple Meraki organizations:
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

**`Resources/catalyst_center_clusters.yaml`** - Configure multiple Catalyst Center clusters:
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

### Stdio

```json
{
  "mcpServers": {
    "meraki_stdio": {
      "command": "/path/to/your/venv/bin/python3",
      "args": ["/path/to/your/meraki_stdio.py"]
    },
    "catalyst_center_stdio": {
      "command": "/path/to/your/venv/bin/python3",
      "args": ["/path/to/your/catalyst_center_stdio.py"]
    }
  }
}
```

**Note**: Update the paths to match your actual Python virtual environment and script locations.

### Remote (Streamable HTTP)

Add to your Claude Desktop `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "meraki_remote": {
      "url": "http://127.0.0.1:8001/mcp"
    },
    "catalyst_center_remote": {
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

Start remote servers first:
```bash
python meraki_remote.py --host 0.0.0.0 --port 8001
python catalyst_center_remote.py --host 0.0.0.0 --port 8000
```

Each server provides declarative tools plus AI-powered API exploration for complete network automation capabilities.
