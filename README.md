## Servers

This project is an open-source, published as is. It is not intended to be used in production.

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

## Quick Start

1. **Environment Setup**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Credentials**

   Create `environment.env` with your API keys and endpoints

3. **Run Servers**

   Validate the servers are running local. They will be started as subprocesses by the MCP Client
      ```bash
      python enhanced_declarative_meraki.py
      python enhanced_declarative_catalyst.py
      ```

## Configuration

### Core Configuration Files

- `meraki_config.yaml` - Meraki API endpoints and parameters
- `catalyst_config.yaml` - Catalyst Center API endpoints and parameters
- `requirements.txt` - Python dependencies including MCP, API clients, ML libraries

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
