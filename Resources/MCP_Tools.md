Here is the complete catalog of all available tools across both platforms:

🛠️ Complete Tool Catalog
🔵 Cisco Catalyst Center Tools (17 Tools)
📊 Inventory Tools
#	Tool	Description
1	get_site_count	Get total count of sites across the network
2	get_sites	Get detailed site info (areas, buildings, floors)
3	get_site_topology	Get network topology for sites
4	get_network_devices_count	Get total count of network devices (filterable by family, role, reachability)
5	get_network_devices	Get detailed network device info (IPs, models, software versions)
6	get_compliance_detail	Get device compliance status (IMAGE, PSIRT, EOX, NETWORK_PROFILE, etc.)
🚨 Assurance Tools
#	Tool	Description
7	get_issues_count	Get total count of issues (filterable by priority, status, site, device, client)
8	get_issues	Get detailed issue list with suggested actions
9	get_issue_details	Get full details + suggested actions for a specific issue ID
10	get_clients_count	Get total count of network clients
11	get_clients	Get detailed client analytics and health data
12	get_interfaces_count	Get total count of network interfaces
13	get_device_interfaces	Get interface statistics and analytics for a device
⚙️ Operations Tools
#	Tool	Description
14	run_read_only_commands	Execute read-only CLI commands on network devices
15	get_task_status	Get status of a specific task by ID
16	get_task_detail	Get detailed task information and results
17	download_file	Download files by ID (command outputs, reports)
18	execute_issue_suggested_actions	Execute Catalyst Center suggested actions for an issue
19	get_business_api_execution_results	Get execution details of a Business API by execution ID
20	get_config_changes	Identify recent configuration changes (predefined diagnostic commands — auto-executed)
🔧 Administration Tools
#	Tool	Description
21	get_clusters	Get information about available Catalyst Center clusters
22	help	List correct usage patterns and AI flow guidance
🔍 Explorer / Dynamic API Tools
#	Tool	Description
23	explore_catalyst_api_endpoints	Search any Catalyst Center API endpoint using natural language
24	execute_catalyst_api_endpoint	Execute any Catalyst Center API endpoint dynamically (GET/POST/PUT/DELETE/PATCH)
25	get_catalyst_endpoint_info	Get detailed info (params, usage) about a specific API endpoint
26	get_catalyst_explorer_analytics	View usage patterns and success rates for explored API endpoints
🟢 Cisco Meraki Tools (16 Tools)
🏢 Organization Tools
#	Tool	Description
1	get_configured_organizations	List YOUR configured orgs from config (Prod, Lab). Start here!
2	list_organizations_details	Call Meraki API to get org IDs, names, URLs
🌐 Network Tools
#	Tool	Description
3	list_networks	List all networks in a Meraki organization
4	get_network	Get detailed information about a specific network
📟 Device Tools
#	Tool	Description
5	list_devices	List all devices in a network
👥 Client Tools
#	Tool	Description
6	get_network_clients	List all clients in a network (filterable by IP, MAC, OS, VLAN, status, connection type)
7	get_device_clients	List clients connected to a specific device
8	get_client_details	Get detailed information about a specific client
9	search_client_by_mac	Search for a client by MAC address across all networks in an org
🚨 Alerts & Health Tools
#	Tool	Description
10	get_organization_health_alerts	Get comprehensive health alerts for a Meraki org (filterable by severity, type, network, device)
11	get_organization_alerts_overview	Get summary alert counts by severity and network
12	get_alert_details	Get detailed information about a specific alert by ID
🔍 Explorer / Dynamic API Tools
#	Tool	Description
13	explore_meraki_api_endpoints	Search any Meraki API endpoint using natural language
14	execute_meraki_api_endpoint	Execute any Meraki API endpoint dynamically
15	get_meraki_endpoint_info	Get detailed info about a specific Meraki API endpoint
16	get_meraki_explorer_analytics	View usage patterns and success rates for explored Meraki API endpoints
🔑 Key Usage Notes
Catalyst Center
sql

Copy code
✅ Always use cluster='all' for network-wide queries
✅ Use get_clusters first to discover available clusters
✅ get_config_changes — just pass deviceUuids, commands are auto-selected
✅ run_read_only_commands — you provide custom commands + deviceUuids
✅ Use explore → get_info → execute workflow for any API not covered above

Meraki
sql

Copy code
✅ ALWAYS call get_configured_organizations FIRST
✅ ALWAYS pass organization='Prod' or organization='Lab' to every tool
✅ Network IDs (L_xxx) are org-specific — don't mix between orgs
✅ Use explore → get_info → execute workflow for unlisted API features

🤖 AI Troubleshooting Flow (Catalyst Center)
markdown

Copy code
1. get_issues_count       → How many active issues?
2. get_issues             → What are they?
3. get_site_topology      → Understand the topology
4. get_issue_details      → Deep dive on critical issues
5. execute_suggested_actions → Run automated remediation
6. get_config_changes     → Check for recent config drift
7. run_read_only_commands → Baseline + Config + Routing + Protocol checks

Total: 26 Catalyst Center tools + 16 Meraki tools = 42 tools available 🚀