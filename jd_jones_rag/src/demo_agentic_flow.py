
import asyncio
import os
import sys
import json
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agentic.agents.react_agent import ReActAgent
from src.agentic.mcp_client import MCPClient
from src.agentic.tools.mcp_tool_adapter import MCPToolAdapter
from src.agentic.tools.web_search_tool import WebSearchTool

# Configure logging to show the "Thought" and "Action" loop clearly
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger("agent_demo")

async def run_agentic_demo():
    print("\n" + "="*60)
    print("JD JONES AGENTIC FLOW DEMO (ReAct + MCP + Web Search)")
    print("="*60 + "\n")

    # 1. Initialize MCP Core
    mcp_client = MCPClient()
    # Both demo_agentic_flow.py and mcp_server.py are in src/
    server_script = os.path.join(os.path.dirname(__file__), "mcp_server.py")
    
    print("Step 1: Connecting to Internal MCP Server...")
    # Using the container name 'mcp-server' which we defined in docker-compose.yml
    mcp_url = "http://mcp-server:8000/sse"
    print(f"   [Debug] Server URL: {mcp_url}")
    try:
        # Connect via SSE (much more stable in Docker)
        await mcp_client.connect_via_sse(
            "internal-tools",
            mcp_url
        )
        print("   [MCP] Connected to JD Jones Internal Tools server via SSE.")
    except Exception as e:
        print(f"   [Error] Could not connect to MCP server: {e}")
        return

    # 2. Discover and Adapt MCP Tools
    print("\nStep 2: Discovering and Adapting Tools via MCP...")
    mcp_tools_metadata = await mcp_client.list_tools("internal-tools")
    
    agent_tools = {}
    for tool_meta in mcp_tools_metadata:
        adapter = MCPToolAdapter(
            mcp_client=mcp_client,
            server_name="internal-tools",
            tool_name=tool_meta.name,
            description=tool_meta.description,
            input_schema=tool_meta.inputSchema
        )
        agent_tools[tool_meta.name] = adapter
        print(f"   [MCP Tool Registration] {tool_meta.name}")

    # 3. Add External Web Search Tool
    print("\nStep 3: Adding External Web Search Tool...")
    web_search = WebSearchTool()
    agent_tools[web_search.name] = web_search
    print(f"   [Native Tool Registration] {web_search.name}")

    # 4. Initialize ReAct Agent
    print("\nStep 4: Initializing ReAct Agent with unified toolkit...")
    agent = ReActAgent(tools=agent_tools, verbose=True)
    
    # 5. Execute a complex multi-step query
    query = (
        "We have an inquiry from Saudi Aramco. They are interested in 1000 units of PACMAAN-500. "
        "Can you verify if they are an existing customer, check if we have enough stock, "
        "calculate the total price with any discounts, and find any industry standards "
        "for fugitive emissions related to this product type?"
    )
    
    print("\n" + "-"*60)
    print(f"USER QUERY: {query}")
    print("-"*60 + "\n")

    try:
        # We run the agent. The agent will START the ReAct loop:
        # Thought -> Action -> Observation -> Repeat
        result = await agent.execute(query)
        
        print("\n" + "="*60)
        print("FINAL AGENT RESPONSE")
        print("="*60)
        print(result.final_answer)
        
        print("\nAGENT TRACE SUMMARY:")
        print(f"   - Iterations: {result.iterations}")
        print(f"   - Tools Used: {', '.join(result.tools_used)}")
        print(f"   - Time taken: {result.total_time_ms / 1000:.2f}s")
        
    except Exception as e:
        print(f"   [Error during execution] {e}")
    finally:
        await mcp_client.cleanup()

if __name__ == "__main__":
    # Ensure dependencies are available
    # NOTE: In a real system, you'd have your OPENAI_API_KEY set.
    # For this demo, we assume the environment is properly configured.
    asyncio.run(run_agentic_demo())
