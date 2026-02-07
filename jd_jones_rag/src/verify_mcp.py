
import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agentic.mcp_client import MCPClient
from src.agentic.tools.mcp_tool_adapter import MCPToolAdapter

async def verify_mcp_integration():
    print("1. Initializing MCP Client...")
    # configuration to run our own mcp_server.py
    # Since this script and mcp_server.py are both in src/, we can find it directly
    server_script = os.path.join(os.path.dirname(__file__), "mcp_server.py")
    
    # We will run the server via stdio for this test
    # This requires 'uv' or just python if dependencies are installed.
    # Assuming python is available and has mcp installed.
    
    client = MCPClient()
    
    print(f"2. Connecting to internal MCP server at {server_script}...")
    try:
        session = await client.connect_to_server(
            "internal-tools",
            "python", 
            [server_script],
            env=os.environ.copy()
        )
        print("   Success: Connected.")
        
        print("3. Listing available tools...")
        tools = await client.list_tools("internal-tools")
        print(f"   Found {len(tools)} tools:")
        for tool in tools:
            print(f"   - {tool.name}: {tool.description}")
            
        print("4. Testing tool execution (Calculator)...")
        # Direct tool call via client, NO LLM involvement
        result = await client.call_tool(
            "internal-tools", 
            "calculator", 
            {"a": 10, "b": 5, "operation": "multiply"}
        )
        print(f"   Result: {result}")
        
        print("\nIntegration Verified: MCP Client <-> MCP Server connection working.")
        
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(verify_mcp_integration())
