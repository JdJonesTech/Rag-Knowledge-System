
import asyncio
import sys

# Redirect all standard output to stderr to prevent JSON-RPC protocol corruption
# FastMCP handles the actual stdio communication using lower-level streams.
sys.stdout = sys.stderr

from typing import Dict, Any, List, Optional
from mcp.server.fastmcp import FastMCP

# Import existing tools
from src.agentic.tools.vector_search_tool import VectorSearchTool
from src.agentic.tools.sql_query_tool import SQLQueryTool
from src.agentic.tools.crm_tool import CRMTool

# Initialize FastMCP server
mcp = FastMCP("JD Jones Agentic Systems")

# Initialize real tools
vector_search = VectorSearchTool()
erp_query = SQLQueryTool()
crm = CRMTool()

@mcp.tool()
async def lookup_customer(company: str = None, email: str = None) -> Dict[str, Any]:
    """
    Look up customer information in the CRM.
    
    Args:
        company: Company name to search for
        email: Contact email to search for
    """
    result = await crm(query=f"Lookup {company or email}", parameters={
        "action": "lookup",
        "company": company,
        "email": email
    })
    
    if not result.success:
        return {"error": result.error}
        
    return result.data

@mcp.tool()
async def search_knowledge_base(query: str, user_role: str = "employee", limit: int = 5) -> str:
    """
    Search product specifications, technical docs, and safety info.
    
    Args:
        query: Semantic search query
        user_role: User role for access control (employee, admin, engineer)
        limit: Max results to return
    """
    result = await vector_search(query=query, parameters={"user_role": user_role, "limit": limit})
    if not result.success:
        return f"Error: {result.error}"
    
    # Format results for the agent
    output = []
    for r in result.data.get("results", []):
        output.append(f"Source: {r['source']}\nContent: {r['content'][:500]}...")
    
    return "\n\n---\n\n".join(output) if output else "No results found."

@mcp.tool()
async def check_inventory(product_ids: List[str]) -> Dict[str, Any]:
    """
    Check stock levels and pricing for specific JD Jones products.
    
    Args:
        product_ids: List of product IDs (e.g., ['PACMAAN-500', 'FLEXSEAL-API'])
    """
    result = await erp_query(query=f"Check stock for {product_ids}", parameters={
        "query_type": "stock",
        "product_ids": product_ids
    })
    
    if not result.success:
        return {"error": result.error}
        
    return result.data

@mcp.tool()
async def get_order_status(order_id: str) -> Dict[str, Any]:
    """
    Check the status of a customer order.
    
    Args:
        order_id: Order number (e.g., 'ORD-2024-1234')
    """
    result = await erp_query(query=f"Status of {order_id}", parameters={
        "query_type": "order",
        "order_id": order_id
    })
    
    if not result.success:
        return {"error": result.error}
        
    return result.data

@mcp.tool()
async def health_check() -> Dict[str, Any]:
    """
    Health check for MCP server.
    
    Returns server status and tool availability.
    """
    return {
        "status": "healthy",
        "server": "JD Jones Agentic Systems",
        "tools_available": [
            "lookup_customer",
            "search_knowledge_base", 
            "check_inventory",
            "get_order_status"
        ],
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import sys
    import os
    
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    port = int(os.getenv("MCP_PORT", "8000"))
    
    if transport == "sse":
        # Use uvicorn with custom health endpoint
        from starlette.applications import Starlette
        from starlette.routing import Route, Mount
        from starlette.responses import JSONResponse
        import uvicorn
        
        async def health_endpoint(request):
            """HTTP health endpoint for Docker healthcheck."""
            return JSONResponse({
                "status": "healthy",
                "service": "mcp-server",
                "transport": "sse"
            })
        
        # Get the underlying ASGI app from FastMCP (newer API)
        # Try http_app() first, fall back to sse_app() for older versions
        try:
            mcp_app = mcp.http_app(transport="sse")
        except (AttributeError, TypeError):
            try:
                mcp_app = mcp.sse_app()
            except AttributeError:
                # Fallback: use the mcp object directly as ASGI app if available
                mcp_app = mcp
        
        # Create wrapper app with health endpoint
        async def app(scope, receive, send):
            if scope["type"] == "http" and scope["path"] == "/health":
                response = JSONResponse({
                    "status": "healthy",
                    "service": "mcp-server"
                })
                await response(scope, receive, send)
            else:
                await mcp_app(scope, receive, send)
        
        print(f"Starting MCP SSE Server with health endpoint on port {port}...", file=sys.stderr)
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        print("Starting MCP stdio transport...", file=sys.stderr)
        mcp.run(transport="stdio")

