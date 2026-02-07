
import asyncio
import logging
from typing import Dict, Any, List, Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

class MCPClient:
    """
    Client for interacting with Model Context Protocol (MCP) servers.
    """
    
    def __init__(self, servers: Dict[str, Dict[str, Any]] = None):
        """
        Initialize MCP Client.
        
        Args:
            servers: Dictionary mapping server names to their configuration parameters
                     Format: {"server_name": {"command": "...", "args": [...], "env": {...}}}
        """
        self.server_configs = servers or {}
        self.sessions: Dict[str, ClientSession] = {}
        self.exit_stack = AsyncExitStack()
        
    async def connect_to_server(self, server_name: str, command: str, args: List[str], env: Optional[Dict[str, str]] = None) -> ClientSession:
        """Connect to a specific MCP server via stdio."""
        server_params = StdioServerParameters(command=command, args=args, env=env)
        try:
            read_stream, write_stream = await self.exit_stack.enter_async_context(stdio_client(server_params))
            session = await self.exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()
            self.sessions[server_name] = session
            return session
        except Exception as e:
            logger.error(f"Failed to connect to MCP server {server_name}: {e}")
            raise

    async def connect_via_sse(self, server_name: str, url: str) -> ClientSession:
        """Connect to an MCP server via SSE (HTTP)."""
        from mcp.client.sse import sse_client
        try:
            read_stream, write_stream = await self.exit_stack.enter_async_context(sse_client(url))
            session = await self.exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()
            self.sessions[server_name] = session
            logger.info(f"Connected to MCP server via SSE: {server_name} at {url}")
            return session
        except Exception as e:
            logger.error(f"Failed to connect to MCP server via SSE {server_name}: {e}")
            raise

    async def list_tools(self, server_name: str) -> List[Any]:
        """List available tools on a connected server."""
        if server_name not in self.sessions:
            raise ValueError(f"Not connected to server: {server_name}")
            
        session = self.sessions[server_name]
        result = await session.list_tools()
        return result.tools

    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on a connected server."""
        if server_name not in self.sessions:
            raise ValueError(f"Not connected to server: {server_name}")
            
        session = self.sessions[server_name]
        result = await session.call_tool(tool_name, arguments)
        return result

    async def cleanup(self):
        """Close all connections."""
        await self.exit_stack.aclose()
        self.sessions.clear()
