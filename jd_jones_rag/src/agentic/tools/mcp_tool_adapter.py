
from typing import Dict, Any, Optional
import json
import logging

from src.agentic.tools.base_tool import BaseTool, ToolResult
from src.agentic.mcp_client import MCPClient

logger = logging.getLogger(__name__)

class MCPToolAdapter(BaseTool):
    """
    Adapter to expose an MCP tool as a BaseTool for the ReAct Agent.
    """
    
    def __init__(
        self, 
        mcp_client: MCPClient,
        server_name: str,
        tool_name: str,
        description: str,
        input_schema: Dict[str, Any]
    ):
        super().__init__(name=tool_name, description=description)
        self.mcp_client = mcp_client
        self.server_name = server_name
        self._schema = input_schema
        
    def get_schema(self) -> Dict[str, Any]:
        return self._schema
        
    async def execute(
        self, 
        query: str, 
        parameters: Dict[str, Any],
        intent: Optional[str] = None
    ) -> ToolResult:
        try:
            logger.info(f"Calling MCP tool {self.name} on server {self.server_name}")
            result = await self.mcp_client.call_tool(
                self.server_name, 
                self.name, 
                parameters
            )
            
            # Parse MCP result
            output_text = ""
            if hasattr(result, 'content'):
                for content in result.content:
                    if hasattr(content, 'text'):
                        output_text += content.text + "\n"
            else:
                output_text = str(result)
                
            from src.agentic.tools.base_tool import ToolStatus
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.SUCCESS,
                data={"output": output_text.strip()}
            )
            
        except Exception as e:
            logger.error(f"Error calling MCP tool {self.name}: {e}")
            from src.agentic.tools.base_tool import ToolStatus
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error=str(e)
            )
