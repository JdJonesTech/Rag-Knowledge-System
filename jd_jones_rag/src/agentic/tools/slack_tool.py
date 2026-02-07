"""
Slack Tool
Integrates with Slack for internal communication and knowledge access.
Searches messages, channels, and enables posting notifications.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import os

from src.agentic.tools.base_tool import BaseTool, ToolResult


class SlackChannelType(str, Enum):
    """Types of Slack channels."""
    PUBLIC = "public_channel"
    PRIVATE = "private_channel"
    DM = "im"
    GROUP_DM = "mpim"


@dataclass
class SlackMessage:
    """Represents a Slack message."""
    ts: str  # Timestamp ID
    channel: str
    user: str
    text: str
    timestamp: datetime
    thread_ts: Optional[str] = None
    reactions: List[str] = None
    files: List[Dict[str, str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts": self.ts,
            "channel": self.channel,
            "user": self.user,
            "text": self.text,
            "timestamp": self.timestamp.isoformat(),
            "thread_ts": self.thread_ts,
            "reactions": self.reactions or [],
            "has_files": bool(self.files)
        }


class SlackTool(BaseTool):
    """
    Tool for Slack integration.
    
    Capabilities:
    - Search messages across channels
    - Read channel history
    - Post messages and notifications
    - Access user information
    - Search files shared in Slack
    """
    
    name = "slack_tool"
    description = """
    Accesses Slack for internal communications. Use for:
    - Searching past discussions on topics
    - Finding decisions made in channels
    - Looking up shared knowledge from conversations
    - Posting notifications to channels
    - Finding files shared in Slack
    """
    
    # Channel mappings for JD Jones
    CHANNEL_MAPPINGS = {
        "engineering": "C0123ENGR",
        "sales": "C0123SALE",
        "support": "C0123SUPP",
        "production": "C0123PROD",
        "quality": "C0123QUAL",
        "general": "C0123GENL",
        "announcements": "C0123ANNO"
    }
    
    def __init__(
        self,
        bot_token: Optional[str] = None,
        user_token: Optional[str] = None,
        workspace: Optional[str] = None
    ):
        """
        Initialize Slack tool.
        
        Args:
            bot_token: Slack bot token (xoxb-...)
            user_token: Slack user token (xoxp-...)
            workspace: Workspace identifier
        """
        super().__init__(
            name=self.name,
            description=self.description
        )
        self.bot_token = bot_token or os.getenv("SLACK_BOT_TOKEN")
        self.user_token = user_token or os.getenv("SLACK_USER_TOKEN")
        self.workspace = workspace or os.getenv("SLACK_WORKSPACE", "jdjones")
        
        self._client = None
    
    def _get_client(self):
        """Lazy initialization of Slack client."""
        if self._client is None:
            try:
                from slack_sdk import WebClient
                self._client = WebClient(token=self.bot_token)
            except ImportError:
                raise ImportError("slack_sdk not installed. Run: pip install slack_sdk")
        return self._client
    
    async def execute(
        self,
        query: str,
        parameters: Dict[str, Any],
        intent: Optional[str] = None
    ) -> ToolResult:
        """
        Execute a Slack action.
        
        Args:
            query: Search query or action description
            parameters: Action parameters including 'action' key
            intent: Optional intent from router
        
        Actions (specified in parameters['action']):
        - search: Search messages
        - get_history: Get channel history
        - post_message: Post a message
        - get_user: Get user info
        - search_files: Search files
        - list_channels: List available channels
        """
        try:
            # Get action from parameters or default to search
            action = parameters.get("action", "search")
            
            # Add query to parameters for search
            if "query" not in parameters:
                parameters["query"] = query
            
            if action == "search":
                return await self._search(parameters)
            elif action == "get_history":
                return await self._get_history(parameters)
            elif action == "post_message":
                return await self._post_message(parameters)
            elif action == "get_user":
                return await self._get_user(parameters)
            elif action == "search_files":
                return await self._search_files(parameters)
            elif action == "list_channels":
                return await self._list_channels(parameters)
            else:
                return ToolResult(
                    success=False,
                    data={},
                    error=f"Unknown action: {action}"
                )
        except Exception as e:
            return ToolResult(
                success=False,
                data={},
                error=str(e)
            )
    
    async def _search(self, params: Dict[str, Any]) -> ToolResult:
        """Search Slack messages."""
        query = params.get("query", "")
        channel = params.get("channel")
        from_user = params.get("from_user")
        days_back = params.get("days_back", 30)
        max_results = params.get("max_results", 20)
        
        if not query:
            return ToolResult(
                success=False,
                data={},
                error="Missing required field: query"
            )
        
        # Build search query
        search_query = query
        if channel:
            channel_id = self.CHANNEL_MAPPINGS.get(channel.lower(), channel)
            search_query += f" in:{channel_id}"
        if from_user:
            search_query += f" from:{from_user}"
        if days_back:
            search_query += f" after:{days_back}d"
        
        # Simulated search results
        results = [
            {
                "ts": "1704067200.000100",
                "channel": "engineering",
                "user": "john.doe",
                "text": f"Discussion about {query}: We decided to use the new API standard...",
                "timestamp": datetime.now().isoformat(),
                "permalink": f"https://{self.workspace}.slack.com/archives/C0123/p1704067200000100"
            },
            {
                "ts": "1703980800.000200",
                "channel": "quality",
                "user": "jane.smith",
                "text": f"Regarding {query}: The test results show improvement...",
                "timestamp": (datetime.now() - timedelta(days=1)).isoformat(),
                "permalink": f"https://{self.workspace}.slack.com/archives/C0124/p1703980800000200"
            }
        ]
        
        return ToolResult(
            success=True,
            data={
                "query": search_query,
                "total_results": len(results),
                "results": results[:max_results]
            },
            metadata={"action": "search", "days_back": days_back}
        )
    
    async def _get_history(self, params: Dict[str, Any]) -> ToolResult:
        """Get channel message history."""
        channel = params.get("channel")
        limit = params.get("limit", 50)
        oldest = params.get("oldest")  # Unix timestamp
        
        if not channel:
            return ToolResult(
                success=False,
                data={},
                error="Missing required field: channel"
            )
        
        channel_id = self.CHANNEL_MAPPINGS.get(channel.lower(), channel)
        
        # Simulated history
        messages = [
            {
                "ts": "1704153600.000100",
                "user": "john.doe",
                "text": "Good morning team! Here's the update on the project...",
                "timestamp": datetime.now().isoformat()
            },
            {
                "ts": "1704150000.000200",
                "user": "jane.smith",
                "text": "Thanks for the update. I'll review the documents.",
                "timestamp": (datetime.now() - timedelta(hours=1)).isoformat()
            }
        ]
        
        return ToolResult(
            success=True,
            data={
                "channel": channel,
                "channel_id": channel_id,
                "message_count": len(messages),
                "messages": messages[:limit]
            },
            metadata={"action": "get_history"}
        )
    
    async def _post_message(self, params: Dict[str, Any]) -> ToolResult:
        """Post a message to a channel."""
        channel = params.get("channel")
        text = params.get("text")
        thread_ts = params.get("thread_ts")  # For replies
        blocks = params.get("blocks")  # Rich formatting
        
        if not channel or not text:
            return ToolResult(
                success=False,
                data={},
                error="Missing required fields: channel, text"
            )
        
        channel_id = self.CHANNEL_MAPPINGS.get(channel.lower(), channel)
        
        # In production, would call Slack API
        message_ts = f"{datetime.now().timestamp():.6f}"
        
        return ToolResult(
            success=True,
            data={
                "channel": channel,
                "channel_id": channel_id,
                "message_ts": message_ts,
                "text_preview": text[:100] + "..." if len(text) > 100 else text,
                "posted": True,
                "permalink": f"https://{self.workspace}.slack.com/archives/{channel_id}/p{message_ts.replace('.', '')}"
            },
            metadata={"action": "post_message", "thread_ts": thread_ts}
        )
    
    async def _get_user(self, params: Dict[str, Any]) -> ToolResult:
        """Get user information."""
        user_id = params.get("user_id")
        email = params.get("email")
        
        if not user_id and not email:
            return ToolResult(
                success=False,
                data={},
                error="Missing required field: user_id or email"
            )
        
        # Simulated user info
        user = {
            "id": user_id or "U0123ABC",
            "name": "john.doe",
            "real_name": "John Doe",
            "email": email or "john.doe@jdjones.com",
            "title": "Senior Engineer",
            "department": "Engineering",
            "status": "active",
            "timezone": "America/New_York",
            "is_admin": False
        }
        
        return ToolResult(
            success=True,
            data=user,
            metadata={"action": "get_user"}
        )
    
    async def _search_files(self, params: Dict[str, Any]) -> ToolResult:
        """Search files shared in Slack."""
        query = params.get("query", "")
        file_type = params.get("file_type")  # pdf, doc, image, etc.
        channel = params.get("channel")
        max_results = params.get("max_results", 10)
        
        # Build search
        search_query = query
        if file_type:
            search_query += f" type:{file_type}"
        if channel:
            channel_id = self.CHANNEL_MAPPINGS.get(channel.lower(), channel)
            search_query += f" in:{channel_id}"
        
        # Simulated file results
        files = [
            {
                "id": "F0123ABC",
                "name": "Technical_Specification.pdf",
                "filetype": "pdf",
                "size_bytes": 245678,
                "user": "john.doe",
                "created": datetime.now().isoformat(),
                "permalink": f"https://{self.workspace}.slack.com/files/U0123/F0123ABC/Technical_Specification.pdf",
                "preview": "This document outlines the technical specifications..."
            }
        ]
        
        return ToolResult(
            success=True,
            data={
                "query": search_query,
                "total_results": len(files),
                "files": files[:max_results]
            },
            metadata={"action": "search_files", "file_type": file_type}
        )
    
    async def _list_channels(self, params: Dict[str, Any]) -> ToolResult:
        """List available channels."""
        include_private = params.get("include_private", False)
        
        channels = [
            {"id": "C0123ENGR", "name": "engineering", "is_private": False, "member_count": 45},
            {"id": "C0123SALE", "name": "sales", "is_private": False, "member_count": 32},
            {"id": "C0123SUPP", "name": "support", "is_private": False, "member_count": 28},
            {"id": "C0123PROD", "name": "production", "is_private": False, "member_count": 56},
            {"id": "C0123QUAL", "name": "quality", "is_private": False, "member_count": 18},
            {"id": "C0123GENL", "name": "general", "is_private": False, "member_count": 150},
        ]
        
        if not include_private:
            channels = [c for c in channels if not c["is_private"]]
        
        return ToolResult(
            success=True,
            data={
                "channel_count": len(channels),
                "channels": channels
            },
            metadata={"action": "list_channels", "include_private": include_private}
        )
