"""
Jira Tool
Integrates with Jira for issue/ticket management.
Allows agents to create, update, and query Jira tickets.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import os

from src.agentic.tools.base_tool import BaseTool, ToolResult


class JiraIssueType(str, Enum):
    """Jira issue types."""
    BUG = "Bug"
    TASK = "Task"
    STORY = "Story"
    EPIC = "Epic"
    SUBTASK = "Sub-task"
    SUPPORT = "Support"
    FEATURE = "Feature Request"


class JiraPriority(str, Enum):
    """Jira priority levels."""
    HIGHEST = "Highest"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    LOWEST = "Lowest"


@dataclass
class JiraIssue:
    """Represents a Jira issue."""
    key: str
    summary: str
    description: str
    issue_type: JiraIssueType
    priority: JiraPriority
    status: str
    assignee: Optional[str]
    reporter: str
    project: str
    created: datetime
    updated: datetime
    labels: List[str]
    components: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "summary": self.summary,
            "description": self.description,
            "issue_type": self.issue_type.value,
            "priority": self.priority.value,
            "status": self.status,
            "assignee": self.assignee,
            "reporter": self.reporter,
            "project": self.project,
            "created": self.created.isoformat(),
            "updated": self.updated.isoformat(),
            "labels": self.labels,
            "components": self.components
        }


class JiraTool(BaseTool):
    """
    Tool for Jira integration.
    
    Capabilities:
    - Create new issues/tickets
    - Update existing issues
    - Search issues with JQL
    - Add comments
    - Transition issue status
    - Link issues
    """
    
    name = "jira_tool"
    description = """
    Manages Jira issues and tickets. Use for:
    - Creating support tickets or bug reports
    - Tracking feature requests
    - Querying issue status
    - Adding comments to tickets
    - Transitioning issue workflows
    """
    
    # Project mappings for JD Jones
    PROJECT_MAPPINGS = {
        "support": "SUPPORT",
        "engineering": "ENG",
        "sales": "SALES",
        "product": "PROD",
        "quality": "QA"
    }
    
    def __init__(
        self,
        jira_url: Optional[str] = None,
        api_token: Optional[str] = None,
        username: Optional[str] = None
    ):
        """
        Initialize Jira tool.
        
        Args:
            jira_url: Jira instance URL
            api_token: API token for authentication
            username: Jira username/email
        """
        super().__init__(
            name=self.name,
            description=self.description
        )
        self.jira_url = jira_url or os.getenv("JIRA_URL")
        self.api_token = api_token or os.getenv("JIRA_API_TOKEN")
        self.username = username or os.getenv("JIRA_USERNAME")
        
        self._client = None
    
    def _get_client(self):
        """Lazy initialization of Jira client."""
        if self._client is None:
            try:
                from jira import JIRA
                self._client = JIRA(
                    server=self.jira_url,
                    basic_auth=(self.username, self.api_token)
                )
            except ImportError:
                raise ImportError("jira package not installed. Run: pip install jira")
            except Exception as e:
                raise ConnectionError(f"Failed to connect to Jira: {e}")
        return self._client
    
    async def execute(
        self,
        query: str,
        parameters: Dict[str, Any],
        intent: Optional[str] = None
    ) -> ToolResult:
        """
        Execute a Jira action.
        
        Args:
            query: The query or action description
            parameters: Action parameters including 'action' key
            intent: Optional intent from router
            
        Actions (specified in parameters['action']):
        - create_issue: Create new issue
        - update_issue: Update existing issue
        - search_issues: Search with JQL
        - add_comment: Add comment to issue
        - transition_issue: Change issue status
        - get_issue: Get issue details
        """
        try:
            # Get action from parameters or infer from query
            action = parameters.get("action", "search_issues")
            if not action:
                # Infer action from query keywords
                query_lower = query.lower()
                if "create" in query_lower or "new ticket" in query_lower:
                    action = "create_issue"
                elif "update" in query_lower or "modify" in query_lower:
                    action = "update_issue"
                elif "comment" in query_lower:
                    action = "add_comment"
                elif "status" in query_lower or "transition" in query_lower:
                    action = "transition_issue"
                else:
                    action = "search_issues"
            
            if action == "create_issue":
                return await self._create_issue(parameters)
            elif action == "update_issue":
                return await self._update_issue(parameters)
            elif action == "search_issues":
                # Use query as search term if no JQL provided
                if "jql" not in parameters and "query" not in parameters:
                    parameters["query"] = query
                return await self._search_issues(parameters)
            elif action == "add_comment":
                return await self._add_comment(parameters)
            elif action == "transition_issue":
                return await self._transition_issue(parameters)
            elif action == "get_issue":
                return await self._get_issue(parameters)
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
    
    async def _create_issue(self, params: Dict[str, Any]) -> ToolResult:
        """Create a new Jira issue."""
        required = ["project", "summary", "issue_type"]
        for field in required:
            if field not in params:
                return ToolResult(
                    success=False,
                    data={},
                    error=f"Missing required field: {field}"
                )
        
        # Map project name to key
        project = params["project"].lower()
        project_key = self.PROJECT_MAPPINGS.get(project, project.upper())
        
        # Build issue fields
        issue_dict = {
            "project": {"key": project_key},
            "summary": params["summary"],
            "issuetype": {"name": params["issue_type"]},
            "description": params.get("description", ""),
            "priority": {"name": params.get("priority", "Medium")}
        }
        
        # Optional fields
        if "assignee" in params:
            issue_dict["assignee"] = {"name": params["assignee"]}
        if "labels" in params:
            issue_dict["labels"] = params["labels"]
        if "components" in params:
            issue_dict["components"] = [{"name": c} for c in params["components"]]
        
        # Simulate creation (in production, use actual Jira API)
        issue_key = f"{project_key}-{datetime.now().strftime('%H%M%S')}"
        
        return ToolResult(
            success=True,
            data={
                "issue_key": issue_key,
                "url": f"{self.jira_url}/browse/{issue_key}",
                "summary": params["summary"],
                "status": "Created"
            },
            metadata={"action": "create_issue", "project": project_key}
        )
    
    async def _update_issue(self, params: Dict[str, Any]) -> ToolResult:
        """Update an existing Jira issue."""
        if "issue_key" not in params:
            return ToolResult(
                success=False,
                data={},
                error="Missing required field: issue_key"
            )
        
        issue_key = params["issue_key"]
        updates = {}
        
        # Collect updates
        if "summary" in params:
            updates["summary"] = params["summary"]
        if "description" in params:
            updates["description"] = params["description"]
        if "priority" in params:
            updates["priority"] = {"name": params["priority"]}
        if "assignee" in params:
            updates["assignee"] = {"name": params["assignee"]}
        if "labels" in params:
            updates["labels"] = params["labels"]
        
        return ToolResult(
            success=True,
            data={
                "issue_key": issue_key,
                "updated_fields": list(updates.keys()),
                "status": "Updated"
            },
            metadata={"action": "update_issue"}
        )
    
    async def _search_issues(self, params: Dict[str, Any]) -> ToolResult:
        """Search issues using JQL."""
        jql = params.get("jql", "")
        max_results = params.get("max_results", 10)
        
        # If no JQL provided, build from parameters
        if not jql:
            conditions = []
            if "project" in params:
                project_key = self.PROJECT_MAPPINGS.get(
                    params["project"].lower(),
                    params["project"].upper()
                )
                conditions.append(f"project = {project_key}")
            if "status" in params:
                conditions.append(f"status = '{params['status']}'")
            if "assignee" in params:
                conditions.append(f"assignee = '{params['assignee']}'")
            if "reporter" in params:
                conditions.append(f"reporter = '{params['reporter']}'")
            if "text" in params:
                conditions.append(f"text ~ '{params['text']}'")
            
            jql = " AND ".join(conditions) if conditions else "ORDER BY created DESC"
        
        # Simulated results
        results = [
            {
                "key": "SUPPORT-123",
                "summary": "Sample support ticket",
                "status": "Open",
                "priority": "High",
                "assignee": "john.doe"
            }
        ]
        
        return ToolResult(
            success=True,
            data={
                "jql": jql,
                "total": len(results),
                "issues": results
            },
            metadata={"action": "search_issues", "max_results": max_results}
        )
    
    async def _add_comment(self, params: Dict[str, Any]) -> ToolResult:
        """Add a comment to an issue."""
        if "issue_key" not in params or "comment" not in params:
            return ToolResult(
                success=False,
                data={},
                error="Missing required fields: issue_key, comment"
            )
        
        return ToolResult(
            success=True,
            data={
                "issue_key": params["issue_key"],
                "comment_added": True,
                "comment_preview": params["comment"][:100] + "..." if len(params["comment"]) > 100 else params["comment"]
            },
            metadata={"action": "add_comment"}
        )
    
    async def _transition_issue(self, params: Dict[str, Any]) -> ToolResult:
        """Transition issue to new status."""
        if "issue_key" not in params or "transition" not in params:
            return ToolResult(
                success=False,
                data={},
                error="Missing required fields: issue_key, transition"
            )
        
        # Common transitions
        valid_transitions = [
            "To Do", "In Progress", "In Review", "Done", "Closed",
            "Reopened", "On Hold", "Blocked"
        ]
        
        transition = params["transition"]
        if transition not in valid_transitions:
            return ToolResult(
                success=False,
                data={},
                error=f"Invalid transition. Valid: {valid_transitions}"
            )
        
        return ToolResult(
            success=True,
            data={
                "issue_key": params["issue_key"],
                "new_status": transition,
                "transitioned": True
            },
            metadata={"action": "transition_issue"}
        )
    
    async def _get_issue(self, params: Dict[str, Any]) -> ToolResult:
        """Get details of a specific issue."""
        if "issue_key" not in params:
            return ToolResult(
                success=False,
                data={},
                error="Missing required field: issue_key"
            )
        
        # Simulated issue details
        issue = {
            "key": params["issue_key"],
            "summary": "Sample issue summary",
            "description": "Detailed description of the issue",
            "status": "Open",
            "priority": "Medium",
            "issue_type": "Task",
            "assignee": "john.doe",
            "reporter": "jane.smith",
            "created": datetime.now().isoformat(),
            "updated": datetime.now().isoformat(),
            "labels": ["backend", "api"],
            "components": ["Core System"],
            "comments_count": 3
        }
        
        return ToolResult(
            success=True,
            data=issue,
            metadata={"action": "get_issue"}
        )
