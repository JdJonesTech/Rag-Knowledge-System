"""
Agentic Tools Module
Provides tools that agents can use for multi-source retrieval and actions.
"""

from src.agentic.tools.base_tool import BaseTool, ToolResult
from src.agentic.tools.vector_search_tool import VectorSearchTool
from src.agentic.tools.sql_query_tool import SQLQueryTool
from src.agentic.tools.api_tool import ExternalAPITool
from src.agentic.tools.email_tool import EmailTool
from src.agentic.tools.crm_tool import CRMTool
from src.agentic.tools.document_generator_tool import DocumentGeneratorTool
from src.agentic.tools.compliance_checker_tool import ComplianceCheckerTool
from src.agentic.tools.jira_tool import JiraTool
from src.agentic.tools.sharepoint_tool import SharePointTool
from src.agentic.tools.web_search_tool import WebSearchTool
from src.agentic.tools.code_interpreter_tool import CodeInterpreterTool
from src.agentic.tools.slack_tool import SlackTool

__all__ = [
    "BaseTool",
    "ToolResult",
    "VectorSearchTool",
    "SQLQueryTool",
    "ExternalAPITool",
    "EmailTool",
    "CRMTool",
    "DocumentGeneratorTool",
    "ComplianceCheckerTool",
    "JiraTool",
    "SharePointTool",
    "WebSearchTool",
    "CodeInterpreterTool",
    "SlackTool"
]
