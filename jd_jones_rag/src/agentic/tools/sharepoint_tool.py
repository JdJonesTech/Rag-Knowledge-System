"""
SharePoint Tool
Connects to Microsoft SharePoint for document retrieval and management.
Enables access to enterprise document libraries, lists, and sites.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import os

from src.agentic.tools.base_tool import BaseTool, ToolResult


class SharePointItemType(str, Enum):
    """Types of SharePoint items."""
    DOCUMENT = "document"
    FOLDER = "folder"
    LIST_ITEM = "list_item"
    PAGE = "page"
    NEWS = "news"


@dataclass
class SharePointDocument:
    """Represents a SharePoint document."""
    id: str
    name: str
    path: str
    site: str
    library: str
    content_type: str
    size_bytes: int
    created: datetime
    modified: datetime
    created_by: str
    modified_by: str
    version: str
    url: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "path": self.path,
            "site": self.site,
            "library": self.library,
            "content_type": self.content_type,
            "size_bytes": self.size_bytes,
            "created": self.created.isoformat(),
            "modified": self.modified.isoformat(),
            "created_by": self.created_by,
            "modified_by": self.modified_by,
            "version": self.version,
            "url": self.url
        }


class SharePointTool(BaseTool):
    """
    Tool for SharePoint integration.
    
    Capabilities:
    - Search documents across sites
    - Retrieve document content
    - List library contents
    - Access SharePoint lists
    - Download files
    - Get document metadata
    """
    
    name = "sharepoint_tool"
    description = """
    Accesses Microsoft SharePoint for enterprise documents. Use for:
    - Searching company documents, policies, procedures
    - Retrieving technical documentation
    - Accessing shared files and libraries
    - Finding documents by metadata or content
    """
    
    # Site mappings for JD Jones
    SITE_MAPPINGS = {
        "engineering": "/sites/Engineering",
        "sales": "/sites/Sales",
        "hr": "/sites/HumanResources",
        "quality": "/sites/QualityAssurance",
        "production": "/sites/Production",
        "legal": "/sites/Legal",
        "finance": "/sites/Finance"
    }
    
    # Document libraries
    LIBRARY_MAPPINGS = {
        "datasheets": "Product Datasheets",
        "policies": "Company Policies",
        "procedures": "Standard Procedures",
        "templates": "Document Templates",
        "training": "Training Materials",
        "specifications": "Technical Specifications"
    }
    
    def __init__(
        self,
        tenant_id: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        site_url: Optional[str] = None
    ):
        """
        Initialize SharePoint tool.
        
        Args:
            tenant_id: Azure AD tenant ID
            client_id: App registration client ID
            client_secret: App registration client secret
            site_url: Base SharePoint site URL
        """
        super().__init__(
            name=self.name,
            description=self.description
        )
        self.tenant_id = tenant_id or os.getenv("SHAREPOINT_TENANT_ID")
        self.client_id = client_id or os.getenv("SHAREPOINT_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("SHAREPOINT_CLIENT_SECRET")
        self.site_url = site_url or os.getenv("SHAREPOINT_SITE_URL", "https://jdjones.sharepoint.com")
        
        self._client = None
        self._access_token = None
    
    async def _get_access_token(self) -> str:
        """Get OAuth access token for SharePoint."""
        if self._access_token:
            return self._access_token
        
        # In production, use MSAL or requests to get token
        # This is a placeholder
        return "simulated_token"
    
    async def execute(
        self,
        query: str,
        parameters: Dict[str, Any],
        intent: Optional[str] = None
    ) -> ToolResult:
        """
        Execute a SharePoint action.
        
        Args:
            query: Search query or action description
            parameters: Action parameters including 'action' key
            intent: Optional intent from router
            
        Actions (specified in parameters['action']):
        - search: Search across SharePoint
        - get_document: Get document content
        - list_library: List library contents
        - get_list_items: Get items from a list
        - get_metadata: Get document metadata
        - download: Download a file
        """
        try:
            # Get action from parameters or default to search
            action = parameters.get("action", "search")
            
            # Add query to parameters for search
            if "query" not in parameters:
                parameters["query"] = query
            
            if action == "search":
                return await self._search(parameters)
            elif action == "get_document":
                return await self._get_document(parameters)
            elif action == "list_library":
                return await self._list_library(parameters)
            elif action == "get_list_items":
                return await self._get_list_items(parameters)
            elif action == "get_metadata":
                return await self._get_metadata(parameters)
            elif action == "download":
                return await self._download(parameters)
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
        """Search SharePoint for documents."""
        query = params.get("query", "")
        site = params.get("site")
        file_type = params.get("file_type")
        max_results = params.get("max_results", 10)
        
        # Build search query
        search_query = query
        if site:
            site_path = self.SITE_MAPPINGS.get(site.lower(), f"/sites/{site}")
            search_query += f" path:{site_path}"
        if file_type:
            search_query += f" filetype:{file_type}"
        
        # Simulated search results
        results = [
            {
                "id": "doc_001",
                "name": "PACMAAN-500_Datasheet.pdf",
                "path": "/sites/Engineering/Product Datasheets",
                "snippet": f"...{query}... specifications for high-temperature applications...",
                "modified": datetime.now().isoformat(),
                "url": f"{self.site_url}/sites/Engineering/Product%20Datasheets/PACMAAN-500_Datasheet.pdf"
            },
            {
                "id": "doc_002",
                "name": "Quality_Procedures_Manual.docx",
                "path": "/sites/QualityAssurance/Standard Procedures",
                "snippet": f"...quality control procedures for {query}...",
                "modified": datetime.now().isoformat(),
                "url": f"{self.site_url}/sites/QualityAssurance/Standard%20Procedures/Quality_Procedures_Manual.docx"
            }
        ]
        
        return ToolResult(
            success=True,
            data={
                "query": search_query,
                "total_results": len(results),
                "results": results[:max_results]
            },
            metadata={"action": "search", "site": site, "file_type": file_type}
        )
    
    async def _get_document(self, params: Dict[str, Any]) -> ToolResult:
        """Get document content."""
        if "document_id" not in params and "path" not in params:
            return ToolResult(
                success=False,
                data={},
                error="Missing required field: document_id or path"
            )
        
        # Simulated document content
        document = {
            "id": params.get("document_id", "doc_001"),
            "name": "Sample_Document.pdf",
            "content_type": "application/pdf",
            "size_bytes": 245678,
            "content_preview": "This document contains technical specifications...",
            "full_text_available": True,
            "can_extract_text": True
        }
        
        return ToolResult(
            success=True,
            data=document,
            metadata={"action": "get_document"}
        )
    
    async def _list_library(self, params: Dict[str, Any]) -> ToolResult:
        """List contents of a document library."""
        site = params.get("site", "engineering")
        library = params.get("library", "datasheets")
        folder_path = params.get("folder_path", "/")
        
        site_path = self.SITE_MAPPINGS.get(site.lower(), f"/sites/{site}")
        library_name = self.LIBRARY_MAPPINGS.get(library.lower(), library)
        
        # Simulated library contents
        items = [
            {
                "name": "PACMAAN-500_Datasheet.pdf",
                "type": "document",
                "size_bytes": 245678,
                "modified": datetime.now().isoformat()
            },
            {
                "name": "FLEXSEAL_Series",
                "type": "folder",
                "item_count": 12,
                "modified": datetime.now().isoformat()
            },
            {
                "name": "Certification_Documents.xlsx",
                "type": "document",
                "size_bytes": 89012,
                "modified": datetime.now().isoformat()
            }
        ]
        
        return ToolResult(
            success=True,
            data={
                "site": site_path,
                "library": library_name,
                "folder": folder_path,
                "item_count": len(items),
                "items": items
            },
            metadata={"action": "list_library"}
        )
    
    async def _get_list_items(self, params: Dict[str, Any]) -> ToolResult:
        """Get items from a SharePoint list."""
        site = params.get("site", "engineering")
        list_name = params.get("list_name")
        filter_query = params.get("filter")
        
        if not list_name:
            return ToolResult(
                success=False,
                data={},
                error="Missing required field: list_name"
            )
        
        # Simulated list items
        items = [
            {"ID": 1, "Title": "Item 1", "Status": "Active", "Created": datetime.now().isoformat()},
            {"ID": 2, "Title": "Item 2", "Status": "Pending", "Created": datetime.now().isoformat()}
        ]
        
        return ToolResult(
            success=True,
            data={
                "list_name": list_name,
                "item_count": len(items),
                "items": items
            },
            metadata={"action": "get_list_items", "filter": filter_query}
        )
    
    async def _get_metadata(self, params: Dict[str, Any]) -> ToolResult:
        """Get document metadata."""
        if "document_id" not in params and "path" not in params:
            return ToolResult(
                success=False,
                data={},
                error="Missing required field: document_id or path"
            )
        
        metadata = {
            "id": params.get("document_id", "doc_001"),
            "name": "Sample_Document.pdf",
            "title": "Sample Document Title",
            "content_type": "application/pdf",
            "created": datetime.now().isoformat(),
            "modified": datetime.now().isoformat(),
            "created_by": "john.doe@jdjones.com",
            "modified_by": "jane.smith@jdjones.com",
            "version": "2.0",
            "version_history": [
                {"version": "1.0", "modified": "2024-01-15", "modified_by": "john.doe"},
                {"version": "2.0", "modified": "2024-03-20", "modified_by": "jane.smith"}
            ],
            "custom_metadata": {
                "Document_Type": "Technical Specification",
                "Product_Line": "PACMAAN",
                "Review_Status": "Approved"
            }
        }
        
        return ToolResult(
            success=True,
            data=metadata,
            metadata={"action": "get_metadata"}
        )
    
    async def _download(self, params: Dict[str, Any]) -> ToolResult:
        """Download a file from SharePoint."""
        if "document_id" not in params and "path" not in params:
            return ToolResult(
                success=False,
                data={},
                error="Missing required field: document_id or path"
            )
        
        # In production, this would return actual file content or a download URL
        return ToolResult(
            success=True,
            data={
                "document_id": params.get("document_id", "doc_001"),
                "download_url": f"{self.site_url}/_layouts/download.aspx?SourceUrl=...",
                "expires_in_seconds": 3600,
                "file_name": "Sample_Document.pdf",
                "size_bytes": 245678
            },
            metadata={"action": "download"}
        )
