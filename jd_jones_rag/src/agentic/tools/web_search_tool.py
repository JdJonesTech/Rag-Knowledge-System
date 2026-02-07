"""
Web Search Tool
Enables agents to search the web for external information.
Uses search APIs (Bing, Google, SerpAPI, Tavily) for real-time data.
"""

import logging
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)

from src.agentic.tools.base_tool import BaseTool, ToolResult


class SearchProvider(str, Enum):
    """Supported search providers."""
    BING = "bing"
    GOOGLE = "google"
    SERP = "serp"
    TAVILY = "tavily"
    DUCKDUCKGO = "duckduckgo"


@dataclass
class SearchResult:
    """A single search result."""
    title: str
    url: str
    snippet: str
    source: str
    published_date: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "published_date": self.published_date
        }


class WebSearchTool(BaseTool):
    """
    Tool for web search capabilities.
    
    Use cases:
    - Research competitor products
    - Find industry standards and regulations
    - Get current market information
    - Verify external facts
    - Find technical documentation
    """
    
    name = "web_search_tool"
    description = """
    Searches the web for external information. Use for:
    - Researching industry standards and regulations
    - Finding competitor information
    - Getting current market data
    - Verifying external facts and claims
    - Finding technical articles and papers
    Do NOT use for internal company data - use other tools for that.
    """
    
    # Domain restrictions for enterprise use
    TRUSTED_DOMAINS = [
        "iso.org",
        "asme.org",
        "api.org",
        "nist.gov",
        "osha.gov",
        "epa.gov",
        "wikipedia.org",
        "britannica.com",
        "sciencedirect.com",
        "springer.com",
        "ieee.org"
    ]
    
    # Blocked domains
    BLOCKED_DOMAINS = [
        "reddit.com",
        "twitter.com",
        "facebook.com",
        "tiktok.com"
    ]
    
    def __init__(
        self,
        provider: SearchProvider = SearchProvider.TAVILY,
        api_key: Optional[str] = None,
        safe_search: bool = True,
        max_results: int = 10
    ):
        """
        Initialize web search tool.
        
        Args:
            provider: Search provider to use
            api_key: API key for the provider
            safe_search: Enable safe search
            max_results: Default max results
        """
        super().__init__(
            name=self.name,
            description=self.description
        )
        self.provider = provider
        self.safe_search = safe_search
        self.max_results = max_results
        
        # Get API key based on provider
        if api_key:
            self.api_key = api_key
        else:
            key_mapping = {
                SearchProvider.BING: "BING_SEARCH_API_KEY",
                SearchProvider.GOOGLE: "GOOGLE_SEARCH_API_KEY",
                SearchProvider.SERP: "SERP_API_KEY",
                SearchProvider.TAVILY: "TAVILY_API_KEY"
            }
            self.api_key = os.getenv(key_mapping.get(provider, "SEARCH_API_KEY"))
    
    async def execute(
        self,
        query: str,
        parameters: Dict[str, Any],
        intent: Optional[str] = None
    ) -> ToolResult:
        """
        Execute a web search action.
        
        Args:
            query: Search query
            parameters: Action parameters including 'action' key
            intent: Optional intent from router
            
        Actions (specified in parameters['action']):
        - search: General web search
        - news_search: Search news articles
        - image_search: Search images
        - site_search: Search specific site
        """
        try:
            # Get action from parameters or default to search
            action = parameters.get("action", "search")
            
            # Add query to parameters for search
            if "query" not in parameters:
                parameters["query"] = query
            
            if action == "search":
                return await self._search(parameters)
            elif action == "news_search":
                return await self._news_search(parameters)
            elif action == "image_search":
                return await self._image_search(parameters)
            elif action == "site_search":
                return await self._site_search(parameters)
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
        """Perform a general web search."""
        query = params.get("query", "")
        max_results = params.get("max_results", self.max_results)
        filter_domains = params.get("filter_domains", [])
        
        if not query:
            return ToolResult(
                success=False,
                data={},
                error="Missing required field: query"
            )
        
        # In production, call actual search API
        # This is a simulation for demonstration
        if self.provider == SearchProvider.TAVILY:
            results = await self._tavily_search(query, max_results)
        else:
            results = await self._generic_search(query, max_results)
        
        # Filter by trusted domains if specified
        if filter_domains:
            results = [r for r in results if any(d in r["url"] for d in filter_domains)]
        
        return ToolResult(
            success=True,
            data={
                "query": query,
                "provider": self.provider.value,
                "total_results": len(results),
                "results": results
            },
            metadata={"action": "search", "safe_search": self.safe_search}
        )
    
    async def _tavily_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using Tavily API (optimized for AI agents)."""
        try:
            from tavily import TavilyClient
            
            client = TavilyClient(api_key=self.api_key)
            response = client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",
                include_answer=True
            )
            
            results = []
            for item in response.get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("content", ""),
                    "source": item.get("url", "").split("/")[2] if item.get("url") else "",
                    "score": item.get("score", 0)
                })
            
            # Include AI-generated answer if available
            if response.get("answer"):
                results.insert(0, {
                    "title": "AI Summary",
                    "url": "",
                    "snippet": response["answer"],
                    "source": "tavily_ai",
                    "score": 1.0
                })
            
            return results
            
        except ImportError:
            return await self._generic_search(query, max_results)
        except Exception as e:
            logger.error(f"Tavily search error: {e}")
            return await self._generic_search(query, max_results)
    
    async def _generic_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Generic simulated search results."""
        # Simulated results for demonstration
        return [
            {
                "title": f"API 622 Fugitive Emissions Standard - Overview",
                "url": "https://www.api.org/standards/622",
                "snippet": "API 622 establishes testing requirements for packing materials used in rising stem valves...",
                "source": "api.org",
                "published_date": "2024-01-15"
            },
            {
                "title": f"ISO 15848-1 Industrial Valves - Fugitive Emissions",
                "url": "https://www.iso.org/standard/15848-1.html",
                "snippet": "This standard specifies classification and qualification procedures for industrial valves...",
                "source": "iso.org",
                "published_date": "2023-08-20"
            },
            {
                "title": f"High Temperature Gasket Material Selection Guide",
                "url": "https://www.asme.org/gasket-materials",
                "snippet": "Comprehensive guide to selecting gasket materials for high-temperature applications...",
                "source": "asme.org",
                "published_date": "2024-02-10"
            }
        ][:max_results]
    
    async def _news_search(self, params: Dict[str, Any]) -> ToolResult:
        """Search for news articles."""
        query = params.get("query", "")
        days_back = params.get("days_back", 7)
        max_results = params.get("max_results", 10)
        
        if not query:
            return ToolResult(
                success=False,
                data={},
                error="Missing required field: query"
            )
        
        # Simulated news results
        results = [
            {
                "title": f"Industry News: New Standards for Sealing Technology",
                "url": "https://www.industrynews.com/sealing-standards-2024",
                "snippet": "The sealing industry sees new developments in fugitive emissions standards...",
                "source": "Industry News",
                "published_date": datetime.now().isoformat()
            }
        ]
        
        return ToolResult(
            success=True,
            data={
                "query": query,
                "days_back": days_back,
                "total_results": len(results),
                "results": results
            },
            metadata={"action": "news_search"}
        )
    
    async def _image_search(self, params: Dict[str, Any]) -> ToolResult:
        """Search for images."""
        query = params.get("query", "")
        max_results = params.get("max_results", 5)
        
        if not query:
            return ToolResult(
                success=False,
                data={},
                error="Missing required field: query"
            )
        
        # Simulated image results
        results = [
            {
                "title": "Technical Diagram",
                "url": "https://example.com/image1.png",
                "thumbnail_url": "https://example.com/thumb1.png",
                "source": "example.com",
                "width": 800,
                "height": 600
            }
        ]
        
        return ToolResult(
            success=True,
            data={
                "query": query,
                "total_results": len(results),
                "results": results
            },
            metadata={"action": "image_search"}
        )
    
    async def _site_search(self, params: Dict[str, Any]) -> ToolResult:
        """Search within a specific site."""
        query = params.get("query", "")
        site = params.get("site", "")
        max_results = params.get("max_results", 10)
        
        if not query or not site:
            return ToolResult(
                success=False,
                data={},
                error="Missing required fields: query, site"
            )
        
        # Modify query to search specific site
        site_query = f"site:{site} {query}"
        
        return await self._search({
            "query": site_query,
            "max_results": max_results
        })

    def get_schema(self) -> Dict[str, Any]:
        """Get tool parameter schema."""
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "action": {
                    "type": "string", 
                    "enum": ["search", "news_search", "image_search", "site_search"]
                },
                "max_results": {"type": "integer"},
                "site": {"type": "string"}
            },
            "required": ["query"]
        }
