"""
Vector Search Tool
Wraps RAG retrieval for the agentic system.
Integrates with document-level RBAC for security.

Full RAG Pipeline:
1. SemanticCache check (if enabled)
2. HybridSearch (vector + BM25)
3. Reranker (Cohere/CrossEncoder/LLM)
4. RBAC filtering
5. Cache update
"""

from typing import Dict, Any, Optional, List

from src.agentic.tools.base_tool import BaseTool, ToolResult, ToolStatus
from src.knowledge_base.retriever import HierarchicalRetriever, UserRole
from src.knowledge_base.main_context import MainContextDatabase
from src.auth.document_access import (
    DocumentAccessController,
    UserAccessContext,
    filter_results_by_access
)


class VectorSearchTool(BaseTool):
    """
    Tool for semantic search across the knowledge base.
    Uses the full RAG pipeline with advanced retrieval.
    
    RAG Pipeline:
    1. SemanticCache - Check for cached similar queries
    2. HybridSearch - Combine vector search + BM25 keyword search
    3. Reranker - Re-rank results using Cohere/CrossEncoder/LLM
    4. RBAC - Filter based on user permissions
    
    Security Features:
    - Role-based department access
    - Document-level access control
    - Data classification filtering
    - Sensitive field redaction
    - Audit logging
    """
    
    def __init__(
        self,
        use_hybrid_search: bool = True,
        use_reranker: bool = True,
        use_cache: bool = True,
        reranker_type: str = "cross_encoder",  # cross_encoder (default, most accurate), cohere, llm
        use_colbert: bool = False   # ColBERT: faster than cross-encoder but less accurate (enable explicitly)
    ):
        """Initialize vector search tool with full RAG pipeline."""
        super().__init__(
            name="vector_search",
            description="""
            Searches the company knowledge base using semantic similarity.
            Use this to find:
            - Product specifications and datasheets
            - Technical documentation
            - Installation guides
            - Safety information
            - Industry standards information
            
            Results are automatically filtered based on user permissions.
            """
        )
        
        # Core retrieval
        self.retriever = HierarchicalRetriever()
        self.main_db = MainContextDatabase()
        self.access_controller = DocumentAccessController()
        
        # Advanced retrieval options
        self.use_hybrid_search = use_hybrid_search
        self.use_reranker = use_reranker
        self.use_cache = use_cache
        self.reranker_type = reranker_type
        self.use_colbert = use_colbert  # Explicit opt-in for ColBERT
        
        # Lazy initialization for advanced components
        self._hybrid_search = None
        self._reranker = None
        self._cache = None
        self._colbert_reranker = None
    
    def _get_hybrid_search(self):
        """Lazy initialization of HybridSearch."""
        if self._hybrid_search is None and self.use_hybrid_search:
            from src.agentic.retrieval.hybrid_search import HybridSearch
            self._hybrid_search = HybridSearch(
                vector_weight=0.7,
                bm25_weight=0.3
            )
        return self._hybrid_search
    
    def _get_reranker(self):
        """Lazy initialization of Reranker."""
        if self._reranker is None and self.use_reranker:
            from src.agentic.retrieval.reranker import Reranker
            self._reranker = Reranker(
                mode=self.reranker_type,
                top_k=10
            )
        return self._reranker
    
    def _get_cache(self):
        """Lazy initialization of SemanticCache."""
        if self._cache is None and self.use_cache:
            from src.agentic.retrieval.semantic_cache import SemanticCache
            self._cache = SemanticCache(
                similarity_threshold=0.92,
                ttl_seconds=3600,
                max_entries=1000
            )
        return self._cache
    
    def _get_colbert_reranker(self):
        """Lazy initialization of ColBERT reranker (opt-in, faster than cross-encoder)."""
        if self._colbert_reranker is None and self.use_colbert:
            try:
                from src.sota.colbert_reranker import get_colbert_reranker
                self._colbert_reranker = get_colbert_reranker()
            except ImportError:
                pass  # ColBERT/RAGatouille not available
        return self._colbert_reranker
    
    async def execute(
        self,
        query: str,
        parameters: Dict[str, Any],
        intent: Optional[str] = None
    ) -> ToolResult:
        """
        Execute semantic search with full RAG pipeline.
        
        Pipeline:
        1. Check SemanticCache for similar cached queries
        2. Execute HybridSearch (vector + BM25)
        3. Rerank results for relevance
        4. Apply RBAC filtering
        5. Update cache with results
        
        Args:
            query: Search query
            parameters: Additional parameters (user_role, department, filters, user_context)
            intent: Query intent for context
            
        Returns:
            ToolResult with filtered search results
        """
        try:
            # Determine user role for access control
            role_str = parameters.get("user_role", "employee")
            try:
                user_role = UserRole(role_str)
            except ValueError:
                user_role = UserRole.EMPLOYEE
            
            department = parameters.get("department")
            n_results = parameters.get("limit", 10)
            use_cache = parameters.get("use_cache", self.use_cache)
            use_rerank = parameters.get("use_rerank", self.use_reranker)
            
            # Get user context for document-level filtering
            user_context = parameters.get("user_context", {})
            if not user_context:
                user_context = {
                    "user_id": parameters.get("user_id", "anonymous"),
                    "role": role_str,
                    "department": department or "general"
                }
            
            # === STEP 1: Check SemanticCache ===
            cache_hit = False
            if use_cache:
                cache = self._get_cache()
                if cache:
                    cached_result = await cache.get(query, context=user_context)
                    if cached_result:
                        cache_hit = True
                        return ToolResult(
                            tool_name=self.name,
                            status=ToolStatus.SUCCESS,
                            data={
                                "results": cached_result.get("results", []),
                                "total_found": cached_result.get("total_found", 0),
                                "query": query,
                                "cache_hit": True
                            },
                            sources=cached_result.get("sources", []),
                            metadata={
                                "user_role": role_str,
                                "department": department,
                                "intent": intent,
                                "cache_hit": True
                            }
                        )
            
            # === STEP 2: Execute Retrieval ===
            # Use HybridSearch if enabled, otherwise standard retrieval
            if self.use_hybrid_search:
                hybrid = self._get_hybrid_search()
                if hybrid:
                    # Get more results for reranking
                    retrieval_response = self.retriever.retrieve(
                        query=query,
                        user_role=user_role,
                        user_department=department,
                        n_results=n_results * 3
                    )
                    
                    # Index results for hybrid search
                    docs_for_hybrid = [
                        {
                            "content": r.content,
                            "metadata": r.metadata,
                            "embedding": None  # Will be computed
                        }
                        for r in retrieval_response.all_results
                    ]
                    
                    if docs_for_hybrid:
                        hybrid.index_documents(docs_for_hybrid)
                        hybrid_results = await hybrid.search(
                            query=query,
                            top_k=n_results * 2
                        )
                        
                        # Map back to our format
                        raw_results = []
                        for hr in hybrid_results:
                            raw_results.append({
                                "id": hr.metadata.get("doc_id", str(hash(hr.content[:50]))),
                                "content": hr.content,
                                "relevance_score": hr.combined_score,
                                "source": hr.metadata.get("source", "knowledge_base"),
                                "metadata": hr.metadata
                            })
                    else:
                        raw_results = []
                else:
                    # Fallback to standard retrieval
                    retrieval_response = self.retriever.retrieve(
                        query=query,
                        user_role=user_role,
                        user_department=department,
                        n_results=n_results * 2
                    )
                    raw_results = self._format_retrieval_results(retrieval_response)
            else:
                # Standard retrieval
                retrieval_response = self.retriever.retrieve(
                    query=query,
                    user_role=user_role,
                    user_department=department,
                    n_results=n_results * 2
                )
                raw_results = self._format_retrieval_results(retrieval_response)
            
            # === STEP 3: Rerank Results ===
            # Accuracy: Cross-Encoder (default) > ColBERT (opt-in) > Dense
            # Speed:    Dense > ColBERT > Cross-Encoder
            if use_rerank and raw_results:
                reranked = False
                
                # Option 1: Use ColBERT if explicitly enabled (faster, slightly less accurate)
                if self.use_colbert:
                    colbert = self._get_colbert_reranker()
                    if colbert:
                        try:
                            colbert_results = colbert.rerank(
                                query=query,
                                documents=raw_results,
                                top_k=n_results * 2
                            )
                            raw_results = [
                                {
                                    **r.to_dict(),
                                    "relevance_score": r.score,
                                    "rerank_score": r.score,
                                    "reranker": "colbert"
                                }
                                for r in colbert_results
                            ]
                            reranked = True
                        except Exception as e:
                            logger.debug(f"ColBERT rerank failed: {e}")
                
                # Option 2: Default - Use cross-encoder/standard reranker (most accurate)
                if not reranked:
                    reranker = self._get_reranker()
                    if reranker:
                        reranked_results = await reranker.rerank(
                            query=query,
                            documents=raw_results,
                            top_k=n_results * 2
                        )
                        raw_results = [
                            {
                                **r.document,
                                "relevance_score": r.score,
                                "rerank_score": r.score,
                                "reranker": self.reranker_type
                            }
                            for r in reranked_results
                        ]
            
            # === STEP 4: Apply RBAC Filtering ===
            filtered_results = filter_results_by_access(user_context, raw_results)
            
            # Limit to requested count after filtering
            filtered_results = filtered_results[:n_results]
            
            # Build response
            results = []
            sources = []
            
            for result in filtered_results:
                results.append({
                    "content": result["content"],
                    "relevance_score": result.get("relevance_score", 0.0),
                    "source": result.get("source", "knowledge_base"),
                    "metadata": result.get("metadata", {})
                })
                
                sources.append({
                    "document": result.get("metadata", {}).get("file_name", "Knowledge Base"),
                    "relevance": round(result.get("relevance_score", 0.0), 3)
                })
            
            # === STEP 5: Update Cache ===
            if use_cache and not cache_hit and results:
                cache = self._get_cache()
                if cache:
                    await cache.set(
                        query=query,
                        result={
                            "results": results,
                            "sources": sources,
                            "total_found": len(results)
                        },
                        context=user_context
                    )
            
            # Calculate stats
            total_before = len(raw_results) if raw_results else 0
            filtered_count = total_before - len(filtered_results)
            
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.SUCCESS if results else ToolStatus.PARTIAL,
                data={
                    "results": results,
                    "total_found": len(filtered_results),
                    "total_before_filtering": total_before,
                    "filtered_by_access": filtered_count,
                    "query": query,
                    "cache_hit": cache_hit,
                    "hybrid_search": self.use_hybrid_search,
                    "reranked": use_rerank
                },
                sources=sources,
                metadata={
                    "user_role": role_str,
                    "department": department,
                    "intent": intent,
                    "access_filtered": filtered_count > 0,
                    "pipeline": {
                        "cache": use_cache,
                        "hybrid_search": self.use_hybrid_search,
                        "reranker": use_rerank
                    }
                }
            )
            
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error=str(e)
            )
    
    def _format_retrieval_results(self, retrieval_response) -> List[Dict[str, Any]]:
        """Format retrieval response into standard result format."""
        raw_results = []
        for result in retrieval_response.all_results:
            raw_results.append({
                "id": result.metadata.get("doc_id", str(hash(result.content[:50]))),
                "content": result.content,
                "relevance_score": result.relevance_score,
                "source": result.source,
                "metadata": result.metadata
            })
        return raw_results
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool parameter schema."""
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "user_role": {
                    "type": "string",
                    "description": "User role for access control",
                    "enum": ["employee", "sales_rep", "engineer", "manager", "admin"]
                },
                "department": {
                    "type": "string",
                    "description": "User department"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results",
                    "default": 10
                },
                "filters": {
                    "type": "object",
                    "description": "Additional filters",
                    "properties": {
                        "document_type": {"type": "string"},
                        "category": {"type": "string"}
                    }
                }
            },
            "required": ["query"]
        }


class ProductDatabaseTool(BaseTool):
    """
    Tool for searching the product database.
    Used for product selection and specification lookups.
    """
    
    # Product catalog - loaded dynamically from JSON files via JDJonesDataLoader
    # No longer hardcoded - see _ensure_catalog_loaded() method
    PRODUCT_CATALOG = {}  # Populated on first access via _ensure_catalog_loaded()
    _catalog_loaded = False
    
    # OPTIMIZATION: Pre-indexed hash maps for O(1) lookups
    _products_by_certification = {}  # certification -> [product_codes]
    _products_by_media = {}          # media -> [product_codes]
    _products_by_industry = {}       # industry -> [product_codes]
    _product_flat = {}               # product_code -> product_data (flat access)
    
    @classmethod
    def _ensure_catalog_loaded(cls):
        """Load product catalog from JSON files if not already loaded."""
        if cls._catalog_loaded:
            return
            
        try:
            from src.data_ingestion.jd_jones_data_loader import get_data_loader
            loader = get_data_loader()
            products = loader.get_all_products()
            
            # Group products by series/category for organized access
            categories = {}
            
            # OPTIMIZATION: Build inverted indexes while loading
            cert_index = {}
            media_index = {}
            industry_index = {}
            flat_products = {}
            
            for code, prod in products.items():
                category = prod.get("category", "Other")
                if category not in categories:
                    categories[category] = {
                        "name": category,
                        "type": "packing",
                        "description": f"Products in {category} category",
                        "variants": {}
                    }
                
                specs = prod.get("specs", {})
                variant_data = {
                    "name": prod.get("name", ""),
                    "max_temperature": specs.get("temperature_max"),
                    "min_temperature": specs.get("temperature_min"),
                    "max_pressure": specs.get("pressure_static"),
                    "pressure_rotary": specs.get("pressure_rotary"),
                    "pressure_reciprocating": specs.get("pressure_reciprocating"),
                    "shaft_speed": specs.get("shaft_speed_rotary"),
                    "media": prod.get("service_media", []),
                    "certifications": prod.get("certifications", []),
                    "industries": prod.get("industries", []),
                    "applications": prod.get("applications", []),
                }
                categories[category]["variants"][code] = variant_data
                flat_products[code] = variant_data
                
                # Build inverted indexes for O(1) lookups
                for cert in prod.get("certifications", []):
                    cert_lower = cert.lower()
                    if cert_lower not in cert_index:
                        cert_index[cert_lower] = []
                    cert_index[cert_lower].append(code)
                
                for m in prod.get("service_media", []):
                    m_lower = m.lower()
                    if m_lower not in media_index:
                        media_index[m_lower] = []
                    media_index[m_lower].append(code)
                
                for ind in prod.get("industries", []):
                    ind_lower = ind.lower()
                    if ind_lower not in industry_index:
                        industry_index[ind_lower] = []
                    industry_index[ind_lower].append(code)
            
            cls.PRODUCT_CATALOG = categories
            cls._products_by_certification = cert_index
            cls._products_by_media = media_index
            cls._products_by_industry = industry_index
            cls._product_flat = flat_products
            cls._catalog_loaded = True
            
            import logging
            logging.info(f"Loaded product catalog with {len(categories)} categories, "
                        f"{len(flat_products)} products, "
                        f"{len(cert_index)} cert indexes, "
                        f"{len(media_index)} media indexes from JSON files")
            
        except ImportError:
            import logging
            logging.warning("JDJonesDataLoader not available, PRODUCT_CATALOG will remain empty")
            cls._catalog_loaded = True
        except Exception as e:
            import logging
            logging.error(f"Error loading product catalog from JSON: {e}")
            cls._catalog_loaded = True


    def __init__(self):
        """Initialize product database tool."""
        super().__init__(
            name="product_database",
            description="""
            Searches the product catalog for matching products.
            Returns product specifications, certifications, and suitability.
            Use this for:
            - Finding products matching requirements
            - Getting detailed specifications
            - Checking certifications
            """
        )
    
    async def execute(
        self,
        query: str,
        parameters: Dict[str, Any],
        intent: Optional[str] = None
    ) -> ToolResult:
        """
        Search product database.
        
        Args:
            query: Search query
            parameters: Filter parameters (temperature, pressure, media, etc.)
            intent: Query intent
            
        Returns:
            ToolResult with matching products
        """
        try:
            # Ensure product catalog is loaded
            self._ensure_catalog_loaded()
            
            matching_products = []
            
            # Extract filter criteria
            max_temp = parameters.get("temperature")
            max_pressure = parameters.get("pressure")
            media = parameters.get("media")
            certifications = parameters.get("certifications", [])
            product_type = parameters.get("product_type")
            
            # Parse numeric values
            if max_temp:
                import re
                temp_match = re.search(r'(\d+)', str(max_temp))
                max_temp = int(temp_match.group(1)) if temp_match else None
            
            if max_pressure:
                import re
                pressure_match = re.search(r'(\d+)', str(max_pressure))
                max_pressure = int(pressure_match.group(1)) if pressure_match else None
            
            # Search products
            for series_id, series in self.PRODUCT_CATALOG.items():
                # Filter by type if specified
                if product_type and series["type"] != product_type:
                    continue
                
                for variant_id, variant in series["variants"].items():
                    score = 0
                    matches = []
                    
                    # Check temperature
                    if max_temp:
                        if variant["max_temperature"] >= max_temp:
                            score += 1
                            matches.append("temperature")
                        else:
                            continue  # Doesn't meet requirement
                    
                    # Check pressure
                    if max_pressure:
                        if variant["max_pressure"] >= max_pressure:
                            score += 1
                            matches.append("pressure")
                        else:
                            continue  # Doesn't meet requirement
                    
                    # Check media
                    if media:
                        media_lower = media.lower()
                        if any(m.lower() in media_lower or media_lower in m.lower() 
                               for m in variant["media"]):
                            score += 1
                            matches.append("media")
                    
                    # Check certifications
                    if certifications:
                        matched_certs = [
                            c for c in certifications 
                            if any(c.upper() in cert.upper() for cert in variant["certifications"])
                        ]
                        if matched_certs:
                            score += len(matched_certs)
                            matches.append(f"certifications: {', '.join(matched_certs)}")
                    
                    # Check if query mentions product name
                    if series_id.lower() in query.lower() or variant_id.lower() in query.lower():
                        score += 2
                        matches.append("name_match")
                    
                    if score > 0:
                        matching_products.append({
                            "product_id": variant_id,
                            "series": series["name"],
                            "type": series["type"],
                            "specifications": {
                                "max_temperature": f"{variant['max_temperature']}°C",
                                "max_pressure": f"{variant['max_pressure']} bar",
                                "compatible_media": variant["media"],
                                "certifications": variant["certifications"],
                                "applications": variant["applications"]
                            },
                            "match_score": score,
                            "matches": matches
                        })
            
            # Sort by match score
            matching_products.sort(key=lambda x: x["match_score"], reverse=True)
            
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.SUCCESS if matching_products else ToolStatus.PARTIAL,
                data={
                    "products": matching_products[:10],
                    "total_matches": len(matching_products),
                    "filters_applied": {
                        k: v for k, v in parameters.items() 
                        if v and k in ["temperature", "pressure", "media", "certifications"]
                    }
                },
                sources=[{
                    "document": "Product Database",
                    "relevance": 1.0
                }],
                metadata={"query": query}
            )
            
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                status=ToolStatus.FAILED,
                error=str(e)
            )
    
    def get_schema(self) -> Dict[str, Any]:
        """Get tool parameter schema."""
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "temperature": {"type": "string", "description": "Required temperature rating"},
                "pressure": {"type": "string", "description": "Required pressure rating"},
                "media": {"type": "string", "description": "Fluid/media in contact"},
                "certifications": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Required certifications"
                },
                "product_type": {
                    "type": "string",
                    "enum": ["gasket", "packing", "expansion_joint"]
                }
            }
        }
