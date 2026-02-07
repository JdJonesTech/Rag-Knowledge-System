"""
Main Context Database - Level 0 (Company-Wide) Knowledge Base.
Stores and retrieves company-wide information accessible to all internal users.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.data_ingestion.vector_store import VectorStoreManager, SearchResult
from src.data_ingestion.embedding_generator import EmbeddedDocument
from src.data_ingestion.document_processor import AccessLevel
from src.config.settings import settings


@dataclass
class MainContextResult:
    """Result from main context query."""
    content: str
    metadata: Dict[str, Any]
    relevance_score: float
    document_id: str
    chunk_index: int
    is_public: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "relevance_score": self.relevance_score,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "is_public": self.is_public
        }


class MainContextDatabase:
    """
    Level 0 (Main Context) Knowledge Base.
    
    Stores company-wide information including:
    - Product catalog
    - Company policies
    - Technical specifications
    - FAQs
    - Safety guidelines
    
    Supports both internal and public (customer-facing) subsets.
    """
    
    # Collection names
    INTERNAL_COLLECTION = "jd_jones_main_internal"
    PUBLIC_COLLECTION = "jd_jones_main_public"
    
    def __init__(self):
        """Initialize main context database."""
        self.vector_store = VectorStoreManager()
        
        # Ensure collections exist
        self._initialize_collections()
    
    def _initialize_collections(self):
        """Create collections if they don't exist."""
        self.vector_store.create_collection(
            self.INTERNAL_COLLECTION,
            metadata={
                "description": "JD Jones main context - internal access",
                "level": "0",
                "access": "internal"
            }
        )
        
        self.vector_store.create_collection(
            self.PUBLIC_COLLECTION,
            metadata={
                "description": "JD Jones main context - public access",
                "level": "0",
                "access": "public"
            }
        )
    
    def add_documents(
        self,
        documents: List[EmbeddedDocument],
        is_public: bool = False
    ) -> int:
        """
        Add documents to the main context database.
        
        Args:
            documents: List of embedded documents
            is_public: Whether documents should be publicly accessible
            
        Returns:
            Number of documents added
        """
        # Add to internal collection
        internal_count = self.vector_store.add_documents(
            self.INTERNAL_COLLECTION,
            documents,
            upsert=True
        )
        
        # Also add to public collection if marked as public
        if is_public:
            # Update metadata to indicate public access
            public_docs = []
            for doc in documents:
                public_doc = EmbeddedDocument(
                    content=doc.content,
                    embedding=doc.embedding,
                    metadata={**doc.metadata, "is_public": True},
                    document_id=doc.document_id,
                    chunk_index=doc.chunk_index
                )
                public_docs.append(public_doc)
            
            self.vector_store.add_documents(
                self.PUBLIC_COLLECTION,
                public_docs,
                upsert=True
            )
        
        return internal_count
    
    def query(
        self,
        query_text: str,
        n_results: int = 10,
        include_public_only: bool = False,
        access_level: Optional[AccessLevel] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[MainContextResult]:
        """
        Query the main context database.
        
        Args:
            query_text: Search query
            n_results: Maximum number of results
            include_public_only: If True, only search public collection
            access_level: Filter by access level
            filter_metadata: Additional metadata filters
            
        Returns:
            List of MainContextResult objects
        """
        # Determine which collection to search
        collection_name = (
            self.PUBLIC_COLLECTION if include_public_only 
            else self.INTERNAL_COLLECTION
        )
        
        # Build where filter
        where = filter_metadata.copy() if filter_metadata else {}
        
        if access_level:
            where["access_level"] = access_level.value
        
        # Perform search
        search_results = self.vector_store.search(
            collection_name=collection_name,
            query_text=query_text,
            n_results=n_results,
            where=where if where else None
        )
        
        # Convert to MainContextResult
        results = []
        for sr in search_results:
            result = MainContextResult(
                content=sr.content,
                metadata=sr.metadata,
                relevance_score=sr.relevance_score,
                document_id=sr.document_id,
                chunk_index=sr.chunk_index,
                is_public=include_public_only or sr.metadata.get("is_public", False)
            )
            results.append(result)
        
        return results
    
    def query_by_category(
        self,
        query_text: str,
        category: str,
        n_results: int = 10,
        include_public_only: bool = False
    ) -> List[MainContextResult]:
        """
        Query documents by category.
        
        Args:
            query_text: Search query
            category: Document category (e.g., 'product_catalog', 'policy')
            n_results: Maximum results
            include_public_only: Whether to search public only
            
        Returns:
            List of results
        """
        return self.query(
            query_text=query_text,
            n_results=n_results,
            include_public_only=include_public_only,
            filter_metadata={"category": category}
        )
    
    def query_by_document_type(
        self,
        query_text: str,
        document_type: str,
        n_results: int = 10
    ) -> List[MainContextResult]:
        """
        Query documents by file type.
        
        Args:
            query_text: Search query
            document_type: File type (e.g., 'pdf', 'docx')
            n_results: Maximum results
            
        Returns:
            List of results
        """
        return self.query(
            query_text=query_text,
            n_results=n_results,
            filter_metadata={"file_type": document_type}
        )
    
    def get_document_count(self, include_public_only: bool = False) -> int:
        """
        Get total document count.
        
        Args:
            include_public_only: Count only public documents
            
        Returns:
            Document count
        """
        collection_name = (
            self.PUBLIC_COLLECTION if include_public_only 
            else self.INTERNAL_COLLECTION
        )
        return self.vector_store.get_collection_count(collection_name)
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete a document by ID from both collections.
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            True if successful
        """
        # Delete from internal
        internal_deleted = self.vector_store.delete_documents(
            self.INTERNAL_COLLECTION,
            where={"document_id": document_id}
        )
        
        # Delete from public
        public_deleted = self.vector_store.delete_documents(
            self.PUBLIC_COLLECTION,
            where={"document_id": document_id}
        )
        
        return internal_deleted or public_deleted
    
    def update_document_access(
        self,
        document_id: str,
        make_public: bool
    ) -> bool:
        """
        Update document public/private status.
        
        Args:
            document_id: Document ID
            make_public: Whether to make public
            
        Returns:
            True if successful
        """
        # Get document from internal collection
        doc = self.vector_store.get_document_by_id(
            self.INTERNAL_COLLECTION,
            document_id
        )
        
        if not doc:
            return False
        
        if make_public:
            # Add to public collection
            self.vector_store.add_documents_without_embeddings(
                self.PUBLIC_COLLECTION,
                [{
                    "id": document_id,
                    "content": doc["content"],
                    "metadata": {**doc["metadata"], "is_public": True}
                }],
                upsert=True
            )
        else:
            # Remove from public collection
            self.vector_store.delete_documents(
                self.PUBLIC_COLLECTION,
                ids=[document_id]
            )
        
        return True
    
    def format_context_for_llm(
        self,
        results: List[MainContextResult],
        max_tokens: int = 3000
    ) -> str:
        """
        Format search results into a context string for LLM.
        
        Args:
            results: List of search results
            max_tokens: Maximum token limit (approximate)
            
        Returns:
            Formatted context string
        """
        if not results:
            return "No relevant information found in the knowledge base."
        
        context_parts = []
        current_tokens = 0
        tokens_per_char = 0.25  # Rough estimate
        
        for result in results:
            # Estimate tokens for this result
            result_text = f"[Source: {result.metadata.get('file_name', 'Unknown')}]\n{result.content}\n"
            estimated_tokens = int(len(result_text) * tokens_per_char)
            
            if current_tokens + estimated_tokens > max_tokens:
                break
            
            context_parts.append(result_text)
            current_tokens += estimated_tokens
        
        return "\n---\n".join(context_parts)
