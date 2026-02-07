"""
Vector store manager for ChromaDB operations.
Handles collection management, document storage, and similarity search.
"""
import logging
import uuid
from typing import List, Dict, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)
from dataclasses import dataclass
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions

from src.config.settings import settings
from src.data_ingestion.embedding_generator import EmbeddedDocument


@dataclass
class SearchResult:
    """Represents a search result from the vector store."""
    document_id: str
    content: str
    metadata: Dict[str, Any]
    relevance_score: float
    chunk_index: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_id": self.document_id,
            "content": self.content,
            "metadata": self.metadata,
            "relevance_score": self.relevance_score,
            "chunk_index": self.chunk_index
        }


class VectorStoreManager:
    """
    Manages ChromaDB vector store operations.
    Handles collection creation, document storage, and similarity search.
    """
    
    def __init__(
        self,
        host: str = None,
        port: int = None,
        persist_directory: str = None,
        use_http_client: bool = True
    ):
        """
        Initialize vector store manager.
        
        Args:
            host: ChromaDB server host
            port: ChromaDB server port
            persist_directory: Directory for persistent storage
            use_http_client: Whether to use HTTP client (for Docker)
        """
        self.host = host or settings.chroma_host
        self.port = port or settings.chroma_port
        self.persist_directory = persist_directory or settings.chroma_persist_directory
        
        # Initialize ChromaDB client
        if use_http_client:
            try:
                self.client = chromadb.HttpClient(
                    host=self.host,
                    port=self.port
                )
            except Exception:
                # Fall back to persistent client if HTTP fails
                self.client = chromadb.PersistentClient(
                    path=self.persist_directory
                )
        else:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory
            )
        
        # Initialize embedding function based on provider setting
        embedding_provider = getattr(settings, 'embedding_provider', 'openai').lower()
        
        if embedding_provider == 'local':
            # Use sentence-transformers for local embeddings (FREE)
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=settings.embedding_model
            )
        else:
            # Use OpenAI embeddings (requires API key)
            self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                api_key=settings.openai_api_key,
                model_name=settings.embedding_model
            )
    
    def create_collection(
        self,
        collection_name: str,
        metadata: Optional[Dict[str, Any]] = None,
        distance_metric: str = "cosine"
    ) -> chromadb.Collection:
        """
        Create or get a collection.
        
        Args:
            collection_name: Name of the collection
            metadata: Optional metadata for the collection
            distance_metric: Distance metric (cosine, l2, ip)
            
        Returns:
            ChromaDB Collection object
        """
        collection_metadata = metadata or {}
        collection_metadata["hnsw:space"] = distance_metric
        
        return self.client.get_or_create_collection(
            name=collection_name,
            metadata=collection_metadata,
            embedding_function=self.embedding_function
        )
    
    def get_collection(self, collection_name: str) -> chromadb.Collection:
        """
        Get an existing collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            ChromaDB Collection object
        """
        return self.client.get_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            True if deleted successfully
        """
        try:
            self.client.delete_collection(name=collection_name)
            return True
        except Exception as e:
            logger.error(f"Error deleting collection {collection_name}: {str(e)}")
            return False
    
    def list_collections(self) -> List[str]:
        """
        List all collection names.
        
        Returns:
            List of collection names
        """
        collections = self.client.list_collections()
        return [c.name for c in collections]
    
    def add_documents(
        self,
        collection_name: str,
        documents: List[EmbeddedDocument],
        upsert: bool = True
    ) -> int:
        """
        Add documents to a collection.
        
        Args:
            collection_name: Name of the collection
            documents: List of EmbeddedDocument objects
            upsert: Whether to upsert (update if exists)
            
        Returns:
            Number of documents added
        """
        if not documents:
            return 0
        
        collection = self.create_collection(collection_name)
        
        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        contents = []
        metadatas = []
        
        for doc in documents:
            doc_id = f"{doc.document_id}_{doc.chunk_index}"
            ids.append(doc_id)
            embeddings.append(doc.embedding)
            contents.append(doc.content)
            metadatas.append(doc.metadata)
        
        # Add or upsert documents
        if upsert:
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=contents,
                metadatas=metadatas
            )
        else:
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=contents,
                metadatas=metadatas
            )
        
        return len(documents)
    
    def add_documents_without_embeddings(
        self,
        collection_name: str,
        documents: List[Dict[str, Any]],
        upsert: bool = True
    ) -> int:
        """
        Add documents without pre-computed embeddings.
        ChromaDB will generate embeddings using the embedding function.
        
        Args:
            collection_name: Name of the collection
            documents: List of dicts with 'id', 'content', 'metadata'
            upsert: Whether to upsert
            
        Returns:
            Number of documents added
        """
        if not documents:
            return 0
        
        collection = self.create_collection(collection_name)
        
        ids = [doc["id"] for doc in documents]
        contents = [doc["content"] for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]
        
        if upsert:
            collection.upsert(
                ids=ids,
                documents=contents,
                metadatas=metadatas
            )
        else:
            collection.add(
                ids=ids,
                documents=contents,
                metadatas=metadatas
            )
        
        return len(documents)
    
    def search(
        self,
        collection_name: str,
        query_text: str,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
        include_embeddings: bool = False
    ) -> List[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            collection_name: Name of the collection
            query_text: Query text
            n_results: Number of results to return
            where: Metadata filter
            where_document: Document content filter
            include_embeddings: Whether to include embeddings in results
            
        Returns:
            List of SearchResult objects
        """
        collection = self.get_collection(collection_name)
        
        include = ["documents", "metadatas", "distances"]
        if include_embeddings:
            include.append("embeddings")
        
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=include
        )
        
        search_results = []
        
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                # ChromaDB returns distance, convert to similarity
                distance = results["distances"][0][i] if results["distances"] else 0
                # For cosine distance, similarity = 1 - distance
                relevance_score = 1 - distance
                
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                
                search_result = SearchResult(
                    document_id=metadata.get("document_id", doc_id.split("_")[0]),
                    content=results["documents"][0][i] if results["documents"] else "",
                    metadata=metadata,
                    relevance_score=relevance_score,
                    chunk_index=metadata.get("chunk_index", 0)
                )
                search_results.append(search_result)
        
        return search_results
    
    def search_with_embedding(
        self,
        collection_name: str,
        query_embedding: List[float],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Search using a pre-computed embedding vector.
        
        Args:
            collection_name: Name of the collection
            query_embedding: Query embedding vector
            n_results: Number of results
            where: Metadata filter
            
        Returns:
            List of SearchResult objects
        """
        collection = self.get_collection(collection_name)
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        search_results = []
        
        if results and results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0
                relevance_score = 1 - distance
                
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                
                search_result = SearchResult(
                    document_id=metadata.get("document_id", doc_id.split("_")[0]),
                    content=results["documents"][0][i] if results["documents"] else "",
                    metadata=metadata,
                    relevance_score=relevance_score,
                    chunk_index=metadata.get("chunk_index", 0)
                )
                search_results.append(search_result)
        
        return search_results
    
    def delete_documents(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Delete documents from a collection.
        
        Args:
            collection_name: Name of the collection
            ids: List of document IDs to delete
            where: Metadata filter for deletion
            
        Returns:
            True if successful
        """
        try:
            collection = self.get_collection(collection_name)
            collection.delete(ids=ids, where=where)
            return True
        except Exception as e:
            logger.error(f"Error deleting documents: {str(e)}")
            return False
    
    def get_collection_count(self, collection_name: str) -> int:
        """
        Get the number of documents in a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Document count
        """
        try:
            collection = self.get_collection(collection_name)
            return collection.count()
        except Exception:
            return 0
    
    def get_document_by_id(
        self,
        collection_name: str,
        document_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get a document by its ID.
        
        Args:
            collection_name: Name of the collection
            document_id: Document ID
            
        Returns:
            Document data or None
        """
        collection = self.get_collection(collection_name)
        
        results = collection.get(
            ids=[document_id],
            include=["documents", "metadatas", "embeddings"]
        )
        
        if results and results["ids"]:
            return {
                "id": results["ids"][0],
                "content": results["documents"][0] if results["documents"] else "",
                "metadata": results["metadatas"][0] if results["metadatas"] else {},
                "embedding": results["embeddings"][0] if results["embeddings"] else None
            }
        
        return None
