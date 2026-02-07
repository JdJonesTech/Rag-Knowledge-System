"""
Embedding generator for creating vector embeddings from text.
Supports both OpenAI and local (SentenceTransformer) embeddings with batch processing.

OPTIMIZATIONS:
- Singleton pattern for model caching
- Lazy initialization with warm-up
- Batched processing for efficiency
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from functools import lru_cache
from threading import Lock

from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.settings import settings
from src.data_ingestion.document_processor import ProcessedDocument


@dataclass
class EmbeddedDocument:
    """Document with embedding vector."""
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    document_id: str
    chunk_index: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index
        }


logger = logging.getLogger(__name__)

# Singleton instance
_embedding_generator_instance = None
_instance_lock = Lock()

class EmbeddingGenerator:
    """
    Generates embeddings for text.
    Supports both OpenAI and local SentenceTransformer embeddings based on settings.
    """
    
    def __init__(
        self,
        model: str = None,
        dimensions: int = None,
        batch_size: int = 100,
    ):
        """
        Initialize embedding generator.
        
        Args:
            model: Embedding model name (default from settings)
            dimensions: Embedding dimensions (default from settings)
            batch_size: Number of texts to embed per batch
        """
        self.model = model or settings.embedding_model
        self.dimensions = dimensions or settings.embedding_dimensions
        self.batch_size = batch_size
        self.provider = getattr(settings, 'embedding_provider', 'openai').lower()
        
        # Lazy initialized embeddings
        self._openai_embeddings = None
        self._sentence_transformer = None
    
    def _get_openai_embeddings(self):
        """Lazy initialization of OpenAI embeddings."""
        if self._openai_embeddings is None:
            from langchain_openai import OpenAIEmbeddings
            self._openai_embeddings = OpenAIEmbeddings(
                model=self.model,
                openai_api_key=settings.openai_api_key,
            )
        return self._openai_embeddings
    
    def _get_sentence_transformer(self):
        """Lazy initialization of SentenceTransformer."""
        if self._sentence_transformer is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._sentence_transformer = SentenceTransformer(self.model)
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
        return self._sentence_transformer
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text")
        
        if self.provider == 'local':
            model = self._get_sentence_transformer()
            embedding = model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        else:
            # OpenAI provider
            embeddings = self._get_openai_embeddings()
            return embeddings.embed_query(text)
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in a single batch.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Filter out empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        
        if not valid_texts:
            raise ValueError("No valid texts to embed")
        
        if self.provider == 'local':
            model = self._get_sentence_transformer()
            embeddings = model.encode(valid_texts, convert_to_numpy=True, show_progress_bar=False)
            return [e.tolist() for e in embeddings]
        else:
            # OpenAI provider
            embeddings = self._get_openai_embeddings()
            return embeddings.embed_documents(valid_texts)
    
    def process_documents(
        self,
        documents: List[ProcessedDocument],
        show_progress: bool = True
    ) -> List[EmbeddedDocument]:
        """
        Process multiple documents and generate embeddings.
        
        Args:
            documents: List of ProcessedDocument objects
            show_progress: Whether to show progress
            
        Returns:
            List of EmbeddedDocument objects with embeddings
        """
        if not documents:
            return []
        
        embedded_docs = []
        total = len(documents)
        
        # Process in batches
        for i in range(0, total, self.batch_size):
            batch = documents[i:i + self.batch_size]
            batch_texts = [doc.content for doc in batch]
            
            try:
                batch_embeddings = self.generate_embeddings_batch(batch_texts)
                
                # Create embedded documents
                for doc, embedding in zip(batch, batch_embeddings):
                    embedded_doc = EmbeddedDocument(
                        content=doc.content,
                        embedding=embedding,
                        metadata=doc.metadata,
                        document_id=doc.document_id,
                        chunk_index=doc.chunk_index
                    )
                    embedded_docs.append(embedded_doc)
                
                if show_progress:
                    processed = min(i + self.batch_size, total)
                    logger.info(f"Embedded {processed}/{total} documents")
                    
            except Exception as e:
                logger.error(f"Error embedding batch {i//self.batch_size + 1}: {str(e)}")
                # Continue with next batch
                continue
        
        return embedded_docs
    
    async def agenerate_embedding(self, text: str) -> List[float]:
        """
        Async version of generate_embedding.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_embedding, text)
    
    async def agenerate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Async version of generate_embeddings_batch.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate_embeddings_batch, texts)
    
    async def aprocess_documents(
        self,
        documents: List[ProcessedDocument],
        show_progress: bool = True
    ) -> List[EmbeddedDocument]:
        """
        Async version of process_documents.
        
        Args:
            documents: List of ProcessedDocument objects
            show_progress: Whether to show progress
            
        Returns:
            List of EmbeddedDocument objects
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self.process_documents, 
            documents, 
            show_progress
        )
    
    @property
    def embedding_dimensions(self) -> int:
        """Get the embedding dimensions."""
        return self.dimensions
    
    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        import numpy as np
        
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))


def get_embedding_generator(warm_up: bool = True) -> EmbeddingGenerator:
    """
    Get the singleton embedding generator instance.
    
    OPTIMIZATION: Reuses a single instance to avoid repeated model loading.
    Optionally warms up the model with a test embedding.
    
    Args:
        warm_up: If True, warms up the model on first access
        
    Returns:
        EmbeddingGenerator singleton instance
    """
    global _embedding_generator_instance
    
    if _embedding_generator_instance is None:
        with _instance_lock:
            if _embedding_generator_instance is None:
                logger.info("Initializing singleton EmbeddingGenerator...")
                _embedding_generator_instance = EmbeddingGenerator(
                    model=settings.embedding_model,
                    dimensions=settings.embedding_dimensions,
                    batch_size=100
                )
                
                if warm_up:
                    try:
                        # Warm up with a test embedding
                        _embedding_generator_instance.generate_embedding(
                            "JD Jones industrial sealing solutions warm-up test"
                        )
                        logger.info("EmbeddingGenerator warmed up successfully")
                    except Exception as e:
                        logger.warning(f"EmbeddingGenerator warm-up failed: {e}")
    
    return _embedding_generator_instance


@lru_cache(maxsize=1)
def get_sentence_transformer_cached(model_name: str):
    """
    Get a cached SentenceTransformer model.
    Thread-safe via lru_cache.
    """
    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        return SentenceTransformer(model_name)
    except ImportError:
        raise ImportError("sentence-transformers not installed")

