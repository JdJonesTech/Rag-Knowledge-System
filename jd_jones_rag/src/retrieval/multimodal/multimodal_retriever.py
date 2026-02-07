"""
Multi-Modal Retriever
Combines text and image retrieval for comprehensive search.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from src.retrieval.multimodal.image_processor import ImageProcessor, ProcessedImage
from src.retrieval.multimodal.visual_embedder import VisualEmbedder

logger = logging.getLogger(__name__)


@dataclass
class MultiModalResult:
    """Result from multi-modal retrieval."""
    content_type: str  # "text", "image", "mixed"
    text_content: Optional[str]
    image_content: Optional[ProcessedImage]
    relevance_score: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "content_type": self.content_type,
            "text_content": self.text_content,
            "relevance_score": self.relevance_score,
            "metadata": self.metadata
        }
        
        if self.image_content:
            result["image"] = {
                "id": self.image_content.image_id,
                "width": self.image_content.width,
                "height": self.image_content.height,
                "format": self.image_content.format,
                "thumbnail": self.image_content.base64_thumbnail,
                "detected_objects": self.image_content.detected_objects
            }
        
        return result


@dataclass
class ImageDocument:
    """Document containing image and associated text."""
    image_id: str
    image_path: str
    text_description: str
    image_embedding: Optional[np.ndarray]
    processed_image: Optional[ProcessedImage]
    metadata: Dict[str, Any]


class MultiModalRetriever:
    """
    Multi-modal retriever combining text and image search.
    
    Features:
    - Text-to-image search using CLIP embeddings
    - Image-to-image similarity search
    - Combined text+image document retrieval
    - Product image matching for JD Jones catalog
    """
    
    def __init__(
        self,
        image_dir: Optional[str] = None,
        vector_store = None  # ChromaDB or similar
    ):
        """
        Initialize multi-modal retriever.
        
        Args:
            image_dir: Directory containing images to index
            vector_store: Optional vector store for image embeddings
        """
        self.image_processor = ImageProcessor()
        self.visual_embedder = VisualEmbedder()
        self.image_dir = Path(image_dir) if image_dir else None
        self.vector_store = vector_store
        
        # In-memory image index
        self._image_documents: Dict[str, ImageDocument] = {}
        self._image_embeddings: Dict[str, np.ndarray] = {}
        
        logger.info("MultiModalRetriever initialized")
    
    def index_image(
        self,
        image_path: str,
        text_description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Index a single image.
        
        Args:
            image_path: Path to image file
            text_description: Optional text description
            metadata: Optional metadata
            
        Returns:
            Image ID
        """
        # Process image
        processed = self.image_processor.process_image(image_path)
        
        # Generate description if not provided
        if not text_description:
            text_description = self.image_processor.image_to_description(processed)
        
        # Create embedding
        embedding = self.visual_embedder.embed_image(image_path)
        
        # Create document
        doc = ImageDocument(
            image_id=processed.image_id,
            image_path=image_path,
            text_description=text_description,
            image_embedding=embedding,
            processed_image=processed,
            metadata=metadata or {}
        )
        
        # Store
        self._image_documents[doc.image_id] = doc
        self._image_embeddings[doc.image_id] = embedding
        
        logger.info(f"Indexed image: {processed.image_id}")
        return processed.image_id
    
    def index_directory(self, directory: str) -> Dict[str, Any]:
        """
        Index all images in a directory.
        
        Args:
            directory: Directory path
            
        Returns:
            Indexing statistics
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        indexed = 0
        failed = 0
        
        # Find all images
        for ext in self.image_processor.SUPPORTED_FORMATS:
            for image_path in dir_path.glob(f"**/*{ext}"):
                try:
                    self.index_image(str(image_path))
                    indexed += 1
                except Exception as e:
                    logger.warning(f"Failed to index {image_path}: {e}")
                    failed += 1
        
        stats = {
            "directory": directory,
            "indexed": indexed,
            "failed": failed,
            "total_images": len(self._image_documents)
        }
        
        logger.info(f"Indexed directory: {stats}")
        return stats
    
    def search_by_text(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[MultiModalResult]:
        """
        Search images by text query.
        
        Args:
            query: Text search query
            top_k: Number of results to return
            min_score: Minimum similarity score
            
        Returns:
            List of matching results
        """
        if not self._image_documents:
            return []
        
        # Create text embedding
        query_embedding = self.visual_embedder.embed_text(query)
        
        # Calculate similarities
        similarities = []
        for image_id, image_embedding in self._image_embeddings.items():
            score = self.visual_embedder.similarity(image_embedding, query_embedding)
            if score >= min_score:
                similarities.append((image_id, score))
        
        # Sort by score
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Build results
        results = []
        for image_id, score in similarities[:top_k]:
            doc = self._image_documents[image_id]
            results.append(MultiModalResult(
                content_type="image",
                text_content=doc.text_description,
                image_content=doc.processed_image,
                relevance_score=score,
                metadata=doc.metadata
            ))
        
        return results
    
    def search_by_image(
        self,
        query_image_path: str,
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[MultiModalResult]:
        """
        Search similar images by image query.
        
        Args:
            query_image_path: Path to query image
            top_k: Number of results to return
            min_score: Minimum similarity score
            
        Returns:
            List of similar images
        """
        if not self._image_documents:
            return []
        
        # Create image embedding
        query_embedding = self.visual_embedder.embed_image(query_image_path)
        
        # Calculate similarities
        similarities = []
        for image_id, image_embedding in self._image_embeddings.items():
            score = self.visual_embedder.similarity(image_embedding, query_embedding)
            if score >= min_score:
                similarities.append((image_id, score))
        
        # Sort by score
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Build results
        results = []
        for image_id, score in similarities[:top_k]:
            doc = self._image_documents[image_id]
            results.append(MultiModalResult(
                content_type="image",
                text_content=doc.text_description,
                image_content=doc.processed_image,
                relevance_score=score,
                metadata=doc.metadata
            ))
        
        return results
    
    def search_mixed(
        self,
        query: str,
        text_results: List[Dict[str, Any]],
        image_weight: float = 0.3,
        top_k: int = 10
    ) -> List[MultiModalResult]:
        """
        Combined search merging text and image results.
        
        Args:
            query: Search query
            text_results: Text retrieval results
            image_weight: Weight for image results (0-1)
            top_k: Total results to return
            
        Returns:
            Mixed results sorted by relevance
        """
        results = []
        
        # Add text results
        text_weight = 1.0 - image_weight
        for text_result in text_results:
            score = text_result.get("relevance_score", 0) * text_weight
            results.append(MultiModalResult(
                content_type="text",
                text_content=text_result.get("content", ""),
                image_content=None,
                relevance_score=score,
                metadata=text_result.get("metadata", {})
            ))
        
        # Add image results
        image_results = self.search_by_text(query, top_k=top_k)
        for img_result in image_results:
            img_result.relevance_score *= image_weight
            results.append(img_result)
        
        # Sort by combined score
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return results[:top_k]
    
    def find_product_images(
        self,
        product_code: str,
        include_related: bool = True
    ) -> List[MultiModalResult]:
        """
        Find images related to a specific product code.
        
        Args:
            product_code: Product code (e.g., "NA 701")
            include_related: Include images of related products
            
        Returns:
            Product images
        """
        results = []
        
        # Normalize product code
        normalized = product_code.upper().replace("-", " ").strip()
        
        # Search by text
        text_results = self.search_by_text(normalized, top_k=10)
        
        # Also search in metadata
        for image_id, doc in self._image_documents.items():
            # Check if product code in metadata or description
            if normalized.lower() in doc.text_description.lower():
                results.append(MultiModalResult(
                    content_type="image",
                    text_content=doc.text_description,
                    image_content=doc.processed_image,
                    relevance_score=1.0,  # Exact match
                    metadata=doc.metadata
                ))
            elif any(
                normalized.lower() in str(v).lower() 
                for v in doc.metadata.values()
            ):
                results.append(MultiModalResult(
                    content_type="image",
                    text_content=doc.text_description,
                    image_content=doc.processed_image,
                    relevance_score=0.9,  # Metadata match
                    metadata=doc.metadata
                ))
        
        # Merge with text search results
        result_ids = {r.image_content.image_id for r in results if r.image_content}
        for text_result in text_results:
            if text_result.image_content and text_result.image_content.image_id not in result_ids:
                results.append(text_result)
        
        # Sort by relevance
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return results
    
    def get_image_document(self, image_id: str) -> Optional[ImageDocument]:
        """Get indexed image document by ID."""
        return self._image_documents.get(image_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        return {
            "total_images": len(self._image_documents),
            "embedding_dimension": self.visual_embedder.dimension,
            "index_size_mb": sum(
                e.nbytes for e in self._image_embeddings.values()
            ) / (1024 * 1024)
        }
