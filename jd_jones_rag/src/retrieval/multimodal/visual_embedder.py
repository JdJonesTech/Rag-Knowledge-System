"""
Visual Embedder for Multi-Modal RAG
Creates embeddings for images using vision models.
"""

import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
import numpy as np

try:
    from PIL import Image
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


class VisualEmbedder:
    """
    Creates embeddings for visual content.
    Uses CLIP or similar models for image embedding.
    
    Note: This is a base implementation. For production, integrate with:
    - OpenAI CLIP
    - HuggingFace sentence-transformers with CLIP
    - Custom vision embedding model
    """
    
    # Default embedding dimension
    DEFAULT_DIMENSION = 512
    
    def __init__(
        self,
        model_name: str = "clip-ViT-B-32",
        device: Optional[str] = None
    ):
        """
        Initialize visual embedder.
        
        Args:
            model_name: Vision model to use
            device: Device for inference ("cpu", "cuda")
        """
        self.model_name = model_name
        self.device = device or "cpu"
        self._model = None
        self._processor = None
        self._dimension = self.DEFAULT_DIMENSION
        
        # Lazy loading
        self._initialized = False
        
        logger.info(f"VisualEmbedder initialized with {model_name}")
    
    def _load_model(self):
        """Lazy load the embedding model."""
        if self._initialized:
            return
        
        try:
            # Try to load sentence-transformers CLIP model
            from sentence_transformers import SentenceTransformer
            
            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            self._initialized = True
            
            logger.info(f"Loaded {self.model_name} with dimension {self._dimension}")
            
        except ImportError:
            logger.warning(
                "sentence-transformers not available. "
                "Using fallback random embeddings for development."
            )
            self._initialized = True
        except Exception as e:
            logger.warning(f"Failed to load vision model: {e}")
            self._initialized = True
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        self._load_model()
        return self._dimension
    
    def embed_image(self, image_path: str) -> np.ndarray:
        """
        Create embedding for a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Numpy array of embedding
        """
        self._load_model()
        
        if self._model is not None:
            try:
                # Load and preprocess image
                with Image.open(image_path) as img:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    
                    # Encode image
                    embedding = self._model.encode(img)
                    return np.array(embedding)
                    
            except Exception as e:
                logger.error(f"Failed to embed image {image_path}: {e}")
        
        # Fallback: return random embedding for development
        return self._fallback_embedding(image_path)
    
    def embed_images(self, image_paths: List[str]) -> List[np.ndarray]:
        """
        Create embeddings for multiple images.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            List of embedding arrays
        """
        self._load_model()
        
        if self._model is not None:
            try:
                images = []
                for path in image_paths:
                    with Image.open(path) as img:
                        if img.mode != "RGB":
                            img = img.convert("RGB")
                        images.append(img.copy())
                
                # Batch encode
                embeddings = self._model.encode(images)
                return [np.array(e) for e in embeddings]
                
            except Exception as e:
                logger.error(f"Failed to batch embed images: {e}")
        
        # Fallback
        return [self._fallback_embedding(p) for p in image_paths]
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Create embedding for text (for text-to-image search).
        
        Args:
            text: Query text
            
        Returns:
            Numpy array of embedding
        """
        self._load_model()
        
        if self._model is not None:
            try:
                embedding = self._model.encode(text)
                return np.array(embedding)
            except Exception as e:
                logger.error(f"Failed to embed text: {e}")
        
        # Fallback
        return self._fallback_text_embedding(text)
    
    def similarity(
        self, 
        image_embedding: np.ndarray, 
        query_embedding: np.ndarray
    ) -> float:
        """
        Calculate cosine similarity between embeddings.
        
        Args:
            image_embedding: Image embedding
            query_embedding: Query embedding (text or image)
            
        Returns:
            Similarity score (0-1)
        """
        # Normalize
        img_norm = image_embedding / (np.linalg.norm(image_embedding) + 1e-8)
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        # Cosine similarity
        similarity = np.dot(img_norm, query_norm)
        return float(max(0, min(1, similarity)))
    
    def _fallback_embedding(self, image_path: str) -> np.ndarray:
        """Generate fallback embedding based on image characteristics."""
        # Create deterministic embedding from image path and basic properties
        np.random.seed(hash(image_path) % (2**32))
        
        embedding = np.random.randn(self._dimension).astype(np.float32)
        
        # Try to incorporate image properties
        if PILLOW_AVAILABLE:
            try:
                with Image.open(image_path) as img:
                    # Add image size info
                    w, h = img.size
                    embedding[0] = w / 1000
                    embedding[1] = h / 1000
                    embedding[2] = (w / h) if h > 0 else 1
            except Exception:
                pass
        
        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        return embedding
    
    def _fallback_text_embedding(self, text: str) -> np.ndarray:
        """Generate fallback embedding for text."""
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.randn(self._dimension).astype(np.float32)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        return embedding


class CLIPEmbedder(VisualEmbedder):
    """
    CLIP-based visual embedder for joint image-text embeddings.
    Inherits from VisualEmbedder with CLIP-specific configuration.
    """
    
    def __init__(self, model_variant: str = "ViT-B/32"):
        """
        Initialize CLIP embedder.
        
        Args:
            model_variant: CLIP model variant ("ViT-B/32", "ViT-L/14", etc.)
        """
        # Map variant to sentence-transformers model name
        model_map = {
            "ViT-B/32": "clip-ViT-B-32",
            "ViT-L/14": "clip-ViT-L-14",
            "ViT-B/16": "clip-ViT-B-16",
        }
        
        model_name = model_map.get(model_variant, "clip-ViT-B-32")
        super().__init__(model_name=model_name)
        
        logger.info(f"CLIPEmbedder initialized with {model_variant}")
