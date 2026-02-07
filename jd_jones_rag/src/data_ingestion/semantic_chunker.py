"""
Semantic Chunker
Breaks text at topic boundaries rather than fixed character counts.
Provides better context preservation for RAG systems.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
import re
import numpy as np

from langchain_core.documents import Document

from src.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class SemanticChunk:
    """A semantically coherent text chunk."""
    content: str
    start_index: int
    end_index: int
    topic_label: Optional[str] = None
    coherence_score: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SemanticChunker:
    """
    Semantic chunking that breaks text at topic boundaries.
    
    Unlike fixed-size chunking, this approach:
    1. Identifies natural topic breaks in the text
    2. Groups related sentences together
    3. Maintains semantic coherence within chunks
    4. Preserves context for better retrieval
    
    Methods:
    - Sentence embedding similarity
    - Topic modeling (simplified)
    - Structural markers (headers, paragraphs)
    """
    
    # Structural break markers
    SECTION_MARKERS = [
        r'^#{1,6}\s+',           # Markdown headers
        r'^[A-Z][A-Z\s]{2,}:',   # ALL CAPS headers
        r'^\d+\.\s+[A-Z]',       # Numbered sections
        r'^[IVX]+\.\s+',         # Roman numeral sections
        r'^•\s+',                # Bullet points
        r'^\*\*[^*]+\*\*$',      # Bold headers
        r'^---+$',               # Horizontal rules
        r'^\s*$',                # Empty lines (paragraph breaks)
    ]
    
    def __init__(
        self,
        max_chunk_size: int = 1500,
        min_chunk_size: int = 200,
        similarity_threshold: float = 0.75,
        use_embeddings: bool = True
    ):
        """
        Initialize semantic chunker.
        
        Args:
            max_chunk_size: Maximum characters per chunk
            min_chunk_size: Minimum characters per chunk
            similarity_threshold: Similarity threshold for topic continuity
            use_embeddings: Whether to use embeddings for similarity
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.similarity_threshold = similarity_threshold
        self.use_embeddings = use_embeddings
        
        # Lazy initialize embedding generator
        self._embedding_generator = None
    
    def _get_embedding_generator(self):
        """Lazy initialization of embedding generator."""
        if self._embedding_generator is None:
            from src.data_ingestion.embedding_generator import EmbeddingGenerator
            self._embedding_generator = EmbeddingGenerator()
        return self._embedding_generator
    
    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[SemanticChunk]:
        """
        Split text into semantic chunks.
        
        Args:
            text: Text to chunk
            metadata: Metadata to attach to chunks
            
        Returns:
            List of SemanticChunk objects
        """
        if not text or not text.strip():
            return []
        
        # Step 1: Split into sentences/segments
        segments = self._split_into_segments(text)
        
        if not segments:
            return [SemanticChunk(
                content=text,
                start_index=0,
                end_index=len(text),
                metadata=metadata or {}
            )]
        
        # Step 2: Find topic boundaries
        if self.use_embeddings and len(segments) > 2:
            boundaries = self._find_semantic_boundaries(segments)
        else:
            boundaries = self._find_structural_boundaries(segments)
        
        # Step 3: Create chunks from boundaries
        chunks = self._create_chunks(text, segments, boundaries, metadata)
        
        # Step 4: Post-process (merge small, split large)
        chunks = self._post_process_chunks(chunks)
        
        return chunks
    
    def _split_into_segments(self, text: str) -> List[Dict[str, Any]]:
        """Split text into sentence-like segments."""
        # First try paragraph-level splits
        paragraphs = re.split(r'\n\s*\n', text)
        
        segments = []
        current_pos = 0
        
        for para in paragraphs:
            if not para.strip():
                current_pos += len(para) + 2
                continue
            
            # Split paragraph into sentences
            sentences = re.split(r'(?<=[.!?])\s+', para)
            
            for sent in sentences:
                if sent.strip():
                    start = text.find(sent, current_pos)
                    if start == -1:
                        start = current_pos
                    
                    segments.append({
                        "text": sent.strip(),
                        "start": start,
                        "end": start + len(sent),
                        "is_header": self._is_header(sent)
                    })
                    current_pos = start + len(sent)
        
        return segments
    
    def _is_header(self, text: str) -> bool:
        """Check if text looks like a header."""
        for pattern in self.SECTION_MARKERS[:6]:  # Check header patterns
            if re.match(pattern, text.strip()):
                return True
        
        # Short lines that are title-case might be headers
        if len(text) < 100 and text.strip().istitle():
            return True
        
        return False
    
    def _find_semantic_boundaries(
        self,
        segments: List[Dict[str, Any]]
    ) -> List[int]:
        """Find topic boundaries using embedding similarity."""
        if len(segments) < 3:
            return []
        
        # Get embeddings for each segment
        texts = [s["text"] for s in segments]
        
        try:
            generator = self._get_embedding_generator()
            embeddings = generator.generate_embeddings_batch(texts)
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return self._find_structural_boundaries(segments)
        
        # Calculate similarity between consecutive segments
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)
        
        # Find boundaries where similarity drops below threshold
        boundaries = []
        for i, sim in enumerate(similarities):
            # Also check if this is a structural boundary (header)
            is_header = segments[i + 1].get("is_header", False)
            
            if sim < self.similarity_threshold or is_header:
                boundaries.append(i + 1)
        
        return boundaries
    
    def _find_structural_boundaries(
        self,
        segments: List[Dict[str, Any]]
    ) -> List[int]:
        """Find boundaries using structural markers."""
        boundaries = []
        
        for i, segment in enumerate(segments):
            if segment.get("is_header", False):
                boundaries.append(i)
        
        return boundaries
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    def _create_chunks(
        self,
        text: str,
        segments: List[Dict[str, Any]],
        boundaries: List[int],
        metadata: Optional[Dict[str, Any]]
    ) -> List[SemanticChunk]:
        """Create chunks from segments and boundaries."""
        chunks = []
        
        # Add implicit boundaries at start and end
        all_boundaries = [0] + sorted(set(boundaries)) + [len(segments)]
        
        for i in range(len(all_boundaries) - 1):
            start_idx = all_boundaries[i]
            end_idx = all_boundaries[i + 1]
            
            if start_idx >= len(segments):
                break
            
            # Get segments in this chunk
            chunk_segments = segments[start_idx:end_idx]
            
            if not chunk_segments:
                continue
            
            # Build chunk content
            chunk_text = " ".join(s["text"] for s in chunk_segments)
            
            # Get positions
            start_pos = chunk_segments[0]["start"]
            end_pos = chunk_segments[-1]["end"]
            
            # Determine topic label from first segment if it's a header
            topic_label = None
            if chunk_segments[0].get("is_header"):
                topic_label = chunk_segments[0]["text"][:100]
            
            chunks.append(SemanticChunk(
                content=chunk_text,
                start_index=start_pos,
                end_index=end_pos,
                topic_label=topic_label,
                metadata=metadata.copy() if metadata else {}
            ))
        
        return chunks
    
    def _post_process_chunks(
        self,
        chunks: List[SemanticChunk]
    ) -> List[SemanticChunk]:
        """Post-process chunks: merge small, split large."""
        if not chunks:
            return chunks
        
        processed = []
        buffer = None
        
        for chunk in chunks:
            if buffer is None:
                buffer = chunk
                continue
            
            combined_length = len(buffer.content) + len(chunk.content)
            
            # If buffer is too small and combined fits, merge
            if len(buffer.content) < self.min_chunk_size and combined_length <= self.max_chunk_size:
                buffer = SemanticChunk(
                    content=buffer.content + " " + chunk.content,
                    start_index=buffer.start_index,
                    end_index=chunk.end_index,
                    topic_label=buffer.topic_label or chunk.topic_label,
                    metadata=buffer.metadata
                )
            else:
                # Process buffer if it's too large
                if len(buffer.content) > self.max_chunk_size:
                    processed.extend(self._split_large_chunk(buffer))
                else:
                    processed.append(buffer)
                buffer = chunk
        
        # Don't forget the last buffer
        if buffer:
            if len(buffer.content) > self.max_chunk_size:
                processed.extend(self._split_large_chunk(buffer))
            else:
                processed.append(buffer)
        
        return processed
    
    def _split_large_chunk(self, chunk: SemanticChunk) -> List[SemanticChunk]:
        """Split a chunk that exceeds max size."""
        text = chunk.content
        result = []
        
        # Split at sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_text = ""
        current_start = chunk.start_index
        
        for sent in sentences:
            if len(current_text) + len(sent) > self.max_chunk_size:
                if current_text:
                    result.append(SemanticChunk(
                        content=current_text.strip(),
                        start_index=current_start,
                        end_index=current_start + len(current_text),
                        topic_label=chunk.topic_label,
                        metadata=chunk.metadata.copy()
                    ))
                current_text = sent
                current_start = current_start + len(current_text)
            else:
                current_text += " " + sent if current_text else sent
        
        if current_text:
            result.append(SemanticChunk(
                content=current_text.strip(),
                start_index=current_start,
                end_index=chunk.end_index,
                topic_label=chunk.topic_label,
                metadata=chunk.metadata.copy()
            ))
        
        return result if result else [chunk]
    
    def chunk_documents(
        self,
        documents: List[Document]
    ) -> List[Document]:
        """
        Chunk a list of LangChain Documents.
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects
        """
        chunked_docs = []
        
        for doc in documents:
            chunks = self.chunk_text(
                text=doc.page_content,
                metadata=doc.metadata.copy()
            )
            
            for i, chunk in enumerate(chunks):
                chunk_metadata = chunk.metadata.copy()
                chunk_metadata.update({
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "topic_label": chunk.topic_label,
                    "semantic_chunked": True
                })
                
                chunked_docs.append(Document(
                    page_content=chunk.content,
                    metadata=chunk_metadata
                ))
        
        return chunked_docs
