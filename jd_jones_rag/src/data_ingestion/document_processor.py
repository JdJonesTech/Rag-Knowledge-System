"""
Document processor for handling various file formats.
Processes PDF, DOCX, TXT, MD, XLSX, PPTX and extracts text with metadata.
Supports semantic chunking and hierarchical indexing.
"""

import os
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    CSVLoader
)
from langchain_core.documents import Document

from src.config.settings import settings


class AccessLevel(str, Enum):
    """Document access levels."""
    LEVEL_0_PUBLIC = "level_0_public"
    LEVEL_0_INTERNAL = "level_0_internal"
    LEVEL_1_SALES = "level_1_sales"
    LEVEL_1_PRODUCTION = "level_1_production"
    LEVEL_1_ENGINEERING = "level_1_engineering"
    LEVEL_1_CUSTOMER_SERVICE = "level_1_customer_service"
    LEVEL_2_MANAGEMENT = "level_2_management"
    LEVEL_3_CONFIDENTIAL = "level_3_confidential"


@dataclass
class ProcessedDocument:
    """Represents a processed document chunk with metadata."""
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    document_id: str = ""
    chunk_index: int = 0
    total_chunks: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "metadata": self.metadata,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks
        }


class DocumentProcessor:
    """
    Processes various document formats into chunks with metadata.
    Supports PDF, DOCX, TXT, MD, XLSX, PPTX, CSV files.
    
    Features:
    - Standard recursive text splitting
    - Semantic chunking (topic-based splits)
    - Hierarchical indexing integration
    """
    
    # File extension to loader mapping
    LOADERS = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".doc": Docx2txtLoader,
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
        ".xlsx": UnstructuredExcelLoader,
        ".xls": UnstructuredExcelLoader,
        ".pptx": UnstructuredPowerPointLoader,
        ".csv": CSVLoader,
    }
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        use_semantic_chunking: bool = False,
        hierarchical_indexer = None
    ):
        """
        Initialize document processor.
        
        Args:
            chunk_size: Size of text chunks (default from settings)
            chunk_overlap: Overlap between chunks (default from settings)
            use_semantic_chunking: Use semantic chunking instead of fixed-size
            hierarchical_indexer: Optional HierarchicalIndexer for categorization
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.use_semantic_chunking = use_semantic_chunking
        self.hierarchical_indexer = hierarchical_indexer
        
        # Standard text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Semantic chunker (lazy initialization)
        self._semantic_chunker = None
    
    def _get_semantic_chunker(self):
        """Lazy initialization of semantic chunker."""
        if self._semantic_chunker is None and self.use_semantic_chunking:
            from src.data_ingestion.semantic_chunker import SemanticChunker
            self._semantic_chunker = SemanticChunker()
        return self._semantic_chunker
    
    def process_file(
        self,
        file_path: str,
        access_level: AccessLevel = AccessLevel.LEVEL_0_INTERNAL,
        department: Optional[str] = None,
        additional_metadata: Optional[Dict[str, Any]] = None,
        use_semantic_chunking: Optional[bool] = None,
        auto_categorize: bool = False
    ) -> List[ProcessedDocument]:
        """
        Process a single file into chunks with metadata.
        
        Args:
            file_path: Path to the file
            access_level: Access level for the document
            department: Department the document belongs to
            additional_metadata: Additional metadata to attach
            use_semantic_chunking: Override instance setting for semantic chunking
            auto_categorize: Use hierarchical indexer to suggest category
            
        Returns:
            List of ProcessedDocument objects
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = path.suffix.lower()
        
        if extension not in self.LOADERS:
            raise ValueError(f"Unsupported file format: {extension}")
        
        # Load document
        loader_class = self.LOADERS[extension]
        
        try:
            if extension == ".csv":
                loader = loader_class(file_path, encoding="utf-8")
            else:
                loader = loader_class(file_path)
            
            documents = loader.load()
        except Exception as e:
            raise RuntimeError(f"Error loading file {file_path}: {str(e)}")
        
        # Generate document ID
        document_id = self._generate_document_id(file_path)
        
        # Get file metadata
        file_stats = path.stat()
        base_metadata = {
            "document_id": document_id,
            "file_name": path.name,
            "file_path": str(path.absolute()),
            "file_type": extension[1:],  # Remove dot
            "file_size": file_stats.st_size,
            "access_level": access_level.value,
            "department": department,
            "created_at": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
            "processed_at": datetime.now().isoformat(),
        }
        
        # Merge additional metadata
        if additional_metadata:
            base_metadata.update(additional_metadata)
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Create processed documents
        processed_docs = []
        total_chunks = len(chunks)
        
        for idx, chunk in enumerate(chunks):
            # Merge chunk metadata with base metadata
            chunk_metadata = {**base_metadata}
            chunk_metadata["chunk_index"] = idx
            chunk_metadata["total_chunks"] = total_chunks
            
            # Add any metadata from the loader
            if chunk.metadata:
                chunk_metadata["source_metadata"] = chunk.metadata
            
            processed_doc = ProcessedDocument(
                content=chunk.page_content,
                metadata=chunk_metadata,
                document_id=document_id,
                chunk_index=idx,
                total_chunks=total_chunks
            )
            processed_docs.append(processed_doc)
        
        return processed_docs
    
    def process_directory(
        self,
        directory_path: str,
        access_level: AccessLevel = AccessLevel.LEVEL_0_INTERNAL,
        department: Optional[str] = None,
        recursive: bool = True,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> List[ProcessedDocument]:
        """
        Process all supported files in a directory.
        
        Args:
            directory_path: Path to the directory
            access_level: Access level for all documents
            department: Department for all documents
            recursive: Whether to process subdirectories
            additional_metadata: Additional metadata to attach
            
        Returns:
            List of all ProcessedDocument objects
        """
        path = Path(directory_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if not path.is_dir():
            raise ValueError(f"Not a directory: {directory_path}")
        
        all_documents = []
        
        # Get all files
        pattern = "**/*" if recursive else "*"
        
        for file_path in path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in self.LOADERS:
                try:
                    docs = self.process_file(
                        str(file_path),
                        access_level=access_level,
                        department=department,
                        additional_metadata=additional_metadata
                    )
                    all_documents.extend(docs)
                except Exception as e:
                    logger.warning(f"Failed to process {file_path}: {str(e)}")
                    continue
        
        return all_documents
    
    def process_text(
        self,
        text: str,
        source_name: str,
        access_level: AccessLevel = AccessLevel.LEVEL_0_INTERNAL,
        department: Optional[str] = None,
        additional_metadata: Optional[Dict[str, Any]] = None
    ) -> List[ProcessedDocument]:
        """
        Process raw text into chunks.
        
        Args:
            text: Raw text content
            source_name: Name/identifier for the text source
            access_level: Access level for the document
            department: Department the document belongs to
            additional_metadata: Additional metadata to attach
            
        Returns:
            List of ProcessedDocument objects
        """
        document_id = self._generate_document_id(source_name + text[:100])
        
        base_metadata = {
            "document_id": document_id,
            "source_name": source_name,
            "file_type": "text",
            "access_level": access_level.value,
            "department": department,
            "processed_at": datetime.now().isoformat(),
        }
        
        if additional_metadata:
            base_metadata.update(additional_metadata)
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        processed_docs = []
        total_chunks = len(chunks)
        
        for idx, chunk_text in enumerate(chunks):
            chunk_metadata = {**base_metadata}
            chunk_metadata["chunk_index"] = idx
            chunk_metadata["total_chunks"] = total_chunks
            
            processed_doc = ProcessedDocument(
                content=chunk_text,
                metadata=chunk_metadata,
                document_id=document_id,
                chunk_index=idx,
                total_chunks=total_chunks
            )
            processed_docs.append(processed_doc)
        
        return processed_docs
    
    def _generate_document_id(self, content: str) -> str:
        """Generate a unique document ID based on content hash."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Get list of supported file extensions."""
        return list(cls.LOADERS.keys())
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Estimate token count for text.
        Rough estimation: ~4 characters per token for English.
        """
        return len(text) // 4
