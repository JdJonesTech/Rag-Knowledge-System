#!/usr/bin/env python3
"""
Document Ingestion Script
Processes and ingests documents into the knowledge base.

Usage:
    python scripts/ingest_documents.py --source /path/to/docs --collection main_internal
    python scripts/ingest_documents.py --source /path/to/file.pdf --collection dept_sales --department sales
    python scripts/ingest_documents.py --source /path/to/docs --collection main_public --public
"""

import argparse
import asyncio
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import settings
from src.data_ingestion.document_processor import DocumentProcessor, AccessLevel
from src.data_ingestion.embedding_generator import EmbeddingGenerator
from src.data_ingestion.vector_store import VectorStoreManager


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into the JD Jones RAG knowledge base"
    )
    
    parser.add_argument(
        "--source", "-s",
        type=str,
        required=True,
        help="Path to file or directory to ingest"
    )
    
    parser.add_argument(
        "--collection", "-c",
        type=str,
        required=True,
        help="Target collection name (e.g., main_internal, dept_sales)"
    )
    
    parser.add_argument(
        "--department", "-d",
        type=str,
        default=None,
        help="Department name for level 1+ documents"
    )
    
    parser.add_argument(
        "--access-level", "-a",
        type=str,
        default="level_0_internal",
        choices=["level_0_public", "level_0_internal", "level_1", "level_2", "level_3"],
        help="Access level for documents"
    )
    
    parser.add_argument(
        "--public",
        action="store_true",
        help="Mark documents as public (shortcut for --access-level level_0_public)"
    )
    
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        default=True,
        help="Recursively process directories (default: True)"
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=100,
        help="Batch size for embedding generation"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Custom chunk size (default: from settings)"
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Custom chunk overlap (default: from settings)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process documents but don't store them"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    return parser.parse_args()


def print_banner():
    """Print script banner."""
    print("=" * 60)
    print("JD Jones RAG - Document Ingestion")
    print("=" * 60)
    print()


def print_summary(
    total_files: int,
    total_chunks: int,
    total_embedded: int,
    total_stored: int,
    elapsed: float,
    errors: list
):
    """Print ingestion summary."""
    print()
    print("=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    print(f"Files processed:    {total_files}")
    print(f"Chunks created:     {total_chunks}")
    print(f"Embeddings created: {total_embedded}")
    print(f"Documents stored:   {total_stored}")
    print(f"Time elapsed:       {elapsed:.2f} seconds")
    
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for error in errors[:10]:
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    
    print("=" * 60)


async def main():
    """Main ingestion function."""
    args = parse_args()
    print_banner()
    
    # Validate source path
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"Error: Source path does not exist: {source_path}")
        sys.exit(1)
    
    # Determine access level
    if args.public:
        access_level = AccessLevel.LEVEL_0_PUBLIC
    else:
        access_level = AccessLevel(args.access_level)
    
    # Print configuration
    print("Configuration:")
    print(f"  Source:       {source_path}")
    print(f"  Collection:   {args.collection}")
    print(f"  Department:   {args.department or 'None'}")
    print(f"  Access Level: {access_level.value}")
    print(f"  Recursive:    {args.recursive}")
    print(f"  Dry Run:      {args.dry_run}")
    print()
    
    # Initialize components
    processor = DocumentProcessor(
        chunk_size=args.chunk_size or settings.chunk_size,
        chunk_overlap=args.chunk_overlap or settings.chunk_overlap
    )
    embedding_generator = EmbeddingGenerator()
    vector_store = VectorStoreManager()
    
    # Track statistics
    start_time = datetime.now()
    total_files = 0
    total_chunks = 0
    total_embedded = 0
    total_stored = 0
    errors = []
    
    # Process documents
    try:
        if source_path.is_file():
            # Single file
            files_to_process = [source_path]
        else:
            # Directory - get all supported files
            extensions = settings.allowed_extensions_list
            if args.recursive:
                files_to_process = [
                    f for f in source_path.rglob("*")
                    if f.suffix.lower() in extensions
                ]
            else:
                files_to_process = [
                    f for f in source_path.glob("*")
                    if f.suffix.lower() in extensions
                ]
        
        print(f"Found {len(files_to_process)} files to process")
        print()
        
        # Process each file
        for file_path in files_to_process:
            try:
                if args.verbose:
                    print(f"Processing: {file_path.name}")
                
                # Process document
                processed_docs = processor.process_file(
                    str(file_path),
                    access_level=access_level,
                    department=args.department,
                    additional_metadata={
                        "ingestion_date": datetime.now().isoformat(),
                        "source_path": str(file_path)
                    }
                )
                
                total_files += 1
                total_chunks += len(processed_docs)
                
                if args.verbose:
                    print(f"  Created {len(processed_docs)} chunks")
                
                if args.dry_run:
                    continue
                
                # Generate embeddings
                embedded_docs = embedding_generator.process_documents(
                    processed_docs,
                    batch_size=args.batch_size,
                    show_progress=args.verbose
                )
                total_embedded += len(embedded_docs)
                
                # Store in vector database
                stored = vector_store.add_documents(args.collection, embedded_docs)
                total_stored += stored
                
                if args.verbose:
                    print(f"  Stored {stored} documents")
                
            except Exception as e:
                error_msg = f"{file_path.name}: {str(e)}"
                errors.append(error_msg)
                if args.verbose:
                    print(f"  Error: {e}")
        
    except KeyboardInterrupt:
        print("\nIngestion interrupted by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
    
    # Calculate elapsed time
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # Print summary
    print_summary(
        total_files=total_files,
        total_chunks=total_chunks,
        total_embedded=total_embedded,
        total_stored=total_stored,
        elapsed=elapsed,
        errors=errors
    )
    
    # Exit with error code if there were errors
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
