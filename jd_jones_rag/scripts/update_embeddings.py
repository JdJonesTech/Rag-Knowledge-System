#!/usr/bin/env python3
"""
Embedding Update Script
Re-generates embeddings for existing documents in the knowledge base.

Usage:
    python scripts/update_embeddings.py --collection main_internal
    python scripts/update_embeddings.py --all
    python scripts/update_embeddings.py --collection dept_sales --batch-size 50
"""

import argparse
import asyncio
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import settings
from src.data_ingestion.embedding_generator import EmbeddingGenerator
from src.data_ingestion.vector_store import VectorStoreManager


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Update embeddings in the JD Jones RAG knowledge base"
    )
    
    parser.add_argument(
        "--collection", "-c",
        type=str,
        default=None,
        help="Collection to update (or use --all)"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Update all collections"
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=100,
        help="Batch size for embedding generation"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without making changes"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Embedding model to use (default: from settings)"
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
    print("JD Jones RAG - Embedding Update")
    print("=" * 60)
    print()


async def update_collection(
    collection_name: str,
    embedding_generator: EmbeddingGenerator,
    vector_store: VectorStoreManager,
    batch_size: int,
    dry_run: bool,
    verbose: bool
) -> dict:
    """
    Update embeddings for a single collection.
    
    Returns:
        Dict with update statistics
    """
    stats = {
        "collection": collection_name,
        "documents_processed": 0,
        "embeddings_updated": 0,
        "errors": []
    }
    
    try:
        # Get collection count
        count = vector_store.get_collection_count(collection_name)
        
        if verbose:
            print(f"\nCollection: {collection_name}")
            print(f"  Documents: {count}")
        
        if count == 0:
            if verbose:
                print("  Skipping (empty collection)")
            return stats
        
        if dry_run:
            print(f"  Would update {count} documents")
            stats["documents_processed"] = count
            return stats
        
        # Get all documents from collection
        # Note: In production, this should be paginated
        collection = vector_store.get_collection(collection_name)
        
        if not collection:
            stats["errors"].append(f"Collection not found: {collection_name}")
            return stats
        
        # Get all document data
        results = collection.get(include=["documents", "metadatas"])
        
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        ids = results.get("ids", [])
        
        if verbose:
            print(f"  Processing {len(documents)} documents...")
        
        # Process in batches
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]
            
            try:
                # Generate new embeddings
                embeddings = embedding_generator.generate_embeddings_batch(
                    batch_docs,
                    show_progress=verbose
                )
                
                # Update in collection
                collection.update(
                    ids=batch_ids,
                    embeddings=embeddings
                )
                
                stats["embeddings_updated"] += len(embeddings)
                
                if verbose:
                    print(f"  Updated batch {i // batch_size + 1}: {len(embeddings)} embeddings")
                
            except Exception as e:
                stats["errors"].append(f"Batch {i}: {str(e)}")
                if verbose:
                    print(f"  Error in batch {i}: {e}")
        
        stats["documents_processed"] = len(documents)
        
    except Exception as e:
        stats["errors"].append(str(e))
        if verbose:
            print(f"  Error: {e}")
    
    return stats


async def main():
    """Main update function."""
    args = parse_args()
    print_banner()
    
    # Validate arguments
    if not args.collection and not args.all:
        print("Error: Must specify --collection or --all")
        sys.exit(1)
    
    # Print configuration
    print("Configuration:")
    print(f"  Collection:  {args.collection or 'ALL'}")
    print(f"  Batch Size:  {args.batch_size}")
    print(f"  Model:       {args.model or settings.embedding_model}")
    print(f"  Dry Run:     {args.dry_run}")
    print()
    
    # Initialize components
    embedding_generator = EmbeddingGenerator(
        model_name=args.model or settings.embedding_model
    )
    vector_store = VectorStoreManager()
    
    # Determine collections to update
    if args.all:
        collections = vector_store.list_collections()
    else:
        collections = [args.collection]
    
    print(f"Collections to update: {len(collections)}")
    for c in collections:
        count = vector_store.get_collection_count(c)
        print(f"  - {c}: {count} documents")
    print()
    
    # Track statistics
    start_time = datetime.now()
    all_stats = []
    
    # Update each collection
    for collection_name in collections:
        stats = await update_collection(
            collection_name=collection_name,
            embedding_generator=embedding_generator,
            vector_store=vector_store,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            verbose=args.verbose
        )
        all_stats.append(stats)
    
    # Calculate totals
    elapsed = (datetime.now() - start_time).total_seconds()
    total_docs = sum(s["documents_processed"] for s in all_stats)
    total_updated = sum(s["embeddings_updated"] for s in all_stats)
    total_errors = sum(len(s["errors"]) for s in all_stats)
    
    # Print summary
    print()
    print("=" * 60)
    print("UPDATE SUMMARY")
    print("=" * 60)
    print(f"Collections processed: {len(collections)}")
    print(f"Documents processed:   {total_docs}")
    print(f"Embeddings updated:    {total_updated}")
    print(f"Errors:                {total_errors}")
    print(f"Time elapsed:          {elapsed:.2f} seconds")
    
    if total_errors > 0:
        print("\nErrors:")
        for stats in all_stats:
            for error in stats["errors"]:
                print(f"  - {stats['collection']}: {error}")
    
    print("=" * 60)
    
    if total_errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
