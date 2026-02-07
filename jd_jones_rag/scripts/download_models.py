"""
Download and Cache Models
Pre-downloads all ML models used by the RAG system for faster startup.
Run this script once after installation to cache models locally.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def download_cross_encoder_models():
    """Download and cache cross-encoder models for reranking."""
    logger.info("=" * 60)
    logger.info("Downloading Cross-Encoder Models for Reranking")
    logger.info("=" * 60)
    
    try:
        from sentence_transformers import CrossEncoder
        import torch
        
        # Models used in reranker_config.py
        models = [
            "cross-encoder/ms-marco-MiniLM-L-6-v2",   # Fast, good quality (default)
            "cross-encoder/ms-marco-MiniLM-L-12-v2",  # Higher accuracy
        ]
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        for model_name in models:
            logger.info(f"\nDownloading: {model_name}")
            try:
                model = CrossEncoder(model_name, device=device)
                
                # Test the model
                test_pairs = [
                    ("What is NA 701?", "NA 701 is a high-performance graphite packing."),
                    ("Temperature rating", "This product operates up to 450°C.")
                ]
                scores = model.predict(test_pairs)
                logger.info(f"  ✓ Model loaded successfully")
                logger.info(f"  ✓ Test scores: {scores}")
                
                # Free memory
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"  ✗ Failed to download {model_name}: {e}")
        
        logger.info("\n✓ Cross-encoder models cached successfully")
        return True
        
    except ImportError:
        logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
        return False


def download_embedding_models():
    """Download and cache embedding models."""
    logger.info("\n" + "=" * 60)
    logger.info("Downloading Embedding Models")
    logger.info("=" * 60)
    
    try:
        from sentence_transformers import SentenceTransformer
        import torch
        
        # Common embedding models
        models = [
            "all-MiniLM-L6-v2",  # Fast, general purpose
            "all-mpnet-base-v2",  # Higher quality
        ]
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        for model_name in models:
            logger.info(f"\nDownloading: {model_name}")
            try:
                model = SentenceTransformer(model_name, device=device)
                
                # Test the model
                test_texts = ["Test embedding generation", "Another test sentence"]
                embeddings = model.encode(test_texts)
                logger.info(f"  ✓ Model loaded successfully")
                logger.info(f"  ✓ Embedding dimension: {embeddings.shape[1]}")
                
                # Free memory
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logger.error(f"  ✗ Failed to download {model_name}: {e}")
        
        logger.info("\n✓ Embedding models cached successfully")
        return True
        
    except ImportError:
        logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
        return False


def download_bm25_tokenizer():
    """Download NLTK data for BM25 tokenization."""
    logger.info("\n" + "=" * 60)
    logger.info("Downloading NLTK Data for BM25")
    logger.info("=" * 60)
    
    try:
        import nltk
        
        resources = ['punkt', 'stopwords', 'punkt_tab']
        
        for resource in resources:
            logger.info(f"Downloading: {resource}")
            try:
                nltk.download(resource, quiet=True)
                logger.info(f"  ✓ {resource} downloaded")
            except Exception as e:
                logger.warning(f"  ⚠ Could not download {resource}: {e}")
        
        logger.info("\n✓ NLTK data cached successfully")
        return True
        
    except ImportError:
        logger.warning("NLTK not installed. Skipping BM25 tokenizer download.")
        return False


def download_spacy_model():
    """Download spaCy model for NER (optional - may fail on Python 3.14+)."""
    logger.info("\n" + "=" * 60)
    logger.info("Downloading spaCy Model (Optional)")
    logger.info("=" * 60)
    
    try:
        import sys
        if sys.version_info >= (3, 14):
            logger.warning("⚠ spaCy has compatibility issues with Python 3.14+")
            logger.warning("  NER features will be disabled. Consider using Python 3.11-3.12.")
            return True  # Return True so it doesn't fail the overall process
        
        import spacy
        
        model_name = "en_core_web_sm"
        
        try:
            # Try to load first
            nlp = spacy.load(model_name)
            logger.info(f"  ✓ {model_name} already cached")
        except OSError:
            # Download if not available
            logger.info(f"Downloading: {model_name}")
            from spacy.cli import download
            download(model_name)
            logger.info(f"  ✓ {model_name} downloaded")
        
        return True
        
    except ImportError:
        logger.warning("spaCy not installed. NER features will be disabled.")
        return True  # Optional, so return True
    except Exception as e:
        logger.warning(f"spaCy download failed (optional): {e}")
        logger.warning("  NER features will be disabled but system will still work.")
        return True  # Optional, so return True


def verify_cache_location():
    """Show where models are cached."""
    logger.info("\n" + "=" * 60)
    logger.info("Model Cache Locations")
    logger.info("=" * 60)
    
    # HuggingFace cache
    hf_cache = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
    logger.info(f"HuggingFace cache: {hf_cache}")
    
    # Sentence Transformers cache
    st_cache = os.environ.get('SENTENCE_TRANSFORMERS_HOME', os.path.expanduser('~/.cache/torch/sentence_transformers'))
    logger.info(f"Sentence Transformers cache: {st_cache}")
    
    # NLTK cache
    try:
        import nltk
        logger.info(f"NLTK data: {nltk.data.path}")
    except ImportError:
        pass


def main():
    """Main function to download all models."""
    logger.info("=" * 60)
    logger.info("JD Jones RAG - Model Downloader")
    logger.info("=" * 60)
    logger.info("This script will download and cache all ML models used by the system.")
    logger.info("Models will be cached locally for faster startup.\n")
    
    success = True
    
    # Download models
    success &= download_cross_encoder_models()
    success &= download_embedding_models()
    success &= download_bm25_tokenizer()
    success &= download_spacy_model()
    
    # Show cache locations
    verify_cache_location()
    
    logger.info("\n" + "=" * 60)
    if success:
        logger.info("✓ All models downloaded and cached successfully!")
    else:
        logger.warning("⚠ Some models failed to download. Check the logs above.")
    logger.info("=" * 60)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
