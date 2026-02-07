"""
JD Jones RAG System - Free Local Setup Verification
Verifies that all free local components are working.
"""

import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def print_header():
    logger.info("-" * 60)
    logger.info("JD JONES RAG SYSTEM - FREE LOCAL SETUP")
    logger.info("-" * 60)

def check_ollama():
    """Check if Ollama is running."""
    logger.info("Checking Ollama LLM...")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            logger.info("   Ollama is running!")
            if models:
                logger.info(f"   Available models: {[m['name'] for m in models]}")
            else:
                logger.warning("   No models installed. Run: ollama pull llama3.2")
            return True
        else:
            logger.error(f"   Ollama responded with status {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"   Ollama not running: {e}")
        logger.info("   Install from: https://ollama.ai/download")
        logger.info("   Then run: ollama pull llama3.2")
        return False

def check_sentence_transformers():
    """Check if sentence-transformers is available."""
    logger.info("Checking Local Embeddings (sentence-transformers)...")
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("   sentence-transformers installed!")
        
        # Try to load the model
        logger.info("   Loading embedding model (first time may download ~80MB)...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Test embedding
        test_embedding = model.encode("test")
        logger.info(f"   Model loaded! Embedding dimension: {len(test_embedding)}")
        return True
    except ImportError:
        logger.error("   sentence-transformers not installed")
        logger.info("   Run: pip install sentence-transformers")
        return False
    except Exception as e:
        logger.error(f"   Error loading model: {e}")
        return False

def check_cross_encoder():
    """Check if cross-encoder reranker is available."""
    logger.info("Checking Local Reranker (cross-encoder)...")
    try:
        from sentence_transformers import CrossEncoder
        logger.info("   CrossEncoder available!")
        
        # Try to load the model
        logger.info("   Loading reranker model (first time may download ~80MB)...")
        model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Test reranking
        scores = model.predict([("query", "document")])
        logger.info(f"   Reranker loaded! Test score: {scores[0]:.4f}")
        return True
    except Exception as e:
        logger.warning(f"   Reranker not available (optional): {e}")
        return False

def check_config():
    """Check configuration is set for local usage."""
    logger.info("Checking Configuration...")
    try:
        from src.config.settings import settings
        
        provider = getattr(settings, 'llm_provider', 'unknown')
        embed_provider = getattr(settings, 'embedding_provider', 'unknown')
        reranker = getattr(settings, 'reranker_provider', 'unknown')
        
        logger.info(f"   LLM Provider: {provider}")
        logger.info(f"   LLM Model: {settings.llm_model}")
        logger.info(f"   Embedding Provider: {embed_provider}")
        logger.info(f"   Embedding Model: {settings.embedding_model}")
        logger.info(f"   Reranker Provider: {reranker}")
        
        if provider == 'ollama':
            logger.info("   Configured for FREE local LLM!")
        else:
            logger.warning("   Using paid API (set LLM_PROVIDER=ollama for free)")
            
        if embed_provider == 'local':
            logger.info("   Configured for FREE local embeddings!")
        else:
            logger.warning("   Using paid embeddings (set EMBEDDING_PROVIDER=local for free)")
            
        return True
    except Exception as e:
        logger.error(f"   Error loading config: {e}")
        return False

def main():
    print_header()
    
    results = {
        "Ollama LLM": check_ollama(),
        "Embeddings": check_sentence_transformers(),
        "Reranker": check_cross_encoder(),
        "Configuration": check_config(),
    }
    
    logger.info("-" * 60)
    logger.info("SUMMARY")
    logger.info("-" * 60)
    
    all_core_ok = True
    for name, status in results.items():
        ready_text = 'Ready' if status else 'Not Ready'
        logger.info(f"   {name}: {ready_text}")
        if name in ["Ollama LLM", "Embeddings", "Configuration"] and not status:
            all_core_ok = False
    
    logger.info("-" * 60)
    
    if all_core_ok:
        logger.info("All core components ready! You can run the system.")
        logger.info("""
Next Steps:
   1. Start Docker services: docker-compose up -d
   2. Ingest data: docker-compose exec api python data/ingest_data.py
   3. Run demo: docker-compose exec api python demo_system.py
   4. Access API: http://localhost:8000/docs
        """)
    else:
        logger.warning("Some components need setup. Follow the instructions above.")
        logger.info("""
Quick Setup (if Ollama not ready):
   1. Download Ollama: https://ollama.ai/download
   2. Install and run it
   3. Pull a model: ollama pull llama3.2
   4. Run this script again
        """)
    
    return 0 if all_core_ok else 1

if __name__ == "__main__":
    sys.exit(main())
