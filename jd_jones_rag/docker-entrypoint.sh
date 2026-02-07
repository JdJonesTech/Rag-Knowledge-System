#!/bin/bash
# JD Jones RAG System - Entrypoint Script
# Downloads ML models if not already cached (using persistent volumes)

set -e

echo "=== JD Jones RAG System Starting ==="
echo "Checking for cached ML models..."

# Check and download models only if not present
python -c "
import os
import sys

def check_and_download_models():
    from pathlib import Path
    
    hf_cache = Path(os.environ.get('HF_HOME', '/home/appuser/.cache/huggingface'))
    st_cache = Path(os.environ.get('SENTENCE_TRANSFORMERS_HOME', '/home/appuser/.cache/torch/sentence_transformers'))
    nltk_data = Path(os.environ.get('NLTK_DATA', '/home/appuser/nltk_data'))
    
    models_needed = []
    
    # Check if sentence transformers models exist
    if not (st_cache / 'cross-encoder_ms-marco-MiniLM-L-6-v2').exists():
        models_needed.append('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    if not (st_cache / 'cross-encoder_ms-marco-MiniLM-L-12-v2').exists():
        models_needed.append('cross-encoder/ms-marco-MiniLM-L-12-v2')
        
    if not (st_cache / 'sentence-transformers_all-MiniLM-L6-v2').exists():
        models_needed.append('all-MiniLM-L6-v2')
        
    if not (st_cache / 'sentence-transformers_all-mpnet-base-v2').exists():
        models_needed.append('all-mpnet-base-v2')
    
    if models_needed:
        print(f'Downloading {len(models_needed)} ML models (first run only)...')
        
        from sentence_transformers import CrossEncoder, SentenceTransformer
        
        for model in models_needed:
            print(f'  Downloading: {model}')
            if 'cross-encoder' in model:
                CrossEncoder(model)
            else:
                SentenceTransformer(model)
        
        print('ML models cached successfully!')
    else:
        print('All ML models already cached (using persistent volume)')
    
    # Check NLTK data
    if not (nltk_data / 'tokenizers').exists():
        print('Downloading NLTK data...')
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        print('NLTK data cached!')
    else:
        print('NLTK data already cached')

check_and_download_models()
"

echo "=== Model check complete ==="
echo "Starting application..."

# Execute the passed command (e.g., uvicorn)
exec "$@"
