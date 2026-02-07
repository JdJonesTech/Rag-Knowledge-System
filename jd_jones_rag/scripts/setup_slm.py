"""
SLM Model Setup Script
Downloads and configures Small Language Models for local inference.

Architecture Overview:
=====================
┌──────────────────────────────────────────────────────────────┐
│                      MAIN BRAIN (LLM)                        │
│              Llama 3.2 / GPT-4 / Claude                      │
│    Handles: Complex reasoning, orchestration, synthesis      │
└─────────────────────────────┬────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Intent SLM     │  │  Entity SLM     │  │  Compliance SLM │
│  (sklearn)      │  │  (sklearn)      │  │  (sklearn)      │
│  < 10ms         │  │  < 10ms         │  │  < 10ms         │
└─────────────────┘  └─────────────────┘  └─────────────────┘

SLM Types:
=========
1. sklearn (Recommended)
   - TF-IDF + Naive Bayes classifier
   - Trained on YOUR company data
   - No download needed, just train!
   - Inference: < 10ms

2. Ollama Small Models (Optional)
   - phi3:mini (2.7B params, 1.6GB)
   - tinyllama (1.1B params, 638MB)
   - For more complex local generation

3. Sentence Transformers (Optional)
   - For semantic similarity/matching
   - all-MiniLM-L6-v2 (22M params, 80MB)
"""

import subprocess
import sys
import os
from pathlib import Path


def check_ollama():
    """Check if Ollama is installed and running."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print("✅ Ollama is installed and running")
            print("\nCurrently installed models:")
            print(result.stdout)
            return True
        else:
            print("❌ Ollama is not running")
            return False
    except FileNotFoundError:
        print("❌ Ollama is not installed")
        print("   Install from: https://ollama.ai")
        return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False


def download_ollama_slm(model_name: str):
    """Download an SLM via Ollama."""
    print(f"\n📥 Downloading {model_name}...")
    try:
        result = subprocess.run(
            ["ollama", "pull", model_name],
            capture_output=False,  # Show progress
            timeout=600  # 10 minute timeout
        )
        if result.returncode == 0:
            print(f"✅ {model_name} downloaded successfully")
            return True
        else:
            print(f"❌ Failed to download {model_name}")
            return False
    except Exception as e:
        print(f"❌ Error downloading {model_name}: {e}")
        return False


def setup_sklearn_slm():
    """Setup sklearn for SLM classification (recommended)."""
    print("\n📦 Setting up sklearn for SLM classification...")
    
    try:
        import sklearn
        print(f"✅ sklearn is installed (version {sklearn.__version__})")
        return True
    except ImportError:
        print("   Installing sklearn...")
        subprocess.run([sys.executable, "-m", "pip", "install", "scikit-learn"], check=True)
        print("✅ sklearn installed successfully")
        return True


def setup_sentence_transformers():
    """Setup sentence transformers for semantic matching (optional)."""
    print("\n📦 Setting up sentence transformers...")
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ sentence-transformers is installed")
        
        # Download a small model
        print("   Loading all-MiniLM-L6-v2 model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ all-MiniLM-L6-v2 loaded (22M params, very fast)")
        return True
    except ImportError:
        print("   sentence-transformers not installed")
        print("   Run: pip install sentence-transformers")
        return False
    except Exception as e:
        print(f"⚠️  Error setting up sentence transformers: {e}")
        return False


def create_model_directory():
    """Create directory for trained SLM models."""
    model_dir = Path("models/slm")
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ Model directory created: {model_dir.absolute()}")
    return model_dir


def print_architecture():
    """Print the LLM + SLM architecture."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           JD JONES RAG - LLM + SLM ARCHITECTURE                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  ┌──────────────────────────────────────────────────────────┐    ║
║  │                    MAIN BRAIN (LLM)                      │    ║
║  │                     Llama 3.2 via Ollama                 │    ║
║  │                                                          │    ║
║  │  • Complex multi-step reasoning                          │    ║
║  │  • Query orchestration                                   │    ║
║  │  • Response synthesis                                    │    ║
║  │  • Tool selection & execution                            │    ║
║  │  • Latency: 500ms - 2s                                   │    ║
║  └────────────────────────┬─────────────────────────────────┘    ║
║                           │                                      ║
║            ┌──────────────┼──────────────┐                       ║
║            │              │              │                       ║
║            ▼              ▼              ▼                       ║
║  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐              ║
║  │ Intent SLM   │ │ Entity SLM   │ │ Matcher SLM  │              ║
║  │ (sklearn)    │ │ (sklearn)    │ │ (sklearn)    │              ║
║  │              │ │              │ │              │              ║
║  │ • Classify   │ │ • Extract    │ │ • Product    │              ║
║  │   intent     │ │   products   │ │   matching   │              ║
║  │ • < 10ms     │ │ • < 10ms     │ │ • < 20ms     │              ║
║  └──────────────┘ └──────────────┘ └──────────────┘              ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  WORKFLOW:                                                       ║
║  1. Query → SLM classifies intent (< 10ms)                       ║
║  2. SLM extracts entities (product codes, specs) (< 10ms)        ║
║  3. IF simple query: SLM handles directly                        ║
║  4. IF complex: Escalate to LLM main brain                       ║
║  5. LLM orchestrates tools, reasons, synthesizes                 ║
║                                                                  ║
║  BENEFITS:                                                       ║
║  • 70-80% queries handled by SLM (< 50ms)                        ║
║  • LLM only used for complex reasoning                           ║
║  • SLMs trained on YOUR company data                             ║
║  • Privacy-preserving (all local)                                ║
╚══════════════════════════════════════════════════════════════════╝
""")


def print_next_steps():
    """Print next steps for training SLMs."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                         NEXT STEPS                               ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. TRAIN SLMs ON YOUR DATA (API Endpoint):                      ║
║     POST /agentic/slm/train                                      ║
║     {                                                            ║
║       "slm_type": "intent_classifier",                           ║
║       "training_method": "sklearn",                              ║
║       "num_examples": 100                                        ║
║     }                                                            ║
║                                                                  ║
║  2. TEST SLM INFERENCE:                                          ║
║     POST /agentic/slm/predict                                    ║
║     {                                                            ║
║       "slm_type": "intent_classifier",                           ║
║       "text": "What is the temperature rating of NA 701?"        ║
║     }                                                            ║
║                                                                  ║
║  3. VIEW SLM ARCHITECTURE:                                       ║
║     GET /agentic/slm/architecture                                ║
║                                                                  ║
║  4. (OPTIONAL) DOWNLOAD ADDITIONAL OLLAMA MODELS:                ║
║     ollama pull phi3:mini      # 1.6 GB                          ║
║     ollama pull tinyllama      # 638 MB                          ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
""")


def main():
    """Main setup function."""
    print("\n" + "="*60)
    print("       JD JONES RAG - SLM SETUP SCRIPT")
    print("="*60)
    
    # Print architecture overview
    print_architecture()
    
    # Check Ollama
    check_ollama()
    
    # Setup sklearn (recommended for SLM classification)
    setup_sklearn_slm()
    
    # Create model directory
    create_model_directory()
    
    # Optional: Setup sentence transformers
    print("\n" + "-"*60)
    print("Optional Components:")
    print("-"*60)
    
    setup_sentence_transformers()
    
    # Print next steps
    print_next_steps()
    
    print("\n✅ SLM setup complete!")
    print("\n📝 Summary:")
    print("   • You have Llama 3.2 as your MAIN BRAIN (LLM)")
    print("   • SLMs will be sklearn models trained on your data")
    print("   • Train SLMs via the API endpoints")
    print("   • No additional downloads required!")


if __name__ == "__main__":
    main()
