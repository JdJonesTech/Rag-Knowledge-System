# JD Jones RAG System - Quick Start Guide (FREE Local Setup)

## ✅ Current Status
All core services are running:
- **API**: http://localhost:8000/docs
- **PostgreSQL**: Connected ✓
- **Redis**: Connected ✓
- **ChromaDB**: Connected ✓

## 🆓 Install Ollama (for FREE LLM)

### Step 1: Download Ollama
1. Open your browser and go to: **https://ollama.com/download/windows**
2. Click "Download for Windows"
3. Run the `OllamaSetup.exe` installer
4. Follow installation prompts

### Step 2: Pull a Model
After installation, open Command Prompt or PowerShell and run:
```bash
ollama pull llama3.2
```

This downloads the Llama 3.2 model (~2GB). Other options:
- `ollama pull mistral` - Fast, good quality
- `ollama pull phi3` - Smaller, faster
- `ollama pull llama2` - Previous generation

### Step 3: Verify Ollama is Running
```bash
ollama list
```
You should see `llama3.2:latest` listed.

## 🚀 Run the Demo

Once Ollama is installed and has a model, run:
```bash
# From the project directory
docker-compose exec api python demo_system.py
```

Or access the API directly:
```bash
# Health check
curl http://localhost:8000/health

# API docs
open http://localhost:8000/docs
```

## 💰 Cost Summary
| Component | Provider | Cost |
|-----------|----------|------|
| LLM | Ollama (llama3.2) | **$0** |
| Embeddings | sentence-transformers | **$0** |
| Reranker | cross-encoder | **$0** |
| Database | PostgreSQL | **$0** |
| Vector Store | ChromaDB | **$0** |

**Total: $0/month!**

## 📋 Quick Commands

```bash
# Start all services
docker-compose up -d api postgres redis chromadb

# Stop services
docker-compose down

# View API logs
docker logs jd_jones_api -f

# Run tests
docker-compose exec api pytest tests/ -q

# Ingest data
docker-compose exec api python data/ingest_data.py
```

## 🔧 Troubleshooting

### Ollama connection issues
If the API can't connect to Ollama, check:
1. Ollama is running: `ollama list`
2. The base URL in `.env` is correct: `OLLAMA_BASE_URL=http://host.docker.internal:11434`

### Database connection issues
Reset the database:
```bash
docker-compose down postgres
docker volume rm jd_jones_rag_postgres_data
docker-compose up -d postgres
docker-compose restart api
```

### ChromaDB embedding issues
Reset ChromaDB data:
```bash
docker-compose down chromadb
docker volume rm jd_jones_rag_chroma_data
docker-compose up -d chromadb
docker-compose restart api
```
