# JD Jones RAG System - Team Collaboration Guide

This guide provides methods for remote colleagues to get the JD Jones RAG system running on their machines.

## Prerequisites

- **Docker Desktop** installed and running
- **Git** installed
- At least **20 GB** free disk space
- Clone the repository:

```bash
git clone https://github.com/JdJonesTech/Rag-Knowledge-System.git
cd Rag-Knowledge-System
```

---

## Method 1: Build from Source (Docker)

This method builds all Docker images locally from the source code. This is the most reliable way to stay in sync.

1. **Navigate to the core directory**:
   ```bash
   cd jd_jones_rag
   ```

2. **Environment Setup**:
   Create a `.env` file from the example:
   ```bash
   cp .env.example .env
   ```

3. **Build Images**:
   ```bash
   docker compose build
   ```

4. **Start Services**:
   ```bash
   docker compose up -d
   ```

---

## Method 2: Native Development (Local Machine)

For debugging or focused backend work, you might want to run some services natively.

1. **Prerequisites**:
   - Python 3.11+
   - Node.js 18+ (for frontends)
   - Docker (still needed for database/redis)

2. **Start Infrastructure Only**:
   ```bash
   docker compose up -d postgres redis chromadb
   ```

3. **Backend Setup**:
   ```bash
   cd jd_jones_rag
   python -m venv venv
   source venv/bin/activate  # venv\Scripts\activate on Windows
   pip install -r requirements.txt
   python src/api/main.py
   ```

4. **Frontend Setup**:
   ```bash
   cd jd_jones_rag/frontend/internal-portal
   npm install
   npm run dev
   ```

---

## Access Points

| Service | URL |
|---------|-----|
| Internal Portal | http://localhost:3000 |
| External Portal | http://localhost:3001 |
| API Backend | http://localhost:8000/docs |
| Flower | http://localhost:5555 |

## Troubleshooting

- **OPENAI_API_KEY warning**: The system will warn if this isn't set, but core search features will still function.
- **Port Conflicts**: Ensure ports 3000, 3001, and 8000 are not being used by other applications.
- **Disk Space**: If the build fails, check if you have enough space for the Docker layers.
