# JD Jones RAG System - Team Collaboration Guide

This guide provides **three methods** for remote colleagues to get the JD Jones RAG system running on their machines. Choose the method that best fits your situation.

---

## Prerequisites (All Methods)

- **Docker Desktop** installed and running ([Download](https://www.docker.com/products/docker-desktop/))
- **Git** installed ([Download](https://git-scm.com/downloads))
- At least **20 GB** free disk space
- Clone the repository first:

```bash
git clone <your-repo-url>
cd jd_jones_rag
```

---

## Method 1: Pull from Docker Hub (Fastest if images are already pushed)

If images have been pushed to Docker Hub under `techjdjones/`, simply pull them:

```bash
# Pull all images
docker pull techjdjones/jd_jones_rag-api:beta
docker pull techjdjones/jd_jones_rag-external-portal:beta
docker pull techjdjones/jd_jones_rag-internal-portal:beta
docker pull techjdjones/jd_jones_rag-celery-worker:beta
docker pull techjdjones/jd_jones_rag-celery-beat:beta
docker pull techjdjones/jd_jones_rag-mcp-server:beta
docker pull techjdjones/jd_jones_rag-flower:beta

# Re-tag to match docker-compose.yml service names
docker tag techjdjones/jd_jones_rag-api:beta jd_jones_rag-api:latest
docker tag techjdjones/jd_jones_rag-external-portal:beta jd_jones_rag-external-portal:latest
docker tag techjdjones/jd_jones_rag-internal-portal:beta jd_jones_rag-internal-portal:latest
docker tag techjdjones/jd_jones_rag-celery-worker:beta jd_jones_rag-celery-worker:latest
docker tag techjdjones/jd_jones_rag-celery-beat:beta jd_jones_rag-celery-beat:latest
docker tag techjdjones/jd_jones_rag-mcp-server:beta jd_jones_rag-mcp-server:latest
docker tag techjdjones/jd_jones_rag-flower:beta jd_jones_rag-flower:latest

# Start all services
docker compose up -d
```

---

## Method 2: Load from Archive Files (Best for unstable internet)

If you've received `.tar` archive files (shared via Google Drive, OneDrive, USB, etc.):

### Archive Files:
| File | Contents | Approx. Size |
|------|----------|-------------|
| `jd_jones_rag-frontends_beta.tar` | External Portal + Internal Portal | ~400 MB |
| `jd_jones_rag-backend_beta.tar` | API + Celery Worker + Celery Beat + MCP Server + Flower | ~15 GB (shared layers) |

### Steps:

```bash
# Load frontend images (fast, ~400 MB)
docker load -i jd_jones_rag-frontends_beta.tar

# Load backend images (takes a few minutes, ~15 GB)
docker load -i jd_jones_rag-backend_beta.tar

# Verify images are loaded
docker images --filter "reference=jd_jones_rag*"
```

Expected output:
```
REPOSITORY                     TAG       SIZE
jd_jones_rag-api               latest    14.4GB
jd_jones_rag-external-portal   latest    222MB
jd_jones_rag-internal-portal   latest    222MB
jd_jones_rag-celery-beat       latest    14.4GB
jd_jones_rag-mcp-server        latest    14.4GB
jd_jones_rag-flower            latest    14.4GB
jd_jones_rag-celery-worker     latest    14.4GB
```

```bash
# Start all services
docker compose up -d
```

### How to Create Archive Files (for the person sharing):

```bash
# Save frontend images (deduplicates shared layers)
docker save -o jd_jones_rag-frontends_beta.tar jd_jones_rag-external-portal:latest jd_jones_rag-internal-portal:latest

# Save backend images (deduplicates shared layers — saves ~60 GB → ~15 GB)
docker save -o jd_jones_rag-backend_beta.tar jd_jones_rag-api:latest jd_jones_rag-celery-worker:latest jd_jones_rag-celery-beat:latest jd_jones_rag-mcp-server:latest jd_jones_rag-flower:latest
```

---

## Method 3: Build from Source (Most reliable, no image sharing needed)

This method builds all Docker images locally from the source code. Best for long-term collaboration where everyone stays in sync via Git.

### Step 1: Clone and Navigate

```bash
git clone <your-repo-url>
cd jd_jones_rag
```

### Step 2: Environment Setup

Create a `.env` file in the project root (if not already present):

```env
# Required for AI features (optional — system works without it)
OPENAI_API_KEY=sk-your-key-here

# Database (defaults are already set in docker-compose.yml)
POSTGRES_USER=jdjones
POSTGRES_PASSWORD=jdjones_secure_2024
POSTGRES_DB=jdjones_rag

# Redis (defaults are already set)
REDIS_URL=redis://redis:6379/0
```

> **Note:** The system works without `OPENAI_API_KEY` — AI features will show a warning but all other features (quotation management, PDF generation, document search, etc.) work fully.

### Step 3: Build All Images

```bash
# Build all services (first time takes 10-15 minutes)
docker compose build

# Or build specific services
docker compose build api               # Backend API
docker compose build external-portal    # Customer-facing portal
docker compose build internal-portal    # Internal dashboard
```

### Step 4: Start Everything

```bash
# Start all services in detached mode
docker compose up -d

# Check everything is running
docker compose ps
```

### Step 5: Verify

| Service | URL | Description |
|---------|-----|-------------|
| Internal Portal | http://localhost:3000 | JD Jones Knowledge Assistant (Quotations, Enquiries, Chat, Documents) |
| External Portal | http://localhost:3001 | Customer-facing quotation request portal |
| API Backend | http://localhost:8000/docs | FastAPI Swagger documentation |
| Flower | http://localhost:5555 | Celery task monitoring dashboard |

---

## Troubleshooting

### "The OPENAI_API_KEY variable is not set"
This is just a warning. The system works without it. To suppress it, create a `.env` file with `OPENAI_API_KEY=not-set`.

### Container won't start (health check failing)
```bash
# Check logs
docker compose logs api
docker compose logs external-portal

# Restart a specific service
docker compose restart api
```

### Port already in use
```bash
# Check what's using the port
netstat -ano | findstr :3000

# Or change ports in docker-compose.yml
```

### Out of disk space
```bash
# Clean up unused Docker resources
docker system prune -a

# Check Docker disk usage
docker system df
```

### Rebuild after code changes
```bash
# Rebuild specific service
docker compose build api
docker compose up -d api

# Or rebuild everything
docker compose build
docker compose up -d
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    Docker Compose Stack                   │
├──────────────┬──────────────┬──────────────┬────────────┤
│  External    │  Internal    │   API        │  Celery    │
│  Portal      │  Portal      │   Backend    │  Workers   │
│  :3001       │  :3000       │   :8000      │            │
│  (Next.js)   │  (Next.js)   │  (FastAPI)   │  (Python)  │
├──────────────┴──────────────┼──────────────┼────────────┤
│                             │  PostgreSQL  │  Redis     │
│                             │  :5432       │  :6379     │
│                             ├──────────────┤            │
│                             │  ChromaDB    │  Flower    │
│                             │  :8001       │  :5555     │
└─────────────────────────────┴──────────────┴────────────┘
```

---

## Quick Reference Commands

```bash
# Start all services
docker compose up -d

# Stop all services
docker compose down

# View logs (follow mode)
docker compose logs -f api

# Rebuild and restart a service
docker compose build api && docker compose up -d api    # Linux/Mac
docker compose build api; docker compose up -d api      # PowerShell

# Check service health
docker compose ps

# Enter a container shell
docker compose exec api bash
```
