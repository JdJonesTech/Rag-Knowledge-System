# JD Jones RAG - Kubernetes Deployment Guide

## Prerequisites
- Kubernetes cluster (1.25+)
- kubectl configured
- Helm 3.x (optional, for cert-manager)
- Container registry access

## Quick Start

### 1. Create namespace and base resources
```bash
kubectl apply -f k8s/00-namespace.yaml
kubectl apply -f k8s/01-secrets-configmap.yaml
kubectl apply -f k8s/02-storage.yaml
```

### 2. Deploy databases
```bash
kubectl apply -f k8s/03-postgres.yaml
kubectl apply -f k8s/04-redis.yaml
kubectl apply -f k8s/05-chromadb.yaml

# Wait for databases to be ready
kubectl wait --for=condition=ready pod -l app=postgres -n jd-jones-rag --timeout=120s
kubectl wait --for=condition=ready pod -l app=redis -n jd-jones-rag --timeout=60s
kubectl wait --for=condition=ready pod -l app=chromadb -n jd-jones-rag --timeout=120s
```

### 3. Deploy application
```bash
kubectl apply -f k8s/06-api.yaml
kubectl apply -f k8s/07-celery.yaml
kubectl apply -f k8s/08-frontends.yaml
```

### 4. Configure ingress
```bash
kubectl apply -f k8s/09-ingress.yaml
```

### 5. Deploy monitoring (optional)
```bash
kubectl apply -f k8s/10-monitoring.yaml
```

## Building Container Images

```bash
# Build and push API image
docker build -t your-registry/jd-jones-rag-api:latest .
docker push your-registry/jd-jones-rag-api:latest

# Build and push frontend images
docker build -t your-registry/jd-jones-rag-internal-portal:latest ./frontend/internal
docker push your-registry/jd-jones-rag-internal-portal:latest

docker build -t your-registry/jd-jones-rag-external-portal:latest ./frontend/external
docker push your-registry/jd-jones-rag-external-portal:latest
```

## Configuration

### Secrets
Update `01-secrets-configmap.yaml` with your actual secrets:
- `POSTGRES_PASSWORD`: Database password
- `JWT_SECRET_KEY`: JWT signing key
- `OPENAI_API_KEY`: OpenAI API key (if using OpenAI)

### Environment Variables
Update the ConfigMap in `01-secrets-configmap.yaml`:
- `LLM_PROVIDER`: `ollama` or `openai`
- `LLM_MODEL`: Model name (e.g., `llama3.2`, `gpt-4`)
- `EMBEDDING_MODEL`: Embedding model name

## Scaling

### Manual Scaling
```bash
kubectl scale deployment api --replicas=5 -n jd-jones-rag
kubectl scale deployment celery-worker --replicas=4 -n jd-jones-rag
```

### Autoscaling
HPAs are configured for:
- API: 2-10 replicas based on CPU/memory
- Celery Workers: 1-8 replicas based on CPU

## Monitoring

### Check pod status
```bash
kubectl get pods -n jd-jones-rag
```

### View logs
```bash
kubectl logs -f deployment/api -n jd-jones-rag
kubectl logs -f deployment/celery-worker -n jd-jones-rag
```

### Port forwarding for local access
```bash
# API
kubectl port-forward svc/api 8000:8000 -n jd-jones-rag

# Grafana dashboard
kubectl port-forward svc/grafana 3000:3000 -n jd-jones-rag
```

## Troubleshooting

### Pod not starting
```bash
kubectl describe pod <pod-name> -n jd-jones-rag
kubectl logs <pod-name> -n jd-jones-rag --previous
```

### Database connection issues
```bash
kubectl exec -it deployment/api -n jd-jones-rag -- python -c "from src.config.settings import settings; print(settings.database_url)"
```

### Check HPA status
```bash
kubectl get hpa -n jd-jones-rag
```
