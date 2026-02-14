# Kubernetes Architecture Review

> **Generated:** February 14, 2026
> **Scope:** JD Jones RAG System (`jd-jones-rag` namespace)

## 1. Architecture Diagram

```mermaid
graph TD
    subgraph "External Access"
        Ingress[Ingress (nginx)]
    end

    subgraph "Frontend Layer"
        ExtPortal[External Portal<br/>(Customer)]
        IntPortal[Internal Portal<br/>(Employee)]
    end

    subgraph "API Layer"
        API[API Server<br/>(FastAPI)]
        HPA_API[HPA: 2-10 replicas]
    end

    subgraph "Async Workers"
        CeleryWorker[Celery Workers]
        CeleryBeat[Celery Beat<br/>(Scheduler)]
        HPA_Worker[HPA: 1-8 replicas]
    end

    subgraph "Data Layer"
        Postgres[(Postgres + pgvector)]
        Redis[(Redis Cache)]
        Chroma[(ChromaDB)]
    end

    subgraph "Storage (PVCs)"
        PVC_PG[postgres-data]
        PVC_Redis[redis-data]
        PVC_Chroma[chroma-data]
        PVC_Cache[model-cache]
        PVC_Uploads[uploads-data]
    end

    subgraph "Monitoring"
        Prometheus[Prometheus]
        Grafana[Grafana]
    end

    %% Ingress Flow
    Ingress -->|support.jdjones.com| ExtPortal
    Ingress -->|portal.jdjones.com| IntPortal
    Ingress -->|api.jdjones.com| API

    %% Service Communication
    ExtPortal --> API
    IntPortal --> API
    API --> Postgres
    API --> Redis
    API --> Chroma
    
    %% Async Flow
    API -.->|Tasks| Redis
    Redis -.->|Broker| CeleryWorker
    CeleryBeat -->|Schedule| Redis
    CeleryWorker --> Postgres
    CeleryWorker --> Chroma

    %% Storage Connections
    Postgres --- PVC_PG
    Redis --- PVC_Redis
    Chroma --- PVC_Chroma
    API --- PVC_Cache
    API --- PVC_Uploads
    CeleryWorker --- PVC_Cache
    CeleryWorker --- PVC_Uploads

    %% Monitoring
    Prometheus -->|Scrape| API
    Prometheus -->|Scrape| Redis
    Prometheus -->|Scrape| Postgres
    Grafana --> Prometheus
```

## 2. Resource Inventory

### Workloads
| Name | Kind | Replicas | HPA | Ports | Description |
|------|------|----------|-----|-------|-------------|
| `api` | Deployment | 3 | 2-10 | 8000 | Main Backend API |
| `celery-worker` | Deployment | 2 | 1-8 | - | Async task worker |
| `celery-beat` | Deployment | 1 | No | - | Periodic task scheduler |
| `internal-portal` | Deployment | 2 | No | 3000 | Employee Dashboard |
| `external-portal` | Deployment | 3 | No | 3000 | Customer Portal |
| `postgres` | Deployment | 1 | No | 5432 | Main Database (User + Vector) |
| `redis` | Deployment | 1 | No | 6379 | Cache & Message Broker |
| `chromadb` | Deployment | 1 | No | 8000 | Vector Database (Embeddings) |
| `prometheus` | Deployment | 1 | No | 9090 | Metrics Collector |
| `grafana` | Deployment | 1 | No | 3000 | Metrics Visualization |

### Storage (PVCs)
| Name | Size | Access Mode | Used By |
|------|------|-------------|---------|
| `postgres-data` | 20Gi | RWO | `postgres` |
| `redis-data` | 5Gi | RWO | `redis` |
| `chroma-data` | 50Gi | RWO | `chromadb` |
| `model-cache` | 10Gi | RWX | `api`, `celery-worker` |
| `uploads-data` | 20Gi | RWX | `api`, `celery-worker` |

### Configuration
- **Secrets**: `jd-jones-secrets` (Contains DB passwords, API keys, JWT secret)
- **ConfigMaps**:
    - `jd-jones-config`: General env vars (DB host, Redis host, LLM config)
    - `postgres-init`: DB initialization SQL
    - `prometheus-config`, `grafana-*`: Monitoring config

## 3. Gap Analysis & Recommendations

### ✅ Strengths
- **Horizontal Scaling**: HPA is configured for both the API and Celery workers, allowing the system to handle variable loads.
- **Resource Management**: Requests and Limits are defined for almost all containers, protecting the cluster from noisy neighbors.
- **Security**: Sensitive data is correctly decoupled using Kubernetes Secrets. Network policies are in place to restrict access.
- **Observability**: A full monitoring stack (Prometheus + Grafana) is included.

### ⚠️ Findings & Risks

#### 1. Missing Probes (Reliability)
- **Severity**: 🟡 Medium
- **Finding**: The following deployments lack `livenessProbe` and `readinessProbe`:
    - `celery-worker`
    - `celery-beat`
    - `prometheus`
    - `grafana`
- **Risk**: Kubernetes cannot detect if these services are deadlocked or unresponsive, preventing auto-restart.

#### 2. Hardcoded Credentials (Security)
- **Severity**: 🔴 High
- **Finding**: In `10-monitoring.yaml`, the Grafana admin password is hardcoded:
  ```yaml
  - name: GF_SECURITY_ADMIN_PASSWORD
    value: "jdjones-admin"
  ```
- **Recommendation**: Move this value to `jd-jones-secrets` and reference it via `secretKeyRef`.

#### 3. Single Point of Failure (Availability)
- **Severity**: 🟡 Medium
- **Finding**: `postgres`, `redis`, and `chromadb` are single-replica deployments.
- **Risk**: If the node hosting these pods fails, there will be downtime until rescheduling occurs. For production, consider StatefulSets with replication or managed cloud services.

#### 4. Shared Storage (Architecture)
- **Severity**: ⚪ Low (Info)
- **Finding**: `model-cache` and `uploads-data` use `ReadWriteMany` (RWX) access mode.
- **Note**: Ensure the underlying storage class (`standard`) supports RWX (e.g., NFS, AWS EFS). Block storage (EBS/PD) typically does not support RWX.

### 📋 Action Plan
1. [ ] **Security**: Move Grafana password to Secrets.
2. [ ] **Reliability**: Add TCP or Exec liveness probes to Celery workers.
3. [ ] **Reliability**: Add HTTP probes to Prometheus (`/-/healthy`) and Grafana (`/api/health`).
4. [ ] **Documentation**: Verify StorageClass capabilities for RWX support.
