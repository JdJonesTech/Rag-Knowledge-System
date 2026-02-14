# Docker Health Checks

This document lists the health checks configured for each service in the JD Jones RAG system.

## Development (`docker-compose.yml`)

| Service | Health Check Command | Interval | Retries | Start Period |
|---------|----------------------|----------|---------|--------------|
| **api** | `curl -f http://localhost:8000/health` | 30s | 3 | - |
| **postgres** | `pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}` | 10s | 5 | - |
| **redis** | `redis-cli ping` | 10s | 5 | - |
| **chromadb** | `curl -f http://localhost:8000/api/v1/heartbeat` | 30s | 3 | 10s |
| **celery-worker** | `celery ... inspect ping ...` | 60s | 3 | 30s |
| **mcp-server** | `curl -f http://localhost:8000/health` | 30s | 3 | 30s |
| **celery-beat** | `test -f celerybeat-schedule` | 60s | 3 | 30s |
| **internal-portal** | `wget --no-verbose --tries=1 --spider http://localhost:3000/` | 30s | 3 | 30s |
| **external-portal** | `wget --no-verbose --tries=1 --spider http://localhost:3000/` | 30s | 3 | 30s |
| **flower** | `curl -f http://localhost:5555/` | 30s | 3 | 30s |

## Production (`docker-compose.prod.yml`)

In addition to the checks above, production services include:

| Service | Health Check Command | Interval | Retries | Start Period |
|---------|----------------------|----------|---------|--------------|
| **nginx** | `wget -q --spider http://localhost:80/` | 30s | 3 | 10s |
| **prometheus** | `wget -q --spider http://localhost:9090/-/healthy` | 30s | 3 | 30s |
| **grafana** | `wget -q --spider http://localhost:3000/api/health` | 30s | 3 | 30s |

## Troubleshooting

If a service is `unhealthy`:
1. Check logs: `docker compose logs <service_name>`
2. Verify port binding: `docker compose ps`
3. Exec into container and run check manually: `docker compose exec <service_name> <command>`
