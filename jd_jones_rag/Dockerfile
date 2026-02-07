# JD Jones RAG System - Dockerfile (Optimized)
# Models are downloaded at runtime using persistent volumes (not build-time)
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies with retry logic for SSL failures
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --retries 5 --timeout 120 -r requirements.txt || \
    (echo "Retrying pip install..." && pip install --no-cache-dir --retries 5 --timeout 120 -r requirements.txt)

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/uploads /app/data /app/logs

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser

# Create cache directories for appuser
RUN mkdir -p /home/appuser/.cache/huggingface \
    /home/appuser/.cache/torch/sentence_transformers \
    /home/appuser/nltk_data && \
    chown -R appuser:appuser /home/appuser/.cache /home/appuser/nltk_data

# Set environment variables for model caching (for appuser)
ENV HF_HOME=/home/appuser/.cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/home/appuser/.cache/torch/sentence_transformers
ENV NLTK_DATA=/home/appuser/nltk_data
ENV TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface/transformers

# NOTE: ML models are NOT downloaded during build!
# They are downloaded at first container start using persistent volumes.
# This makes builds MUCH faster after the first run.

# Make entrypoint executable and fix permissions
RUN chmod +x /app/docker-entrypoint.sh && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use entrypoint for model checking
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Default command
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
