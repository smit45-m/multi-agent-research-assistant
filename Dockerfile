# Stage 1: builder
FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Python packages (uses CPU-only PyTorch wheel index from requirements.txt)
RUN pip install --user --no-cache-dir -r requirements.txt

# Pre-download the sentence-transformers embedding model into HF cache so
# EMBEDDING_LOCAL_ONLY=True works offline on cold container starts
ENV HF_HOME=/root/.cache/huggingface
RUN /root/.local/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" || \
    python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Stage 2: runtime
FROM python:3.12-slim

WORKDIR /app

# Non-root user with fixed UID/GID 999
RUN groupadd -r -g 999 appuser && useradd -r -u 999 -g appuser -m -d /home/appuser appuser

# Install curl for HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies and pre-cached embedding model from builder
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local
COPY --from=builder --chown=appuser:appuser /root/.cache/huggingface /home/appuser/.cache/huggingface

# Ensure local bin, HuggingFace cache, and single-node 2-vCPU thread limits are configured
ENV PATH=/home/appuser/.local/bin:$PATH \
    HF_HOME=/home/appuser/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    TOKENIZERS_PARALLELISM=false

# Copy application code and ensure writable data directory
COPY --chown=appuser:appuser . .
RUN mkdir -p /app/data/vector_store && chown -R appuser:appuser /app/data

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health/ready || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-keep-alive", "30", "--limit-concurrency", "64"]
