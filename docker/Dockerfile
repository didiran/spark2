# Spark-Kafka ML Training Pipeline
# Multi-stage Docker build for production deployment
# Author: Gabriel Demetrios Lafis

# ============================================================
# Stage 1: Builder - install dependencies
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ============================================================
# Stage 2: Runtime - slim production image
# ============================================================
FROM python:3.11-slim AS runtime

LABEL maintainer="Gabriel Demetrios Lafis"
LABEL description="Spark-Kafka ML Training Pipeline"
LABEL version="1.0.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIPELINE_ENV=production \
    LOG_LEVEL=INFO

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 --create-home appuser

COPY --from=builder /install /usr/local

WORKDIR /app

COPY src/ ./src/
COPY config/ ./config/
COPY main.py .

RUN mkdir -p /app/logs /app/data /app/demo_results /app/demo_feature_store \
    && chown -R appuser:appuser /app

USER appuser

HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import src; print('healthy')" || exit 1

ENTRYPOINT ["python", "main.py"]
CMD ["--samples", "5000", "--batches", "3"]
