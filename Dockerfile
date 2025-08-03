# Multi-stage Docker build for California Housing Prediction API
# Stage 1: Base image with Python and dependencies
FROM python:3.9-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/app

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        procps \
        net-tools \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Dependencies installation
FROM base AS dependencies

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install scikit-learn==1.0.2 \
    && pip install pandas joblib numpy

# Stage 3: Production image
FROM dependencies AS production

# Set environment variable to indicate Docker environment
ENV RUNNING_IN_DOCKER=true

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash mlops \
    && chown -R mlops:mlops /app

# Create necessary directories first
RUN mkdir -p /app/logs /app/models /app/data /app/mlruns /app/results /app/mlflow-artifacts \
    && chown -R mlops:mlops /app

# Copy main app files
COPY --chown=mlops:mlops setup.py ./
COPY --chown=mlops:mlops requirements.txt ./
COPY --chown=mlops:mlops docker-entrypoint.sh ./

# Copy application code
COPY --chown=mlops:mlops src/ ./src/

# Copy scripts directory
COPY --chown=mlops:mlops scripts/ ./scripts/

# Copy model files if they exist (optional)
COPY --chown=mlops:mlops models/*.joblib* ./models/ 2>/dev/null || :
COPY --chown=mlops:mlops models/*.json* ./models/ 2>/dev/null || :

# Create models directory if it doesn't exist
RUN mkdir -p /app/models && chown -R mlops:mlops /app/models

# Create data directory if needed
RUN mkdir -p /app/data && chown -R mlops:mlops /app/data

# Switch to non-root user
USER mlops

# Expose port for FastAPI
EXPOSE 8000

# Health check with improved error detection
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f "http://localhost:8000/health" -H "Accept: application/json" | grep -q '"status":"healthy"' || exit 1

# Script to setup and run the application
RUN chmod +x /app/docker-entrypoint.sh

# Command to run the application
ENTRYPOINT ["/app/docker-entrypoint.sh"]
