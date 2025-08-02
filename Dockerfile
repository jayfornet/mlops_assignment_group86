# Multi-stage Docker build for California Housing Prediction API
# Stage 1: Base image with Python and dependencies
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Dependencies installation
FROM base as dependencies

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Stage 3: Production image
FROM dependencies as production

# Set environment variable to indicate Docker environment
ENV RUNNING_IN_DOCKER=true

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash mlops \
    && chown -R mlops:mlops /app

# Copy application code
COPY --chown=mlops:mlops src/ ./src/
COPY --chown=mlops:mlops models/ ./models/
COPY --chown=mlops:mlops logs/ ./logs/
COPY --chown=mlops:mlops setup.py ./
COPY --chown=mlops:mlops requirements.txt ./
COPY --chown=mlops:mlops docker-entrypoint.sh ./
COPY --chown=mlops:mlops data/california_housing.csv* ./data/
COPY --chown=mlops:mlops data/california_housing.joblib* ./data/

# Create necessary directories
RUN mkdir -p /app/logs /app/models /app/data /app/mlruns /app/results /app/mlflow-artifacts \
    && chown -R mlops:mlops /app

# Switch to non-root user
USER mlops

# Expose port for FastAPI
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Script to setup and run the application
COPY --chown=mlops:mlops docker-entrypoint.sh ./
RUN chmod +x /app/docker-entrypoint.sh

# Command to run the application
ENTRYPOINT ["/app/docker-entrypoint.sh"]
