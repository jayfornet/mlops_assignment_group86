# Production Docker image for California Housing Prediction API
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash mlops \
    && chown -R mlops:mlops /app

# Create necessary directories
RUN mkdir -p /app/logs /app/models /app/scripts \
    && chown -R mlops:mlops /app

# Copy application code (API only)
COPY --chown=mlops:mlops src/api/ ./src/api/
COPY --chown=mlops:mlops src/__init__.py ./src/
COPY --chown=mlops:mlops scripts/verify_mlflow_data.py ./scripts/

# Copy the best model (this will be provided by CI/CD)
COPY --chown=mlops:mlops models/ ./models/

# Switch to non-root user
USER mlops

# Expose port for FastAPI
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f "http://localhost:8000/health" || exit 1

# Start the API directly (no training, no DVC)
CMD ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
