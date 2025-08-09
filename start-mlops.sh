#!/bin/bash

# MLOps Stack Startup Script
# This script starts all monitoring and ML services

echo "🚀 Starting MLOps Stack for California Housing Prediction..."

# Check if Docker and Docker Compose are installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p logs models data results mlruns mlflow-artifacts
mkdir -p monitoring/grafana/dashboard-configs
chmod -R 755 monitoring/

# Check if models directory has any models
if [ ! "$(ls -A models/)" ]; then
    echo "⚠️  No models found in models/ directory."
    echo "   Please run the training pipeline first or copy your trained models."
fi

# Start the services
echo "🐳 Starting Docker services..."
docker-compose down --remove-orphans
docker-compose up -d

# Wait for services to start
echo "⏳ Waiting for services to start..."
sleep 30

# Check service health
echo "🔍 Checking service health..."

# Check API
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Housing Prediction API is running"
else
    echo "❌ Housing Prediction API is not responding"
fi

# Check MLflow
if curl -f http://localhost:5000/health > /dev/null 2>&1; then
    echo "✅ MLflow Tracking Server is running"
else
    echo "❌ MLflow Tracking Server is not responding"
fi

# Check Prometheus
if curl -f http://localhost:9090/-/healthy > /dev/null 2>&1; then
    echo "✅ Prometheus is running"
else
    echo "❌ Prometheus is not responding"
fi

# Check Grafana
if curl -f http://localhost:3000/api/health > /dev/null 2>&1; then
    echo "✅ Grafana is running"
else
    echo "❌ Grafana is not responding"
fi

echo ""
echo "🎉 MLOps Stack is ready!"
echo ""
echo "📊 Access your services:"
echo "   🏠 Housing Prediction API:     http://localhost:8000"
echo "   📈 API Documentation:          http://localhost:8000/docs"
echo "   🔬 MLflow Experiments:         http://localhost:5000"
echo "   📊 Prometheus Metrics:         http://localhost:9090"
echo "   📈 Grafana Dashboards:         http://localhost:3000 (admin/admin123)"
echo "   📋 System Metrics:             http://localhost:9100/metrics"
echo "   🐳 Container Metrics:          http://localhost:8080"
echo ""
echo "🔐 Default Grafana Credentials: admin / admin123"
echo ""
echo "📝 To stop all services: docker-compose down"
echo "📝 To view logs: docker-compose logs -f [service-name]"
echo "📝 Available services: mlops-api, mlflow-server, prometheus, grafana, node-exporter, cadvisor"
