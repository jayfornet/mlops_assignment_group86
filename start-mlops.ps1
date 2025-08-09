# MLOps Stack Startup Script for Windows
# This script starts all monitoring and ML services

Write-Host "🚀 Starting MLOps Stack for California Housing Prediction..." -ForegroundColor Green

# Check if Docker is installed
if (!(Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Docker is not installed. Please install Docker Desktop first." -ForegroundColor Red
    exit 1
}

if (!(Get-Command docker-compose -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Docker Compose is not installed. Please install Docker Compose first." -ForegroundColor Red
    exit 1
}

# Create necessary directories
Write-Host "📁 Creating necessary directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "logs", "models", "data", "results", "mlruns", "mlflow-artifacts"
New-Item -ItemType Directory -Force -Path "monitoring\grafana\dashboard-configs"

# Check if models directory has any models
if ((Get-ChildItem -Path "models" -ErrorAction SilentlyContinue).Count -eq 0) {
    Write-Host "⚠️  No models found in models/ directory." -ForegroundColor Yellow
    Write-Host "   Please run the training pipeline first or copy your trained models." -ForegroundColor Yellow
}

# Start the services
Write-Host "🐳 Starting Docker services..." -ForegroundColor Blue
docker-compose down --remove-orphans
docker-compose up -d

# Wait for services to start
Write-Host "⏳ Waiting for services to start..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

# Check service health
Write-Host "🔍 Checking service health..." -ForegroundColor Blue

# Check API
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ Housing Prediction API is running" -ForegroundColor Green
    }
} catch {
    Write-Host "❌ Housing Prediction API is not responding" -ForegroundColor Red
}

# Check MLflow
try {
    $response = Invoke-WebRequest -Uri "http://localhost:5000/health" -UseBasicParsing -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ MLflow Tracking Server is running" -ForegroundColor Green
    }
} catch {
    Write-Host "❌ MLflow Tracking Server is not responding" -ForegroundColor Red
}

# Check Prometheus
try {
    $response = Invoke-WebRequest -Uri "http://localhost:9090/-/healthy" -UseBasicParsing -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ Prometheus is running" -ForegroundColor Green
    }
} catch {
    Write-Host "❌ Prometheus is not responding" -ForegroundColor Red
}

# Check Grafana
try {
    $response = Invoke-WebRequest -Uri "http://localhost:3000/api/health" -UseBasicParsing -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Host "✅ Grafana is running" -ForegroundColor Green
    }
} catch {
    Write-Host "❌ Grafana is not responding" -ForegroundColor Red
}

Write-Host ""
Write-Host "🎉 MLOps Stack is ready!" -ForegroundColor Green
Write-Host ""
Write-Host "📊 Access your services:" -ForegroundColor Cyan
Write-Host "   🏠 Housing Prediction API:     http://localhost:8000" -ForegroundColor White
Write-Host "   📈 API Documentation:          http://localhost:8000/docs" -ForegroundColor White
Write-Host "   🔬 MLflow Experiments:         http://localhost:5000" -ForegroundColor White
Write-Host "   📊 Prometheus Metrics:         http://localhost:9090" -ForegroundColor White
Write-Host "   📈 Grafana Dashboards:         http://localhost:3000 (admin/admin123)" -ForegroundColor White
Write-Host "   📋 System Metrics:             http://localhost:9100/metrics" -ForegroundColor White
Write-Host "   🐳 Container Metrics:          http://localhost:8080" -ForegroundColor White
Write-Host ""
Write-Host "🔐 Default Grafana Credentials: admin / admin123" -ForegroundColor Yellow
Write-Host ""
Write-Host "📝 To stop all services: docker-compose down" -ForegroundColor Cyan
Write-Host "📝 To view logs: docker-compose logs -f [service-name]" -ForegroundColor Cyan
Write-Host "📝 Available services: mlops-api, mlflow-server, prometheus, grafana, node-exporter, cadvisor" -ForegroundColor Cyan
