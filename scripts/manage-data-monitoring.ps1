# Data Monitoring Pipeline Management Script
# This script helps manage the data monitoring infrastructure

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("start", "stop", "status", "test", "trigger", "logs", "restart", "build")]
    [string]$Action,
    
    [string]$Component = "all",  # all, webhook, scheduler, monitoring
    [string]$DataSource = "california_housing",
    [switch]$Force,
    [switch]$Build,
    [string]$LogLevel = "INFO"
)

# Configuration
$WEBHOOK_PORT = 5555
$COMPOSE_FILE = "docker-compose.data-monitoring.yml"
$WEBHOOK_URL = "http://localhost:$WEBHOOK_PORT"

# Colors for output
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    } else {
        $input | Write-Output
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

function Write-Success { Write-ColorOutput Green $args }
function Write-Warning { Write-ColorOutput Yellow $args }
function Write-Error { Write-ColorOutput Red $args }
function Write-Info { Write-ColorOutput Cyan $args }

function Test-Docker {
    try {
        docker version | Out-Null
        return $true
    } catch {
        Write-Error "Docker is not running or not installed"
        return $false
    }
}

function Test-DockerCompose {
    try {
        docker compose version | Out-Null
        return $true
    } catch {
        try {
            docker-compose --version | Out-Null
            return $true
        } catch {
            Write-Error "Docker Compose is not available"
            return $false
        }
    }
}

function Start-DataMonitoring {
    Write-Info "🚀 Starting Data Monitoring Pipeline..."
    
    if (-not (Test-Docker)) { return $false }
    if (-not (Test-DockerCompose)) { return $false }
    
    # Build images if requested
    if ($Build) {
        Write-Info "🔨 Building Docker images..."
        docker compose -f $COMPOSE_FILE build
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to build Docker images"
            return $false
        }
    }
    
    # Start services
    Write-Info "📦 Starting containers..."
    docker compose -f $COMPOSE_FILE up -d
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "✅ Data monitoring pipeline started successfully"
        
        # Wait for services to be ready
        Write-Info "⏳ Waiting for services to be ready..."
        Start-Sleep -Seconds 10
        
        # Check health
        Show-Status
        return $true
    } else {
        Write-Error "❌ Failed to start data monitoring pipeline"
        return $false
    }
}

function Stop-DataMonitoring {
    Write-Info "🛑 Stopping Data Monitoring Pipeline..."
    
    docker compose -f $COMPOSE_FILE down
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "✅ Data monitoring pipeline stopped successfully"
        return $true
    } else {
        Write-Error "❌ Failed to stop data monitoring pipeline"
        return $false
    }
}

function Show-Status {
    Write-Info "📊 Data Monitoring Pipeline Status"
    Write-Info "================================="
    
    # Check Docker containers
    Write-Info "🐳 Docker Containers:"
    docker compose -f $COMPOSE_FILE ps
    
    Write-Info ""
    
    # Check webhook server health
    try {
        $response = Invoke-RestMethod -Uri "$WEBHOOK_URL/health" -TimeoutSec 5
        Write-Success "✅ Webhook Server: Healthy"
        Write-Info "   Status: $($response.status)"
        Write-Info "   Timestamp: $($response.timestamp)"
    } catch {
        Write-Error "❌ Webhook Server: Not responding"
    }
    
    # Check webhook status
    try {
        $response = Invoke-RestMethod -Uri "$WEBHOOK_URL/webhook/status" -TimeoutSec 5
        Write-Success "✅ Webhook Configuration:"
        Write-Info "   GitHub Token: $($response.github_token_configured)"
        Write-Info "   Secret Configured: $($response.webhook_secret_configured)"
        Write-Info "   Repository: $($response.github_repository)"
    } catch {
        Write-Warning "⚠️  Could not get webhook configuration"
    }
    
    # Show recent logs
    Write-Info ""
    Write-Info "📋 Recent Logs (last 10 lines):"
    if (Test-Path "logs/data_monitoring.log") {
        Get-Content "logs/data_monitoring.log" -Tail 10
    } else {
        Write-Info "   No monitoring logs found"
    }
}

function Test-DataMonitoring {
    Write-Info "🧪 Testing Data Monitoring Pipeline..."
    
    # Run test script
    python scripts/test_data_monitoring.py --webhook-url $WEBHOOK_URL
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "✅ All tests passed!"
        return $true
    } else {
        Write-Error "❌ Some tests failed"
        return $false
    }
}

function Trigger-Manual {
    Write-Info "🔄 Triggering manual data monitoring..."
    
    if ($Force) {
        Write-Info "🔥 Force trigger requested"
        python scripts/data_monitoring.py --force-trigger --data-source $DataSource
    } else {
        python scripts/data_monitoring.py --data-source $DataSource
    }
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "✅ Manual trigger completed"
        return $true
    } else {
        Write-Error "❌ Manual trigger failed"
        return $false
    }
}

function Show-Logs {
    Write-Info "📋 Showing logs for: $Component"
    
    switch ($Component) {
        "webhook" {
            docker compose -f $COMPOSE_FILE logs -f data-webhook-server
        }
        "scheduler" {
            docker compose -f $COMPOSE_FILE logs -f data-monitoring-scheduler
        }
        "all" {
            docker compose -f $COMPOSE_FILE logs -f
        }
        "monitoring" {
            if (Test-Path "logs/data_monitoring.log") {
                Get-Content "logs/data_monitoring.log" -Wait
            } else {
                Write-Warning "Monitoring log file not found"
            }
        }
        default {
            docker compose -f $COMPOSE_FILE logs -f $Component
        }
    }
}

function Restart-DataMonitoring {
    Write-Info "🔄 Restarting Data Monitoring Pipeline..."
    
    Stop-DataMonitoring
    Start-Sleep -Seconds 5
    Start-DataMonitoring -Build:$Build
}

function Build-Images {
    Write-Info "🔨 Building Data Monitoring Docker Images..."
    
    docker compose -f $COMPOSE_FILE build
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "✅ Images built successfully"
        return $true
    } else {
        Write-Error "❌ Failed to build images"
        return $false
    }
}

# Main execution
Write-Info "🚀 Data Monitoring Pipeline Manager"
Write-Info "===================================="

# Create necessary directories
New-Item -ItemType Directory -Force -Path "logs", "data", "config" | Out-Null

# Execute action
switch ($Action) {
    "start" {
        $success = Start-DataMonitoring
    }
    "stop" {
        $success = Stop-DataMonitoring
    }
    "status" {
        Show-Status
        $success = $true
    }
    "test" {
        $success = Test-DataMonitoring
    }
    "trigger" {
        $success = Trigger-Manual
    }
    "logs" {
        Show-Logs
        $success = $true
    }
    "restart" {
        $success = Restart-DataMonitoring
    }
    "build" {
        $success = Build-Images
    }
}

# Exit with appropriate code
if ($success) {
    Write-Success "🎉 Operation completed successfully!"
    exit 0
} else {
    Write-Error "💥 Operation failed!"
    exit 1
}
