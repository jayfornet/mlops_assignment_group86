# PowerShell script to run all health checks for the application

Write-Host "Starting health checks at $(Get-Date)"

# Get script directory and project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

# Check if API is running
Write-Host "Checking if API is running..."
$ApiUrl = if ($env:API_URL) { $env:API_URL } else { "http://localhost:8000" }
python "$ScriptDir\api_health_check.py" --url $ApiUrl --retries 5 --retry-delay 3

if ($LASTEXITCODE -ne 0) {
    Write-Host "API health check failed!" -ForegroundColor Red
    exit 1
}

# Validate models
Write-Host "Validating models..."
python "$ScriptDir\validate_models.py" "$ProjectRoot\models"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Model validation failed!" -ForegroundColor Red
    exit 1
}

# Check database connectivity (if applicable)
Write-Host "Checking database connectivity..."
# Add database check here if needed

Write-Host "All health checks passed at $(Get-Date)" -ForegroundColor Green
exit 0
