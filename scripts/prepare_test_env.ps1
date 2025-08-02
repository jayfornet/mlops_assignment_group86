# PowerShell script to prepare testing environment
# Creates all required directories for tests to run

Write-Host "Creating necessary directories for testing..."

# Navigate to project root (assumes script is run from the root or scripts directory)
$projectRoot = if ($PSScriptRoot -match "scripts$") { Split-Path $PSScriptRoot -Parent } else { $PSScriptRoot }
Set-Location $projectRoot

# Create necessary directories
$directories = @("data", "models", "logs", "results", "mlruns", "mlflow-artifacts")

foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "Created directory: $dir"
    } else {
        Write-Host "Directory already exists: $dir"
    }
}

Write-Host "Environment preparation complete."
