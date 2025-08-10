# Simple Data Update and MLOps Trigger Script
# Perfect for assignment demonstrations

param(
    [ValidateSet("check", "update", "trigger")]
    [string]$Action = "check",
    
    [ValidateSet("add_rows", "modify_values", "add_noise")]
    [string]$UpdateType = "add_rows",
    
    [switch]$Force
)

Write-Host "🚀 Simple MLOps Data Trigger Tool" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green

# Create data directory
if (-not (Test-Path "data")) {
    New-Item -ItemType Directory -Path "data" | Out-Null
}

function Test-DataChanges {
    Write-Host "🔍 Checking for data changes..." -ForegroundColor Cyan
    
    $dataFile = "data/california_housing.csv"
    $hashFile = "data/last_hash.txt"
    
    if (-not (Test-Path $dataFile)) {
        Write-Host "❌ Data file not found: $dataFile" -ForegroundColor Red
        return $false
    }
    
    # Calculate current hash (simple approach)
    $currentHash = (Get-FileHash $dataFile -Algorithm MD5).Hash
    
    # Get previous hash
    $previousHash = ""
    if (Test-Path $hashFile) {
        $previousHash = Get-Content $hashFile -Raw
    }
    
    $changed = $currentHash -ne $previousHash
    
    # Save current hash
    $currentHash | Out-File $hashFile -NoNewline
    
    Write-Host "📊 Data Analysis:" -ForegroundColor Yellow
    Write-Host "  - Changed: $changed" -ForegroundColor $(if($changed) {"Red"} else {"Green"})
    Write-Host "  - Current Hash: $($currentHash.Substring(0,8))..." -ForegroundColor Gray
    Write-Host "  - File Size: $((Get-Item $dataFile).Length) bytes" -ForegroundColor Gray
    
    return $changed
}

function Invoke-DataUpdate {
    Write-Host "📊 Simulating data update: $UpdateType" -ForegroundColor Cyan
    
    # Simple data update simulation
    python scripts/update_data_and_trigger.py --simulate-update $UpdateType
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ Data update simulation completed" -ForegroundColor Green
        return $true
    } else {
        Write-Host "❌ Data update failed" -ForegroundColor Red
        return $false
    }
}

function Invoke-MLOpsTrigger {
    Write-Host "🚀 Triggering MLOps Pipeline..." -ForegroundColor Cyan
    
    # Try GitHub CLI first
    try {
        gh workflow run mlops-pipeline.yml --field triggered_by="powershell-script" --field trigger_reason="Data updated via PowerShell script"
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ Pipeline triggered via GitHub CLI" -ForegroundColor Green
            return $true
        }
    } catch {
        Write-Host "⚠️  GitHub CLI not available" -ForegroundColor Yellow
    }
    
    # Manual instructions
    Write-Host ""
    Write-Host "🔧 Manual Trigger Instructions:" -ForegroundColor Yellow
    Write-Host "1. Open: https://github.com/jayfornet/mlops_assignment_group86/actions" -ForegroundColor White
    Write-Host "2. Click 'MLOps Pipeline - California Housing Prediction'" -ForegroundColor White
    Write-Host "3. Click 'Run workflow' button" -ForegroundColor White
    Write-Host "4. Fill triggered_by: 'powershell-script'" -ForegroundColor White
    Write-Host "5. Click 'Run workflow'" -ForegroundColor White
    
    return $false
}

# Main execution
switch ($Action) {
    "check" {
        $changed = Test-DataChanges
        if ($changed) {
            Write-Host "🎯 Data changed - ready to trigger pipeline!" -ForegroundColor Green
        } else {
            Write-Host "📋 No changes detected" -ForegroundColor Gray
        }
    }
    
    "update" {
        $success = Invoke-DataUpdate
        if ($success) {
            $changed = Test-DataChanges
            Write-Host "📈 Update complete - data changed: $changed" -ForegroundColor Green
        }
    }
    
    "trigger" {
        if ($Force) {
            Write-Host "🔥 Force triggering pipeline..." -ForegroundColor Red
            Invoke-MLOpsTrigger
        } else {
            $changed = Test-DataChanges
            if ($changed) {
                Write-Host "📊 Data changed - triggering pipeline..." -ForegroundColor Green
                Invoke-MLOpsTrigger
            } else {
                Write-Host "📋 No changes detected - use -Force to trigger anyway" -ForegroundColor Gray
            }
        }
    }
}

Write-Host ""
Write-Host "🎉 Script completed at $(Get-Date)" -ForegroundColor Green
