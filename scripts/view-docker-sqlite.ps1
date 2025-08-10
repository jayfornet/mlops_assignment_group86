# PowerShell Script to View SQLite Database from Docker Container
# Usage: .\view-docker-sqlite.ps1

param(
    [string]$ContainerName = "housing-prediction-api",
    [string]$OutputDir = ".\exported_logs",
    [string]$Action = "copy-and-view"
)

Write-Host "🏠 California Housing API - SQLite Database Viewer" -ForegroundColor Green
Write-Host "=" * 60

# Check if Docker is running
try {
    docker version | Out-Null
    Write-Host "✅ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is not running or not installed" -ForegroundColor Red
    exit 1
}

# Check if container exists and is running
$containerExists = docker ps -q -f name=$ContainerName
if (-not $containerExists) {
    Write-Host "❌ Container '$ContainerName' is not running" -ForegroundColor Red
    Write-Host "Available containers:" -ForegroundColor Yellow
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    exit 1
}

Write-Host "✅ Container '$ContainerName' is running" -ForegroundColor Green

# Create output directory
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
    Write-Host "📁 Created output directory: $OutputDir" -ForegroundColor Yellow
}

function Copy-DatabaseFiles {
    Write-Host "📥 Copying database files from container..." -ForegroundColor Blue
    
    try {
        # Copy API requests database
        docker cp "${ContainerName}:/app/logs/api_requests.db" "$OutputDir\api_requests.db"
        if (Test-Path "$OutputDir\api_requests.db") {
            $size = (Get-Item "$OutputDir\api_requests.db").Length
            Write-Host "✅ Copied api_requests.db ($size bytes)" -ForegroundColor Green
        }
        
        # Copy predictions database
        docker cp "${ContainerName}:/app/logs/predictions.db" "$OutputDir\predictions.db" 2>$null
        if (Test-Path "$OutputDir\predictions.db") {
            $size = (Get-Item "$OutputDir\predictions.db").Length
            Write-Host "✅ Copied predictions.db ($size bytes)" -ForegroundColor Green
        }
        
        # Copy log files
        docker cp "${ContainerName}:/app/logs/api_requests.log" "$OutputDir\api_requests.log" 2>$null
        if (Test-Path "$OutputDir\api_requests.log") {
            $size = (Get-Item "$OutputDir\api_requests.log").Length
            Write-Host "✅ Copied api_requests.log ($size bytes)" -ForegroundColor Green
        }
        
    } catch {
        Write-Host "❌ Error copying files: $_" -ForegroundColor Red
    }
}

function Show-DatabaseInfo {
    param([string]$DbPath)
    
    if (-not (Test-Path $DbPath)) {
        Write-Host "❌ Database not found: $DbPath" -ForegroundColor Red
        return
    }
    
    Write-Host "📊 Database Information: $DbPath" -ForegroundColor Blue
    Write-Host "-" * 50
    
    # Use Python to show database info
    $pythonScript = @"
import sqlite3
import sys

try:
    conn = sqlite3.connect('$DbPath')
    cursor = conn.cursor()
    
    # Get tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    print("Tables in database:")
    for table in tables:
        table_name = table[0]
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"  📋 {table_name}: {count} rows")
    
    # Show recent requests if api_requests table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='api_requests'")
    if cursor.fetchone():
        print("\n🕒 Recent API Requests:")
        cursor.execute("""
            SELECT 
                r.timestamp,
                r.method,
                r.path,
                COALESCE(resp.status_code, 'N/A') as status
            FROM api_requests r
            LEFT JOIN api_responses resp ON r.id = resp.request_id
            ORDER BY r.timestamp DESC
            LIMIT 5
        """)
        
        for row in cursor.fetchall():
            timestamp = row[0][:19] if row[0] else 'N/A'
            print(f"  {timestamp} | {row[1]} | {row[2]} | {row[3]}")
    
    conn.close()
    
except Exception as e:
    print(f"Error: {e}")
"@
    
    $pythonScript | python
}

function Show-Menu {
    Write-Host "`n🔍 Choose an action:" -ForegroundColor Yellow
    Write-Host "1. Copy databases from container"
    Write-Host "2. View database info"
    Write-Host "3. Export to JSON"
    Write-Host "4. Quick container info"
    Write-Host "5. Open logs directory"
    Write-Host "6. Exit"
    
    $choice = Read-Host "`nEnter choice (1-6)"
    return $choice
}

function Export-ToJSON {
    $dbPath = "$OutputDir\api_requests.db"
    if (-not (Test-Path $dbPath)) {
        Write-Host "❌ Database not found. Copy it first." -ForegroundColor Red
        return
    }
    
    Write-Host "📤 Exporting to JSON..." -ForegroundColor Blue
    
    $pythonScript = @"
import sqlite3
import json
from datetime import datetime

try:
    conn = sqlite3.connect('$dbPath')
    cursor = conn.cursor()
    
    # Export recent requests
    cursor.execute('''
        SELECT 
            r.id,
            r.timestamp,
            r.method,
            r.path,
            r.client_ip,
            resp.status_code,
            resp.processing_time_ms,
            resp.success
        FROM api_requests r
        LEFT JOIN api_responses resp ON r.id = resp.request_id
        ORDER BY r.timestamp DESC
        LIMIT 100
    ''')
    
    columns = ['id', 'timestamp', 'method', 'path', 'client_ip', 'status_code', 'processing_time_ms', 'success']
    rows = cursor.fetchall()
    
    data = []
    for row in rows:
        data.append(dict(zip(columns, row)))
    
    output_file = '$OutputDir\\api_requests_export.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)
    
    print(f"✅ Exported {len(data)} requests to {output_file}")
    
    conn.close()
    
except Exception as e:
    print(f"❌ Error: {e}")
"@
    
    $pythonScript | python
}

function Show-ContainerInfo {
    Write-Host "🐳 Container Information:" -ForegroundColor Blue
    
    # Container details
    docker inspect $ContainerName --format "Container ID: {{.Id}}" 2>$null
    docker inspect $ContainerName --format "Created: {{.Created}}" 2>$null
    docker inspect $ContainerName --format "Status: {{.State.Status}}" 2>$null
    
    Write-Host "`n📁 Log files in container:" -ForegroundColor Blue
    docker exec $ContainerName ls -la /app/logs/ 2>$null
    
    Write-Host "`n📊 Quick database stats:" -ForegroundColor Blue
    docker exec $ContainerName python -c "
import sqlite3
import os

db_path = '/app/logs/api_requests.db'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM api_requests')
    count = cursor.fetchone()[0]
    print(f'Total API requests: {count}')
    conn.close()
else:
    print('Database not found')
" 2>$null
}

# Main execution
switch ($Action) {
    "copy-and-view" {
        Copy-DatabaseFiles
        if (Test-Path "$OutputDir\api_requests.db") {
            Show-DatabaseInfo "$OutputDir\api_requests.db"
        }
    }
    "interactive" {
        do {
            $choice = Show-Menu
            
            switch ($choice) {
                "1" { Copy-DatabaseFiles }
                "2" { 
                    if (Test-Path "$OutputDir\api_requests.db") {
                        Show-DatabaseInfo "$OutputDir\api_requests.db"
                    } else {
                        Write-Host "❌ Database not found. Copy it first." -ForegroundColor Red
                    }
                }
                "3" { Export-ToJSON }
                "4" { Show-ContainerInfo }
                "5" { 
                    if (Test-Path $OutputDir) {
                        Invoke-Item $OutputDir
                    } else {
                        Write-Host "❌ Output directory doesn't exist" -ForegroundColor Red
                    }
                }
                "6" { 
                    Write-Host "👋 Goodbye!" -ForegroundColor Green
                    break
                }
                default { Write-Host "❌ Invalid choice" -ForegroundColor Red }
            }
        } while ($choice -ne "6")
    }
    default {
        Write-Host "Usage:" -ForegroundColor Yellow
        Write-Host "  .\view-docker-sqlite.ps1                          # Copy and view"
        Write-Host "  .\view-docker-sqlite.ps1 -Action interactive      # Interactive menu"
        Write-Host "  .\view-docker-sqlite.ps1 -ContainerName myapi     # Custom container name"
    }
}

Write-Host "`n📂 Output directory: $OutputDir" -ForegroundColor Green
if (Test-Path "$OutputDir\api_requests.db") {
    Write-Host "🔍 You can now use:" -ForegroundColor Yellow
    Write-Host "  python scripts\view_sqlite.py $OutputDir\api_requests.db"
    Write-Host "  sqlite3 $OutputDir\api_requests.db"
}
