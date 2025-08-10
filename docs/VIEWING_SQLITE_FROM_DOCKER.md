# Viewing SQLite Database from Docker Images

This guide shows you multiple ways to view and analyze the SQLite database data from your Docker containers.

## 📊 Database Locations

The API creates these SQLite databases in the `logs/` directory:
- **`logs/api_requests.db`** - All API requests and responses
- **`logs/predictions.db`** - Prediction-specific data

## 🐳 Method 1: Using Docker Commands

### Copy Database from Running Container
```bash
# Copy database from running container to host
docker cp housing-prediction-api:/app/logs/api_requests.db ./api_requests.db

# View the copied database
python scripts/view_sqlite.py api_requests.db
```

### Execute Commands Inside Container
```bash
# Access container shell
docker exec -it housing-prediction-api bash

# Run SQLite commands inside container
sqlite3 /app/logs/api_requests.db "SELECT * FROM api_requests LIMIT 5;"

# Or use Python script inside container
python -c "
import sqlite3
conn = sqlite3.connect('/app/logs/api_requests.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM api_requests')
print('Total requests:', cursor.fetchone()[0])
conn.close()
"
```

### One-liner to View Recent Requests
```bash
# View recent requests from running container
docker exec housing-prediction-api sqlite3 /app/logs/api_requests.db \
  "SELECT timestamp, method, path, status_code FROM request_response_log ORDER BY request_timestamp DESC LIMIT 10;"
```

## 🔧 Method 2: Using API Endpoints

### View Data via HTTP APIs
```bash
# Get recent requests (if API is running)
curl http://localhost:8000/logs/requests?limit=20

# Get statistics
curl http://localhost:8000/logs/stats?hours=24

# Download data as JSON
curl "http://localhost:8000/logs/download?format=json&hours=24" -o api_logs.json

# Download as CSV
curl "http://localhost:8000/logs/download?format=csv&hours=24" -o api_logs.csv
```

## 🛠️ Method 3: Using Python Scripts

### Use the Built-in SQLite Viewer
```bash
# Interactive viewer (works with copied database)
python scripts/view_sqlite.py

# Direct database access
python scripts/view_sqlite.py logs/api_requests.db

# Inside Docker container
docker exec -it housing-prediction-api python scripts/view_sqlite.py /app/logs/api_requests.db
```

### Quick Python One-liners
```bash
# Inside Docker container - view table structure
docker exec housing-prediction-api python -c "
import sqlite3
conn = sqlite3.connect('/app/logs/api_requests.db')
cursor = conn.cursor()
cursor.execute('SELECT name FROM sqlite_master WHERE type=\"table\"')
print('Tables:', [row[0] for row in cursor.fetchall()])
conn.close()
"

# View recent requests
docker exec housing-prediction-api python -c "
import sqlite3, json
conn = sqlite3.connect('/app/logs/api_requests.db')
cursor = conn.cursor()
cursor.execute('SELECT * FROM request_response_log ORDER BY request_timestamp DESC LIMIT 5')
for row in cursor.fetchall():
    print(row)
conn.close()
"
```

## 📋 Method 4: Using Docker Compose

### Mount Database Directory
```yaml
# In docker-compose.yml, add volume mount
services:
  mlops-api:
    volumes:
      - ./logs:/app/logs  # This exposes logs directory to host
```

### Then access directly
```bash
# Start with mounted volume
docker-compose up -d

# Access database directly from host
python scripts/view_sqlite.py logs/api_requests.db
sqlite3 logs/api_requests.db "SELECT * FROM api_requests LIMIT 5;"
```

## 🔍 Method 5: Using SQLite Browser Tools

### Install SQLite Browser
```bash
# On Ubuntu/Debian
sudo apt-get install sqlitebrowser

# On macOS
brew install --cask db-browser-for-sqlite

# On Windows
# Download from: https://sqlitebrowser.org/
```

### Use with Copied Database
```bash
# Copy database from container
docker cp housing-prediction-api:/app/logs/api_requests.db ./api_requests.db

# Open with GUI browser
sqlitebrowser api_requests.db
```

## 📊 Common SQL Queries

### Useful Queries for Analysis
```sql
-- Total number of requests
SELECT COUNT(*) as total_requests FROM api_requests;

-- Requests by endpoint
SELECT path, COUNT(*) as count 
FROM api_requests 
GROUP BY path 
ORDER BY count DESC;

-- Success rate by endpoint
SELECT 
    r.path,
    COUNT(*) as total_requests,
    SUM(CASE WHEN resp.success = 1 THEN 1 ELSE 0 END) as successful,
    ROUND(AVG(resp.processing_time_ms), 2) as avg_response_time
FROM api_requests r
LEFT JOIN api_responses resp ON r.id = resp.request_id
GROUP BY r.path
ORDER BY total_requests DESC;

-- Recent prediction requests
SELECT 
    r.timestamp,
    r.client_ip,
    r.body,
    resp.status_code,
    resp.processing_time_ms
FROM api_requests r
LEFT JOIN api_responses resp ON r.id = resp.request_id
WHERE r.path = '/predict'
ORDER BY r.timestamp DESC
LIMIT 10;

-- Error analysis
SELECT 
    resp.status_code,
    COUNT(*) as count,
    resp.error_message
FROM api_responses resp
WHERE resp.success = 0
GROUP BY resp.status_code, resp.error_message
ORDER BY count DESC;

-- Performance analysis
SELECT 
    r.path,
    MIN(resp.processing_time_ms) as min_time,
    MAX(resp.processing_time_ms) as max_time,
    AVG(resp.processing_time_ms) as avg_time,
    COUNT(*) as request_count
FROM api_requests r
JOIN api_responses resp ON r.id = resp.request_id
WHERE resp.processing_time_ms IS NOT NULL
GROUP BY r.path
ORDER BY avg_time DESC;
```

## 🔧 Automation Scripts

### Create Database Export Script
```bash
#!/bin/bash
# save as: export_db.sh

CONTAINER_NAME="housing-prediction-api"
OUTPUT_DIR="./exported_data"

# Create output directory
mkdir -p $OUTPUT_DIR

# Copy database files
docker cp $CONTAINER_NAME:/app/logs/api_requests.db $OUTPUT_DIR/
docker cp $CONTAINER_NAME:/app/logs/predictions.db $OUTPUT_DIR/

# Export to different formats
docker exec $CONTAINER_NAME python -c "
import sqlite3, json, csv
import os

# Export requests to JSON
conn = sqlite3.connect('/app/logs/api_requests.db')
cursor = conn.cursor()
cursor.execute('SELECT * FROM request_response_log ORDER BY request_timestamp DESC LIMIT 1000')
columns = [desc[0] for desc in cursor.description]
rows = cursor.fetchall()
data = [dict(zip(columns, row)) for row in rows]

with open('/app/logs/requests_export.json', 'w') as f:
    json.dump(data, f, indent=2, default=str)

print('Exported', len(data), 'requests to JSON')
conn.close()
"

# Copy exported files
docker cp $CONTAINER_NAME:/app/logs/requests_export.json $OUTPUT_DIR/

echo "Database exported to $OUTPUT_DIR/"
```

### Make it executable
```bash
chmod +x export_db.sh
./export_db.sh
```

## 🚀 Quick Commands Summary

```bash
# 1. Copy database from container
docker cp housing-prediction-api:/app/logs/api_requests.db ./

# 2. View with Python script
python scripts/view_sqlite.py api_requests.db

# 3. Quick SQLite command
sqlite3 api_requests.db "SELECT COUNT(*) FROM api_requests;"

# 4. View via API (if running)
curl http://localhost:8000/logs/requests?limit=10

# 5. Execute inside container
docker exec -it housing-prediction-api sqlite3 /app/logs/api_requests.db

# 6. Export and view
docker exec housing-prediction-api python scripts/view_sqlite.py /app/logs/api_requests.db
```

## 🐛 Troubleshooting

### Database Not Found
```bash
# Check if logs directory exists in container
docker exec housing-prediction-api ls -la /app/logs/

# Check if API is creating logs
docker exec housing-prediction-api ls -la /app/logs/*.db

# Check container logs
docker logs housing-prediction-api | grep -i database
```

### Permission Issues
```bash
# Fix file permissions after copying
chmod 644 api_requests.db

# Or copy with different user
docker cp housing-prediction-api:/app/logs/api_requests.db ./api_requests.db
sudo chown $USER:$USER api_requests.db
```

### Empty Database
```bash
# Make some requests to populate database
curl http://localhost:8000/health
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"MedInc": 8.3252, "HouseAge": 41.0, "AveRooms": 6.984, "AveBedrms": 1.023, "Population": 322.0, "AveOccup": 2.555, "Latitude": 37.88, "Longitude": -122.23}'

# Then check database again
```

## 📱 GUI Tools

### DB Browser for SQLite (Recommended)
- **Download**: https://sqlitebrowser.org/
- **Features**: Visual table browser, SQL query editor, data export
- **Usage**: Copy database file and open with the browser

### SQLite Studio
- **Download**: https://sqlitestudio.pl/
- **Features**: Advanced SQL IDE, plugins, data visualization

### DBeaver (Universal Database Tool)
- **Download**: https://dbeaver.io/
- **Features**: Supports many databases including SQLite, advanced features

Choose the method that works best for your setup! The Python scripts provide the most flexibility, while GUI tools offer the best visual experience.
