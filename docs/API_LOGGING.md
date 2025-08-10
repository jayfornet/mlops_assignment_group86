# API Request/Response Logging

This document describes the comprehensive request and response logging system implemented for the California Housing Prediction API.

## 📋 Overview

The API now logs ALL incoming requests and outgoing responses to both:
- **SQLite Database** (`logs/api_requests.db`) - Structured data for querying and analysis
- **Plain Text File** (`logs/api_requests.log`) - Human-readable logs for debugging

## 🗄️ Database Schema

### `api_requests` Table
- `id` - Unique request identifier
- `timestamp` - Request timestamp
- `method` - HTTP method (GET, POST, etc.)
- `url` - Full request URL
- `path` - Request path
- `headers` - Request headers (JSON)
- `query_params` - Query parameters (JSON)
- `body` - Request body content
- `client_ip` - Client IP address
- `user_agent` - User agent string
- `request_size` - Request size in bytes
- `processing_start` - Processing start timestamp

### `api_responses` Table
- `id` - Unique response identifier
- `request_id` - Foreign key to request
- `timestamp` - Response timestamp
- `status_code` - HTTP status code
- `headers` - Response headers (JSON)
- `body` - Response body content
- `response_size` - Response size in bytes
- `processing_time_ms` - Processing time in milliseconds
- `success` - Boolean success flag
- `error_message` - Error message if failed

## 📊 Available Endpoints

### View Recent Requests
```bash
GET /logs/requests?limit=50
```
Returns recent API requests and responses.

### View Statistics
```bash
GET /logs/stats?hours=24
```
Returns usage statistics for the specified time period.

### Download Logs
```bash
GET /logs/download?format=json&hours=24
```
Download logs in JSON, CSV, or TXT format.

**Parameters:**
- `format`: `json`, `csv`, or `txt`
- `hours`: Number of hours to look back

## 🛠️ Tools and Scripts

### Log Viewer Script
View logs interactively or via command line:

```bash
# Interactive mode
python scripts/view_logs.py

# Command line mode
python scripts/view_logs.py requests --limit 30 --details
python scripts/view_logs.py stats --hours 12
python scripts/view_logs.py logs --lines 100
```

### Test Logging Script
Test the logging functionality:

```bash
python scripts/test_api_logging.py
```

## 📁 Log File Locations

- **Request/Response Database**: `logs/api_requests.db`
- **Request/Response Text Log**: `logs/api_requests.log`
- **General API Log**: `logs/api.log`
- **Predictions Database**: `logs/predictions.db`

## 🔍 Log Format Examples

### Plain Text Log Entry (Request)
```
2025-08-10 15:30:45 - INFO - REQUEST | {"type": "REQUEST", "id": "req_a1b2c3d4e5f6", "timestamp": "2025-08-10T15:30:45.123456", "method": "POST", "url": "http://localhost:8000/predict", "path": "/predict", "client_ip": "127.0.0.1", "user_agent": "python-requests/2.28.0", "query_params": {}, "request_size": 245, "body": "{\n  \"MedInc\": 8.3252,\n  \"HouseAge\": 41.0,\n  \"AveRooms\": 6.984,\n  \"AveBedrms\": 1.023,\n  \"Population\": 322.0,\n  \"AveOccup\": 2.555,\n  \"Latitude\": 37.88,\n  \"Longitude\": -122.23\n}"}
```

### Plain Text Log Entry (Response)
```
2025-08-10 15:30:45 - INFO - RESPONSE | {"type": "RESPONSE", "request_id": "req_a1b2c3d4e5f6", "timestamp": "2025-08-10T15:30:45.156789", "status_code": 200, "response_size": 312, "processing_time_ms": 33.4, "success": true}
```

## 🔒 Security Features

- **Header Sanitization**: Sensitive headers (authorization, cookies, API keys) are redacted
- **Body Size Limits**: Large request/response bodies are truncated in logs
- **Binary Content Handling**: Binary content is identified and not logged as text

## 📈 Monitoring Integration

The logging system integrates with:
- **Prometheus Metrics**: Request counters and response time histograms
- **Health Checks**: Log database status included in health endpoint
- **Grafana Dashboards**: Can query log database for detailed analytics

## 🚀 Usage Examples

### Start the API with Logging
```bash
cd src/api
python app.py
```

### Make Requests (automatically logged)
```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"MedInc": 8.3252, "HouseAge": 41.0, "AveRooms": 6.984, "AveBedrms": 1.023, "Population": 322.0, "AveOccup": 2.555, "Latitude": 37.88, "Longitude": -122.23}'
```

### View Logs
```bash
# View recent requests via API
curl "http://localhost:8000/logs/requests?limit=10"

# View statistics via API
curl "http://localhost:8000/logs/stats?hours=1"

# Download logs
curl "http://localhost:8000/logs/download?format=csv&hours=24" -o api_logs.csv

# Use log viewer script
python scripts/view_logs.py
```

## 🔧 Configuration

The logging system is automatically configured when the API starts. Key components:

1. **RequestResponseLogger**: Handles database and file logging
2. **RequestResponseLoggingMiddleware**: Captures all requests/responses
3. **Log Endpoints**: Provide access to logged data

No additional configuration is required - logging starts automatically when the API is launched.

## 📊 Performance Impact

The logging system is designed to be lightweight:
- Asynchronous logging to minimize request latency
- Database writes are non-blocking
- Large bodies are truncated to prevent memory issues
- Binary content is detected and skipped

Typical overhead: < 1-2ms per request.

## 🛠️ Troubleshooting

### Database Issues
If the database file is corrupted:
```bash
rm logs/api_requests.db
# Restart the API to recreate the database
```

### Log File Growth
Log files rotate naturally, but you can manually clean them:
```bash
# Archive old logs
mv logs/api_requests.log logs/api_requests.log.old
# Restart API to create new log file
```

### Viewing Logs in Docker
```bash
# Copy logs from container
docker cp housing-prediction-api:/app/logs ./logs

# Or mount logs directory when running
docker run -v ./logs:/app/logs housing-prediction-api
```
