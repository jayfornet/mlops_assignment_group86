#!/bin/bash
# API Health Check Runner
# This script runs the API health check Python script

set -e

# Default values
HOST="localhost"
PORT=8000
MAX_RETRIES=5
RETRY_DELAY=10

# Display usage information
function show_usage {
  echo "Usage: $0 [options]"
  echo "Options:"
  echo "  --host HOST       API host (default: localhost)"
  echo "  --port PORT       API port (default: 8000)"
  echo "  --retries N       Maximum retry attempts (default: 5)"
  echo "  --delay SEC       Delay between retries in seconds (default: 10)"
  echo "  --help            Show this help message"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --retries)
      MAX_RETRIES="$2"
      shift 2
      ;;
    --delay)
      RETRY_DELAY="$2"
      shift 2
      ;;
    --help)
      show_usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      show_usage
      exit 1
      ;;
  esac
done

echo "📋 Running API health checks"
echo "Host: $HOST"
echo "Port: $PORT"
echo "Max retries: $MAX_RETRIES"
echo "Retry delay: $RETRY_DELAY seconds"

# Check if Python is available
if ! command -v python &> /dev/null; then
  echo "❌ Python not found. Please install Python."
  exit 1
fi

# Check if required Python packages are installed
echo "🔍 Checking required Python packages..."
python -c "import requests" &> /dev/null || {
  echo "📦 Installing requests package..."
  python -m pip install requests
}

# Execute the health check script
echo "🚀 Starting health checks..."
python scripts/api_health_check.py --host "$HOST" --port "$PORT"
RESULT=$?

if [ $RESULT -eq 0 ]; then
  echo "✅ All health checks passed!"
else
  echo "❌ Some health checks failed. See logs above for details."
  exit $RESULT
fi
