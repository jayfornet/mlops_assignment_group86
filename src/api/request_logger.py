"""
Request and Response Logger for the California Housing Prediction API.

This module provides comprehensive logging of all API requests and responses
to both SQLite database and plain text files for monitoring and debugging.
"""

import sqlite3
import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import Request, Response
import uuid


class RequestResponseLogger:
    """
    Comprehensive logger for API requests and responses.
    Logs to both SQLite database and plain text files.
    """
    
    def __init__(self, db_path: str = "logs/api_requests.db", log_file: str = "logs/api_requests.log"):
        self.db_path = db_path
        self.log_file = log_file
        
        # Ensure logs directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Initialize database and file logger
        self.init_database()
        self.init_file_logger()
    
    def init_database(self):
        """Initialize SQLite database for request/response logging."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create requests table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS api_requests (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    method TEXT NOT NULL,
                    url TEXT NOT NULL,
                    path TEXT NOT NULL,
                    headers TEXT,
                    query_params TEXT,
                    body TEXT,
                    client_ip TEXT,
                    user_agent TEXT,
                    request_size INTEGER,
                    processing_start TEXT
                )
            ''')
            
            # Create responses table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS api_responses (
                    id TEXT PRIMARY KEY,
                    request_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    status_code INTEGER NOT NULL,
                    headers TEXT,
                    body TEXT,
                    response_size INTEGER,
                    processing_time_ms REAL,
                    success BOOLEAN,
                    error_message TEXT,
                    FOREIGN KEY (request_id) REFERENCES api_requests (id)
                )
            ''')
            
            # Create combined view for easier querying
            cursor.execute('''
                CREATE VIEW IF NOT EXISTS request_response_log AS
                SELECT 
                    r.id as request_id,
                    r.timestamp as request_timestamp,
                    r.method,
                    r.url,
                    r.path,
                    r.client_ip,
                    r.user_agent,
                    r.request_size,
                    resp.timestamp as response_timestamp,
                    resp.status_code,
                    resp.response_size,
                    resp.processing_time_ms,
                    resp.success,
                    resp.error_message
                FROM api_requests r
                LEFT JOIN api_responses resp ON r.id = resp.request_id
                ORDER BY r.timestamp DESC
            ''')
            
            conn.commit()
            conn.close()
            logging.info("Request/Response database initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing request/response database: {str(e)}")
    
    def init_file_logger(self):
        """Initialize file logger for request/response logging."""
        try:
            # Create a separate logger for requests/responses
            self.file_logger = logging.getLogger('api_requests')
            self.file_logger.setLevel(logging.INFO)
            
            # Remove existing handlers to avoid duplicates
            self.file_logger.handlers.clear()
            
            # Create file handler
            file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            
            self.file_logger.addHandler(file_handler)
            
            # Prevent propagation to root logger
            self.file_logger.propagate = False
            
            logging.info("Request/Response file logger initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing request/response file logger: {str(e)}")
    
    def log_request(self, request: Request, request_id: str, body: str = None) -> str:
        """
        Log incoming request to database and file.
        
        Args:
            request: FastAPI Request object
            request_id: Unique identifier for the request
            body: Request body content
            
        Returns:
            str: Request ID for correlation with response
        """
        try:
            timestamp = datetime.now().isoformat()
            
            # Extract request information
            method = request.method
            url = str(request.url)
            path = request.url.path
            headers = dict(request.headers)
            query_params = dict(request.query_params)
            client_ip = self._get_client_ip(request)
            user_agent = headers.get('user-agent', 'Unknown')
            request_size = len(body.encode('utf-8')) if body else 0
            
            # Sanitize sensitive headers
            sanitized_headers = self._sanitize_headers(headers)
            
            # Log to database
            self._log_request_to_db(
                request_id, timestamp, method, url, path,
                sanitized_headers, query_params, body, client_ip, user_agent, request_size
            )
            
            # Log to file
            self._log_request_to_file(
                request_id, timestamp, method, url, path,
                client_ip, user_agent, query_params, body, request_size
            )
            
            return request_id
            
        except Exception as e:
            logging.error(f"Error logging request: {str(e)}")
            return request_id
    
    def log_response(self, request_id: str, response: Response, body: str = None, 
                    processing_time: float = None, error_message: str = None):
        """
        Log outgoing response to database and file.
        
        Args:
            request_id: Request ID for correlation
            response: FastAPI Response object
            body: Response body content
            processing_time: Time taken to process request (in milliseconds)
            error_message: Error message if request failed
        """
        try:
            timestamp = datetime.now().isoformat()
            
            # Extract response information
            status_code = response.status_code
            headers = dict(response.headers) if hasattr(response, 'headers') else {}
            response_size = len(body.encode('utf-8')) if body else 0
            success = 200 <= status_code < 400
            
            # Sanitize sensitive headers
            sanitized_headers = self._sanitize_headers(headers)
            
            # Log to database
            self._log_response_to_db(
                request_id, timestamp, status_code, sanitized_headers,
                body, response_size, processing_time, success, error_message
            )
            
            # Log to file
            self._log_response_to_file(
                request_id, timestamp, status_code, response_size,
                processing_time, success, error_message, body
            )
            
        except Exception as e:
            logging.error(f"Error logging response: {str(e)}")
    
    def _log_request_to_db(self, request_id: str, timestamp: str, method: str, url: str, 
                          path: str, headers: Dict, query_params: Dict, body: str,
                          client_ip: str, user_agent: str, request_size: int):
        """Log request to SQLite database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO api_requests 
                (id, timestamp, method, url, path, headers, query_params, body, 
                 client_ip, user_agent, request_size, processing_start)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                request_id, timestamp, method, url, path,
                json.dumps(headers), json.dumps(query_params), body,
                client_ip, user_agent, request_size, timestamp
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"Error logging request to database: {str(e)}")
    
    def _log_response_to_db(self, request_id: str, timestamp: str, status_code: int,
                           headers: Dict, body: str, response_size: int, processing_time: float,
                           success: bool, error_message: str):
        """Log response to SQLite database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO api_responses 
                (id, request_id, timestamp, status_code, headers, body, response_size,
                 processing_time_ms, success, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                f"{request_id}_resp", request_id, timestamp, status_code,
                json.dumps(headers), body, response_size, processing_time, success, error_message
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logging.error(f"Error logging response to database: {str(e)}")
    
    def _log_request_to_file(self, request_id: str, timestamp: str, method: str, url: str,
                            path: str, client_ip: str, user_agent: str, query_params: Dict,
                            body: str, request_size: int):
        """Log request to plain text file."""
        try:
            log_entry = {
                "type": "REQUEST",
                "id": request_id,
                "timestamp": timestamp,
                "method": method,
                "url": url,
                "path": path,
                "client_ip": client_ip,
                "user_agent": user_agent,
                "query_params": query_params,
                "request_size": request_size
            }
            
            # Add body for POST/PUT requests (truncate if too long)
            if body and method in ['POST', 'PUT', 'PATCH']:
                if len(body) > 1000:
                    log_entry["body"] = body[:1000] + "... [TRUNCATED]"
                else:
                    log_entry["body"] = body
            
            self.file_logger.info(f"REQUEST | {json.dumps(log_entry, ensure_ascii=False)}")
            
        except Exception as e:
            logging.error(f"Error logging request to file: {str(e)}")
    
    def _log_response_to_file(self, request_id: str, timestamp: str, status_code: int,
                             response_size: int, processing_time: float, success: bool,
                             error_message: str, body: str):
        """Log response to plain text file."""
        try:
            log_entry = {
                "type": "RESPONSE",
                "request_id": request_id,
                "timestamp": timestamp,
                "status_code": status_code,
                "response_size": response_size,
                "processing_time_ms": processing_time,
                "success": success
            }
            
            if error_message:
                log_entry["error_message"] = error_message
            
            # Add response body for errors or if small enough
            if body and (not success or len(body) < 500):
                if len(body) > 1000:
                    log_entry["body"] = body[:1000] + "... [TRUNCATED]"
                else:
                    log_entry["body"] = body
            
            self.file_logger.info(f"RESPONSE | {json.dumps(log_entry, ensure_ascii=False)}")
            
        except Exception as e:
            logging.error(f"Error logging response to file: {str(e)}")
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers first (for reverse proxy setups)
        forwarded_for = request.headers.get('x-forwarded-for')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        real_ip = request.headers.get('x-real-ip')
        if real_ip:
            return real_ip
        
        # Fallback to direct client
        return request.client.host if request.client else "unknown"
    
    def _sanitize_headers(self, headers: Dict) -> Dict:
        """Remove sensitive information from headers."""
        sensitive_headers = ['authorization', 'cookie', 'x-api-key', 'x-auth-token']
        sanitized = {}
        
        for key, value in headers.items():
            if key.lower() in sensitive_headers:
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = value
        
        return sanitized
    
    def get_recent_requests(self, limit: int = 50) -> list:
        """
        Get recent requests and responses from database.
        
        Args:
            limit: Maximum number of requests to return
            
        Returns:
            List of request/response pairs
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM request_response_log
                ORDER BY request_timestamp DESC
                LIMIT ?
            ''', (limit,))
            
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            conn.close()
            
            requests = []
            for row in rows:
                request_data = dict(zip(columns, row))
                requests.append(request_data)
            
            return requests
            
        except Exception as e:
            logging.error(f"Error fetching recent requests: {str(e)}")
            return []
    
    def get_request_stats(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get request statistics for the specified time period.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Dictionary with request statistics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Calculate time threshold
            from datetime import datetime, timedelta
            threshold = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            # Total requests
            cursor.execute('''
                SELECT COUNT(*) FROM api_requests 
                WHERE timestamp > ?
            ''', (threshold,))
            total_requests = cursor.fetchone()[0]
            
            # Successful requests
            cursor.execute('''
                SELECT COUNT(*) FROM request_response_log 
                WHERE request_timestamp > ? AND success = 1
            ''', (threshold,))
            successful_requests = cursor.fetchone()[0]
            
            # Failed requests
            cursor.execute('''
                SELECT COUNT(*) FROM request_response_log 
                WHERE request_timestamp > ? AND success = 0
            ''', (threshold,))
            failed_requests = cursor.fetchone()[0]
            
            # Average response time
            cursor.execute('''
                SELECT AVG(processing_time_ms) FROM request_response_log 
                WHERE request_timestamp > ? AND processing_time_ms IS NOT NULL
            ''', (threshold,))
            avg_response_time = cursor.fetchone()[0] or 0
            
            # Most common endpoints
            cursor.execute('''
                SELECT path, COUNT(*) as count FROM api_requests 
                WHERE timestamp > ?
                GROUP BY path
                ORDER BY count DESC
                LIMIT 10
            ''', (threshold,))
            common_endpoints = [{"path": row[0], "count": row[1]} for row in cursor.fetchall()]
            
            # Status code distribution
            cursor.execute('''
                SELECT status_code, COUNT(*) as count FROM request_response_log 
                WHERE request_timestamp > ?
                GROUP BY status_code
                ORDER BY count DESC
            ''', (threshold,))
            status_codes = [{"status_code": row[0], "count": row[1]} for row in cursor.fetchall()]
            
            conn.close()
            
            return {
                "period_hours": hours,
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "failed_requests": failed_requests,
                "success_rate": round((successful_requests / total_requests * 100), 2) if total_requests > 0 else 0,
                "average_response_time_ms": round(avg_response_time, 2),
                "common_endpoints": common_endpoints,
                "status_code_distribution": status_codes
            }
            
        except Exception as e:
            logging.error(f"Error getting request statistics: {str(e)}")
            return {}


# Global instance
request_logger = RequestResponseLogger()
