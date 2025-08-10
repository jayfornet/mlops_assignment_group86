"""
Middleware for logging all API requests and responses.

This middleware captures all incoming requests and outgoing responses
and logs them using the RequestResponseLogger.
"""

import json
import time
import uuid
from typing import Callable
from fastapi import Request, Response
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import logging

from .request_logger import request_logger


class RequestResponseLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log all API requests and responses.
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.logger = logging.getLogger(__name__)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and response, logging both.
        
        Args:
            request: Incoming request
            call_next: Next middleware/endpoint in chain
            
        Returns:
            Response with logging completed
        """
        # Generate unique request ID
        request_id = f"req_{uuid.uuid4().hex[:12]}"
        start_time = time.time()
        
        # Read request body
        request_body = await self._get_request_body(request)
        
        # Log request
        try:
            request_logger.log_request(request, request_id, request_body)
        except Exception as e:
            self.logger.error(f"Failed to log request: {str(e)}")
        
        # Store request info for response logging
        request.state.request_id = request_id
        request.state.start_time = start_time
        
        # Process request
        response = None
        error_message = None
        
        try:
            response = await call_next(request)
        except Exception as e:
            error_message = str(e)
            self.logger.error(f"Request {request_id} failed: {error_message}")
            # Re-raise the exception to maintain normal error handling
            raise
        finally:
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Get response body
            response_body = await self._get_response_body(response)
            
            # Log response
            try:
                request_logger.log_response(
                    request_id=request_id,
                    response=response,
                    body=response_body,
                    processing_time=processing_time,
                    error_message=error_message
                )
            except Exception as e:
                self.logger.error(f"Failed to log response: {str(e)}")
        
        return response
    
    async def _get_request_body(self, request: Request) -> str:
        """
        Extract request body safely.
        
        Args:
            request: FastAPI Request object
            
        Returns:
            str: Request body content or empty string
        """
        try:
            # Skip body reading for GET requests and large files
            if request.method in ["GET", "HEAD", "OPTIONS"]:
                return ""
            
            # Check content type
            content_type = request.headers.get("content-type", "")
            
            # Skip binary content
            if "multipart/form-data" in content_type or "application/octet-stream" in content_type:
                return f"[BINARY CONTENT: {content_type}]"
            
            # Read body
            body = await request.body()
            
            # Limit body size for logging (max 10KB)
            if len(body) > 10240:
                return f"[LARGE BODY: {len(body)} bytes, Content-Type: {content_type}]"
            
            # Try to decode as text
            try:
                body_str = body.decode('utf-8')
                
                # Try to parse as JSON and pretty-print it
                if "application/json" in content_type:
                    try:
                        parsed_json = json.loads(body_str)
                        return json.dumps(parsed_json, indent=2, ensure_ascii=False)
                    except json.JSONDecodeError:
                        pass
                
                return body_str
                
            except UnicodeDecodeError:
                return f"[BINARY DATA: {len(body)} bytes]"
                
        except Exception as e:
            self.logger.error(f"Error reading request body: {str(e)}")
            return f"[ERROR READING BODY: {str(e)}]"
    
    async def _get_response_body(self, response: Response) -> str:
        """
        Extract response body safely.
        
        Args:
            response: FastAPI Response object
            
        Returns:
            str: Response body content or description
        """
        try:
            if not response:
                return "[NO RESPONSE]"
            
            # Handle different response types
            if isinstance(response, StreamingResponse):
                return "[STREAMING RESPONSE]"
            
            # Check if response has body
            if not hasattr(response, 'body'):
                return "[NO BODY ATTRIBUTE]"
            
            body = response.body
            
            if not body:
                return ""
            
            # Limit response body size for logging (max 5KB)
            if len(body) > 5120:
                return f"[LARGE RESPONSE: {len(body)} bytes]"
            
            # Try to decode as text
            try:
                body_str = body.decode('utf-8')
                
                # Try to parse as JSON and pretty-print it
                content_type = response.headers.get("content-type", "")
                if "application/json" in content_type:
                    try:
                        parsed_json = json.loads(body_str)
                        return json.dumps(parsed_json, indent=2, ensure_ascii=False)
                    except json.JSONDecodeError:
                        pass
                
                return body_str
                
            except UnicodeDecodeError:
                return f"[BINARY RESPONSE: {len(body)} bytes]"
                
        except Exception as e:
            self.logger.error(f"Error reading response body: {str(e)}")
            return f"[ERROR READING RESPONSE: {str(e)}]"


def setup_request_logging_middleware(app):
    """
    Setup request/response logging middleware.
    
    Args:
        app: FastAPI application instance
    """
    app.add_middleware(RequestResponseLoggingMiddleware)
    logging.info("Request/Response logging middleware configured")
