"""
Log viewer script for the California Housing Prediction API.

This script provides tools to view and analyze the logged requests and responses.
"""

import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import Optional
import argparse


class LogViewer:
    """
    Tool for viewing and analyzing API request/response logs.
    """
    
    def __init__(self, db_path: str = "logs/api_requests.db", log_file: str = "logs/api_requests.log"):
        self.db_path = db_path
        self.log_file = log_file
    
    def view_recent_requests(self, limit: int = 20, details: bool = False):
        """View recent requests."""
        print(f"📋 Recent {limit} API Requests")
        print("=" * 80)
        
        if not os.path.exists(self.db_path):
            print("❌ Database not found. Make sure the API has been running and processing requests.")
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if details:
                cursor.execute('''
                    SELECT r.timestamp, r.method, r.path, r.client_ip, r.user_agent,
                           resp.status_code, resp.processing_time_ms, resp.success
                    FROM api_requests r
                    LEFT JOIN api_responses resp ON r.id = resp.request_id
                    ORDER BY r.timestamp DESC
                    LIMIT ?
                ''', (limit,))
            else:
                cursor.execute('''
                    SELECT r.timestamp, r.method, r.path, resp.status_code, resp.processing_time_ms
                    FROM api_requests r
                    LEFT JOIN api_responses resp ON r.id = resp.request_id
                    ORDER BY r.timestamp DESC
                    LIMIT ?
                ''', (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                print("No requests found in database.")
                return
            
            # Print header
            if details:
                print(f"{'Timestamp':<20} {'Method':<6} {'Path':<25} {'IP':<15} {'Status':<6} {'Time(ms)':<8} {'Success':<7}")
                print("-" * 95)
            else:
                print(f"{'Timestamp':<20} {'Method':<6} {'Path':<30} {'Status':<6} {'Time(ms)':<8}")
                print("-" * 75)
            
            # Print requests
            for row in rows:
                timestamp = row[0][:19] if row[0] else "N/A"
                method = row[1] or "N/A"
                path = (row[2][:25] + "..." if len(row[2]) > 25 else row[2]) if row[2] else "N/A"
                
                if details:
                    client_ip = (row[3][:15] if row[3] else "N/A")
                    status = row[5] or "N/A"
                    time_ms = f"{row[6]:.1f}" if row[6] else "N/A"
                    success = "✅" if row[7] else "❌" if row[7] is not None else "❓"
                    
                    print(f"{timestamp:<20} {method:<6} {path:<25} {client_ip:<15} {status:<6} {time_ms:<8} {success:<7}")
                else:
                    path = (row[2][:30] + "..." if len(row[2]) > 30 else row[2]) if row[2] else "N/A"
                    status = row[3] or "N/A"
                    time_ms = f"{row[4]:.1f}" if row[4] else "N/A"
                    
                    print(f"{timestamp:<20} {method:<6} {path:<30} {status:<6} {time_ms:<8}")
        
        except Exception as e:
            print(f"❌ Error reading database: {e}")
    
    def view_statistics(self, hours: int = 24):
        """View request statistics."""
        print(f"📊 API Statistics (Last {hours} hours)")
        print("=" * 50)
        
        if not os.path.exists(self.db_path):
            print("❌ Database not found.")
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Calculate time threshold
            threshold = (datetime.now() - timedelta(hours=hours)).isoformat()
            
            # Total requests
            cursor.execute('''
                SELECT COUNT(*) FROM api_requests 
                WHERE timestamp > ?
            ''', (threshold,))
            total_requests = cursor.fetchone()[0]
            
            # Successful requests
            cursor.execute('''
                SELECT COUNT(*) FROM api_requests r
                JOIN api_responses resp ON r.id = resp.request_id
                WHERE r.timestamp > ? AND resp.success = 1
            ''', (threshold,))
            successful_requests = cursor.fetchone()[0]
            
            # Failed requests
            cursor.execute('''
                SELECT COUNT(*) FROM api_requests r
                JOIN api_responses resp ON r.id = resp.request_id
                WHERE r.timestamp > ? AND resp.success = 0
            ''', (threshold,))
            failed_requests = cursor.fetchone()[0]
            
            # Average response time
            cursor.execute('''
                SELECT AVG(resp.processing_time_ms) FROM api_requests r
                JOIN api_responses resp ON r.id = resp.request_id
                WHERE r.timestamp > ? AND resp.processing_time_ms IS NOT NULL
            ''', (threshold,))
            avg_response_time = cursor.fetchone()[0] or 0
            
            # Endpoint usage
            cursor.execute('''
                SELECT r.path, COUNT(*) as count FROM api_requests r
                WHERE r.timestamp > ?
                GROUP BY r.path
                ORDER BY count DESC
                LIMIT 10
            ''', (threshold,))
            endpoint_usage = cursor.fetchall()
            
            # Status code distribution
            cursor.execute('''
                SELECT resp.status_code, COUNT(*) as count FROM api_requests r
                JOIN api_responses resp ON r.id = resp.request_id
                WHERE r.timestamp > ?
                GROUP BY resp.status_code
                ORDER BY count DESC
            ''', (threshold,))
            status_codes = cursor.fetchall()
            
            conn.close()
            
            # Display statistics
            print(f"Total Requests: {total_requests}")
            print(f"Successful: {successful_requests}")
            print(f"Failed: {failed_requests}")
            
            if total_requests > 0:
                success_rate = (successful_requests / total_requests) * 100
                print(f"Success Rate: {success_rate:.1f}%")
            else:
                print("Success Rate: N/A")
            
            print(f"Average Response Time: {avg_response_time:.2f}ms")
            
            print("\n📈 Most Used Endpoints:")
            for path, count in endpoint_usage:
                print(f"  {path:<30} {count:>5} requests")
            
            print("\n📋 Status Code Distribution:")
            for status, count in status_codes:
                status_name = {
                    200: "OK",
                    201: "Created", 
                    400: "Bad Request",
                    401: "Unauthorized",
                    403: "Forbidden",
                    404: "Not Found",
                    422: "Validation Error",
                    500: "Internal Server Error"
                }.get(status, "Unknown")
                
                print(f"  {status} ({status_name}):{count:>5} requests")
        
        except Exception as e:
            print(f"❌ Error reading statistics: {e}")
    
    def view_request_details(self, request_id: str):
        """View detailed information for a specific request."""
        print(f"🔍 Request Details: {request_id}")
        print("=" * 50)
        
        if not os.path.exists(self.db_path):
            print("❌ Database not found.")
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get request details
            cursor.execute('''
                SELECT r.*, resp.status_code, resp.processing_time_ms, resp.success, resp.error_message
                FROM api_requests r
                LEFT JOIN api_responses resp ON r.id = resp.request_id
                WHERE r.id = ?
            ''', (request_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                print(f"❌ Request {request_id} not found.")
                return
            
            # Display details
            print(f"ID: {row[0]}")
            print(f"Timestamp: {row[1]}")
            print(f"Method: {row[2]}")
            print(f"URL: {row[3]}")
            print(f"Path: {row[4]}")
            print(f"Client IP: {row[8]}")
            print(f"User Agent: {row[9]}")
            print(f"Request Size: {row[10]} bytes")
            
            if row[12]:  # status_code exists
                print(f"Status Code: {row[12]}")
                print(f"Processing Time: {row[13]:.2f}ms")
                print(f"Success: {'✅ Yes' if row[14] else '❌ No'}")
                
                if row[15]:  # error_message
                    print(f"Error: {row[15]}")
            
            # Show headers and body if available
            if row[5]:  # headers
                try:
                    headers = json.loads(row[5])
                    print("\nHeaders:")
                    for key, value in headers.items():
                        print(f"  {key}: {value}")
                except:
                    pass
            
            if row[6]:  # query_params
                try:
                    params = json.loads(row[6])
                    if params:
                        print("\nQuery Parameters:")
                        for key, value in params.items():
                            print(f"  {key}: {value}")
                except:
                    pass
            
            if row[7]:  # body
                print(f"\nRequest Body:")
                try:
                    # Try to parse as JSON for pretty printing
                    body_json = json.loads(row[7])
                    print(json.dumps(body_json, indent=2))
                except:
                    # Show as plain text
                    print(row[7])
        
        except Exception as e:
            print(f"❌ Error reading request details: {e}")
    
    def view_log_file(self, lines: int = 50):
        """View recent entries from the log file."""
        print(f"📄 Recent {lines} Log Entries")
        print("=" * 80)
        
        if not os.path.exists(self.log_file):
            print(f"❌ Log file not found: {self.log_file}")
            return
        
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
                recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
                
                for line in recent_lines:
                    print(line.rstrip())
        
        except Exception as e:
            print(f"❌ Error reading log file: {e}")


def main():
    """Main function for the log viewer."""
    parser = argparse.ArgumentParser(description="View API request/response logs")
    parser.add_argument("action", choices=["requests", "stats", "details", "logs"], 
                       help="Action to perform")
    parser.add_argument("--limit", type=int, default=20, help="Number of requests to show")
    parser.add_argument("--hours", type=int, default=24, help="Time period in hours")
    parser.add_argument("--details", action="store_true", help="Show detailed information")
    parser.add_argument("--request-id", help="Specific request ID to view")
    parser.add_argument("--lines", type=int, default=50, help="Number of log lines to show")
    
    args = parser.parse_args()
    
    viewer = LogViewer()
    
    if args.action == "requests":
        viewer.view_recent_requests(args.limit, args.details)
    elif args.action == "stats":
        viewer.view_statistics(args.hours)
    elif args.action == "details":
        if not args.request_id:
            print("❌ --request-id is required for details action")
            return
        viewer.view_request_details(args.request_id)
    elif args.action == "logs":
        viewer.view_log_file(args.lines)


if __name__ == "__main__":
    if len(os.sys.argv) == 1:
        # Interactive mode
        print("🏠 California Housing API - Log Viewer")
        print("=" * 50)
        
        viewer = LogViewer()
        
        while True:
            print("\nChoose an option:")
            print("1. View recent requests")
            print("2. View statistics")
            print("3. View log file")
            print("4. Exit")
            
            choice = input("\nEnter choice (1-4): ").strip()
            
            if choice == "1":
                limit = input("Number of requests to show (default 20): ").strip()
                limit = int(limit) if limit.isdigit() else 20
                
                details = input("Show details? (y/n, default n): ").strip().lower() == 'y'
                
                print()
                viewer.view_recent_requests(limit, details)
            
            elif choice == "2":
                hours = input("Time period in hours (default 24): ").strip()
                hours = int(hours) if hours.isdigit() else 24
                
                print()
                viewer.view_statistics(hours)
            
            elif choice == "3":
                lines = input("Number of log lines to show (default 50): ").strip()
                lines = int(lines) if lines.isdigit() else 50
                
                print()
                viewer.view_log_file(lines)
            
            elif choice == "4":
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please try again.")
    else:
        # Command line mode
        main()
