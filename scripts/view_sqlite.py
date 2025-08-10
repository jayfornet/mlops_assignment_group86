"""
SQLite Database Viewer for Docker Containers

This script provides tools to view SQLite databases from within Docker containers
or by copying them from containers to the host.
"""

import sqlite3
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any


def view_tables(db_path: str):
    """View all tables in the SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print(f"📊 Tables in database: {db_path}")
        print("=" * 50)
        
        for table in tables:
            table_name = table[0]
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            
            print(f"📋 {table_name}: {count} rows")
            
            # Get table schema
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            print("   Columns:")
            for col in columns:
                col_name = col[1]
                col_type = col[2]
                is_pk = " (PRIMARY KEY)" if col[5] else ""
                print(f"     - {col_name}: {col_type}{is_pk}")
            print()
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error viewing tables: {e}")


def view_table_data(db_path: str, table_name: str, limit: int = 10):
    """View data from a specific table."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get column names
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Get data
        cursor.execute(f"SELECT * FROM {table_name} ORDER BY rowid DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        
        print(f"📋 Latest {limit} rows from {table_name}")
        print("=" * 80)
        
        if not rows:
            print("No data found.")
            return
        
        # Print header
        header = " | ".join(f"{col[:15]:<15}" for col in columns)
        print(header)
        print("-" * len(header))
        
        # Print rows
        for row in rows:
            row_str = " | ".join(str(cell)[:15] if cell is not None else "NULL" for cell in row)
            print(row_str)
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error viewing table data: {e}")


def view_requests_detailed(db_path: str, limit: int = 5):
    """View detailed request/response data."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if the request_response_log view exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='view' AND name='request_response_log'")
        if not cursor.fetchone():
            print("❌ request_response_log view not found. Using raw tables.")
            view_table_data(db_path, "api_requests", limit)
            return
        
        # Get recent requests with responses
        cursor.execute('''
            SELECT 
                request_id,
                request_timestamp,
                method,
                path,
                client_ip,
                status_code,
                processing_time_ms,
                success
            FROM request_response_log
            ORDER BY request_timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        
        print(f"🔍 Latest {limit} API Requests with Responses")
        print("=" * 100)
        
        if not rows:
            print("No requests found.")
            return
        
        for i, row in enumerate(rows, 1):
            request_id, timestamp, method, path, client_ip, status_code, processing_time, success = row
            
            success_icon = "✅" if success else "❌" if success is not None else "❓"
            
            print(f"\n{i}. Request ID: {request_id}")
            print(f"   Time: {timestamp}")
            print(f"   {method} {path}")
            print(f"   Client: {client_ip}")
            print(f"   Response: {status_code} {success_icon}")
            if processing_time:
                print(f"   Processing: {processing_time:.2f}ms")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error viewing detailed requests: {e}")


def export_to_json(db_path: str, table_name: str, output_file: str, limit: int = 100):
    """Export table data to JSON file."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get column names
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Get data
        cursor.execute(f"SELECT * FROM {table_name} ORDER BY rowid DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        
        # Convert to list of dictionaries
        data = []
        for row in rows:
            row_dict = dict(zip(columns, row))
            data.append(row_dict)
        
        # Write to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)
        
        print(f"✅ Exported {len(data)} rows from {table_name} to {output_file}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error exporting to JSON: {e}")


def search_requests(db_path: str, search_term: str, limit: int = 20):
    """Search for requests containing a specific term."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Search in various fields
        cursor.execute('''
            SELECT 
                r.id,
                r.timestamp,
                r.method,
                r.path,
                r.client_ip,
                resp.status_code
            FROM api_requests r
            LEFT JOIN api_responses resp ON r.id = resp.request_id
            WHERE r.path LIKE ? OR r.method LIKE ? OR r.client_ip LIKE ?
            ORDER BY r.timestamp DESC
            LIMIT ?
        ''', (f'%{search_term}%', f'%{search_term}%', f'%{search_term}%', limit))
        
        rows = cursor.fetchall()
        
        print(f"🔍 Search results for '{search_term}' (limit: {limit})")
        print("=" * 80)
        
        if not rows:
            print("No matching requests found.")
            return
        
        for row in rows:
            request_id, timestamp, method, path, client_ip, status_code = row
            print(f"{timestamp} | {method} | {path} | {client_ip} | {status_code or 'N/A'}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error searching requests: {e}")


def main():
    """Main function with interactive menu."""
    
    # Default database paths
    db_paths = [
        "logs/api_requests.db",
        "/app/logs/api_requests.db",  # Docker path
        "logs/predictions.db",
        "/app/logs/predictions.db"   # Docker path
    ]
    
    # Find existing database
    db_path = None
    for path in db_paths:
        if os.path.exists(path):
            db_path = path
            break
    
    if not db_path:
        print("❌ No SQLite database found in expected locations:")
        for path in db_paths:
            print(f"   - {path}")
        print("\nPlease specify the database path as an argument:")
        print("   python view_sqlite.py <database_path>")
        return
    
    print(f"📊 SQLite Database Viewer")
    print(f"Database: {db_path}")
    print("=" * 50)
    
    while True:
        print("\n🔍 Choose an option:")
        print("1. View all tables")
        print("2. View table data")
        print("3. View detailed requests")
        print("4. Search requests")
        print("5. Export table to JSON")
        print("6. Exit")
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        try:
            if choice == "1":
                view_tables(db_path)
            
            elif choice == "2":
                table_name = input("Enter table name (api_requests/api_responses): ").strip()
                limit = input("Number of rows to show (default 10): ").strip()
                limit = int(limit) if limit.isdigit() else 10
                view_table_data(db_path, table_name, limit)
            
            elif choice == "3":
                limit = input("Number of requests to show (default 5): ").strip()
                limit = int(limit) if limit.isdigit() else 5
                view_requests_detailed(db_path, limit)
            
            elif choice == "4":
                search_term = input("Enter search term: ").strip()
                limit = input("Number of results (default 20): ").strip()
                limit = int(limit) if limit.isdigit() else 20
                search_requests(db_path, search_term, limit)
            
            elif choice == "5":
                table_name = input("Enter table name: ").strip()
                output_file = input("Output file name (e.g., data.json): ").strip()
                limit = input("Number of rows (default 100): ").strip()
                limit = int(limit) if limit.isdigit() else 100
                export_to_json(db_path, table_name, output_file, limit)
            
            elif choice == "6":
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please try again.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Command line mode with specified database path
        db_path = sys.argv[1]
        if os.path.exists(db_path):
            view_tables(db_path)
            print()
            view_requests_detailed(db_path, 10)
        else:
            print(f"❌ Database not found: {db_path}")
    else:
        # Interactive mode
        main()
