"""
Database Tools for Sharvi AI
==========================
SQL execution and CSV export tools for Supabase/PostgreSQL.
"""
import os
import io
import csv
import uuid
from datetime import datetime
from typing import List, Dict, Any

from supabase import create_client, Client

# Database connection string
DATABASE_URL = os.getenv("DATABASE_URL")


def get_supabase_client() -> Client:
    """Create Supabase client."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
    return create_client(url, key)


def get_db_connection_string() -> str:
    """Build PostgreSQL connection string."""
    if DATABASE_URL:
        return DATABASE_URL
    
    supabase_url = os.getenv("SUPABASE_URL", "")
    db_password = os.getenv("SUPABASE_DB_PASSWORD", "")
    
    # Extract project ref from URL
    project_ref = supabase_url.replace("https://", "").split(".")[0]
    
    # Supabase pooler connection
    return f"postgresql://postgres.{project_ref}:{db_password}@aws-0-us-east-1.pooler.supabase.com:6543/postgres"


def _execute_raw_sql(sql_query: str) -> List[Dict[str, Any]]:
    """Execute raw SQL using psycopg3."""
    import psycopg
    from psycopg.rows import dict_row
    
    conn_string = get_db_connection_string()
    
    with psycopg.connect(conn_string, row_factory=dict_row, prepare_threshold=None) as conn:
        with conn.cursor() as cur:
            cur.execute(sql_query)
            rows = cur.fetchall()
    
    return [dict(row) for row in rows]


def query_aggregate_data(sql_query: str, max_rows: int = 100) -> Dict[str, Any]:
    """
    Execute an aggregation query for analysis.
    
    Args:
        sql_query: PostgreSQL query
        max_rows: Maximum rows to return (default 100)
    
    Returns:
        {"data": [...], "row_count": N} or {"error": "..."}
    """
    try:
        data = _execute_raw_sql(sql_query)
        
        # Truncate to max_rows
        if isinstance(data, list) and len(data) > max_rows:
            data = data[:max_rows]
        
        return {
            "data": data,
            "row_count": len(data) if data else 0
        }
    
    except Exception as e:
        error_msg = str(e)
        # Clean up error message
        if "ERROR:" in error_msg:
            error_msg = error_msg.split("ERROR:")[-1].strip()
        return {"error": error_msg}


def export_large_dataset(sql_query: str, bucket_name: str = "exports", custom_filename: str = "") -> Dict[str, Any]:
    """
    Export query results to CSV and upload to Supabase Storage.
    
    Args:
        sql_query: PostgreSQL query (LIMIT will be removed)
        bucket_name: Supabase storage bucket name
        custom_filename: Optional descriptive filename (e.g., "Overheating Machines by Rank")
    
    Returns:
        {"download_url": "...", "row_count": N, "filename": "..."} or {"error": "..."}
    """
    import psycopg
    from psycopg.rows import dict_row
    
    conn_string = get_db_connection_string()
    
    try:
        # Generate filename - use custom name if provided, otherwise generate
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if custom_filename:
            # Sanitize custom filename
            safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in custom_filename)
            safe_name = safe_name.replace(' ', '_')[:50]  # Limit length
            filename = f"{safe_name}_{timestamp}.csv"
        else:
            filename = f"export_{timestamp}_{uuid.uuid4().hex[:8]}.csv"
        
        # Create CSV in memory
        csv_buffer = io.StringIO()
        writer = None
        row_count = 0
        
        # Use server-side cursor for streaming large results
        with psycopg.connect(conn_string, row_factory=dict_row) as conn:
            with conn.cursor(name="export_cursor") as cur:
                cur.execute(sql_query)
                
                for row in cur:
                    row_dict = dict(row)
                    
                    # Write header on first row
                    if writer is None:
                        writer = csv.DictWriter(csv_buffer, fieldnames=row_dict.keys())
                        writer.writeheader()
                    
                    writer.writerow(row_dict)
                    row_count += 1
        
        if row_count == 0:
            return {"error": "Query returned no results", "row_count": 0}
        
        # Upload to Supabase Storage
        csv_content = csv_buffer.getvalue().encode("utf-8")
        csv_buffer.close()
        
        client = get_supabase_client()
        
        # Upload file
        client.storage.from_(bucket_name).upload(
            path=filename,
            file=csv_content,
            file_options={"content-type": "text/csv"}
        )
        
        # Create signed URL (1 hour expiry)
        signed_url = client.storage.from_(bucket_name).create_signed_url(
            path=filename,
            expires_in=3600
        )
        
        return {
            "download_url": signed_url.get("signedURL") or signed_url.get("signedUrl"),
            "filename": filename,
            "row_count": row_count
        }
    
    except Exception as e:
        return {"error": f"Export failed: {str(e)}"}


def validate_sql_syntax(sql_query: str) -> Dict[str, Any]:
    """
    Validate SQL syntax using EXPLAIN without executing.
    
    Returns:
        {"valid": True} or {"valid": False, "error": "..."}
    """
    import psycopg
    from psycopg.rows import dict_row
    
    conn_string = get_db_connection_string()
    
    try:
        explain_query = f"EXPLAIN {sql_query}"
        
        with psycopg.connect(conn_string, row_factory=dict_row, prepare_threshold=None) as conn:
            with conn.cursor() as cur:
                cur.execute(explain_query)
                return {"valid": True, "error": None}
    
    except psycopg.errors.UndefinedColumn as e:
        return {"valid": False, "error": f"Column error: {str(e)}"}
    except psycopg.errors.UndefinedTable as e:
        return {"valid": False, "error": f"Table error: {str(e)}"}
    except psycopg.errors.SyntaxError as e:
        return {"valid": False, "error": f"Syntax error: {str(e)}"}
    except Exception as e:
        error_msg = str(e)
        if "ERROR:" in error_msg:
            error_msg = error_msg.split("ERROR:")[-1].strip()
        return {"valid": False, "error": error_msg}


def get_table_stats() -> Dict[str, Any]:
    """Get basic statistics about the main table."""
    try:
        sql = '''
        SELECT 
            COUNT(*) as total_rows,
            COUNT(DISTINCT machine_id) as unique_machines,
            COUNT(DISTINCT product_name) as unique_products,
            MIN(date) as earliest_date,
            MAX(date) as latest_date
        FROM "General Machine 2000"
        '''
        result = _execute_raw_sql(sql)
        return {"stats": result[0] if result else {}}
    except Exception as e:
        return {"error": str(e)}
