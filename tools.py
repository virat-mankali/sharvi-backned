"""Tools for Supabase interaction - query execution and CSV export."""
import os
import io
import csv
import uuid
from datetime import datetime
from typing import List, Dict, Any
from supabase import create_client, Client

# Database connection string (set this in .env for direct PostgreSQL access)
DATABASE_URL = os.getenv("DATABASE_URL")


def get_supabase_client() -> Client:
    """Create and return Supabase client."""
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
    return create_client(url, key)


def get_db_connection_string() -> str:
    """Build PostgreSQL connection string from Supabase credentials."""
    if DATABASE_URL:
        return DATABASE_URL

    supabase_url = os.getenv("SUPABASE_URL", "")
    db_password = os.getenv("SUPABASE_DB_PASSWORD", "")

    # Extract project ref from URL (e.g., https://xxx.supabase.co -> xxx)
    project_ref = supabase_url.replace("https://", "").split(".")[0]

    # Supabase pooler connection (Transaction mode)
    return f"postgresql://postgres.{project_ref}:{db_password}@aws-0-us-east-1.pooler.supabase.com:6543/postgres"


def query_aggregate_data(sql_query: str) -> Dict[str, Any]:
    """
    Execute an aggregation query for analysis/visualization.
    Returns small, aggregated results safe for memory.
    """
    try:
        data = _execute_raw_sql(sql_query)

        # Safety check: limit rows
        if isinstance(data, list) and len(data) > 100:
            return {
                "error": "Query returned too many rows. Please add more aggregation or filters.",
                "row_count": len(data),
            }

        return {"data": data, "row_count": len(data) if data else 0}

    except Exception as e:
        return {"error": str(e)}


def _execute_raw_sql(sql_query: str) -> List[Dict[str, Any]]:
    """Execute raw SQL using psycopg3 connection."""
    import psycopg
    from psycopg.rows import dict_row

    conn_string = get_db_connection_string()

    try:
        # prepare_threshold=None disables prepared statements for Supavisor transaction mode
        with psycopg.connect(conn_string, row_factory=dict_row, prepare_threshold=None) as conn:
            with conn.cursor() as cur:
                cur.execute(sql_query)
                rows = cur.fetchall()
        return [dict(row) for row in rows]
    except Exception as e:
        raise Exception(f"SQL execution failed: {str(e)}")


def export_large_dataset(sql_query: str, bucket_name: str = "exports") -> Dict[str, Any]:
    """
    Stream large query results to CSV and upload to Supabase Storage.
    Uses server-side cursor to avoid memory issues.
    """
    import psycopg
    from psycopg.rows import dict_row

    conn_string = get_db_connection_string()

    try:
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"export_{timestamp}_{uuid.uuid4().hex[:8]}.csv"

        # Create CSV in memory buffer
        csv_buffer = io.StringIO()
        writer = None
        row_count = 0

        # Use server-side cursor for streaming
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
            file_options={"content-type": "text/csv"},
        )

        # Create signed URL (valid for 1 hour)
        signed_url = client.storage.from_(bucket_name).create_signed_url(
            path=filename, expires_in=3600
        )

        return {
            "download_url": signed_url.get("signedURL") or signed_url.get("signedUrl"),
            "filename": filename,
            "row_count": row_count,
        }

    except Exception as e:
        return {"error": f"Export failed: {str(e)}"}


def validate_query_size(sql_query: str) -> Dict[str, Any]:
    """Check estimated row count for a query before execution."""
    try:
        explain_query = f"EXPLAIN (FORMAT JSON) {sql_query}"
        result = _execute_raw_sql(explain_query)

        if result and len(result) > 0:
            plan = result[0].get("QUERY PLAN", [{}])[0]
            estimated_rows = plan.get("Plan", {}).get("Plan Rows", 0)
            return {"estimated_rows": estimated_rows}

        return {"estimated_rows": -1}
    except Exception as e:
        return {"error": str(e), "estimated_rows": -1}


def validate_sql_syntax(sql_query: str) -> Dict[str, Any]:
    """
    Validate SQL syntax using EXPLAIN without executing the query.
    This catches syntax errors, missing columns, wrong table names, etc.
    """
    import psycopg
    from psycopg.rows import dict_row

    conn_string = get_db_connection_string()

    try:
        # Use EXPLAIN to validate without executing
        explain_query = f"EXPLAIN {sql_query}"
        
        with psycopg.connect(conn_string, row_factory=dict_row, prepare_threshold=None) as conn:
            with conn.cursor() as cur:
                cur.execute(explain_query)
                # If we get here, the SQL is valid
                return {"valid": True, "error": None}
                
    except psycopg.errors.UndefinedColumn as e:
        return {"valid": False, "error": f"Column error: {str(e)}"}
    except psycopg.errors.UndefinedTable as e:
        return {"valid": False, "error": f"Table error: {str(e)}"}
    except psycopg.errors.SyntaxError as e:
        return {"valid": False, "error": f"Syntax error: {str(e)}"}
    except Exception as e:
        error_msg = str(e)
        # Extract the useful part of the error message
        if "ERROR:" in error_msg:
            error_msg = error_msg.split("ERROR:")[-1].strip()
        return {"valid": False, "error": error_msg}
