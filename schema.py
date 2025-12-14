"""
Database Schema and SQL Reference for Veda AI
==============================================
Contains schema definitions and example queries for the agent.
"""

# =============================================================================
# DATABASE SCHEMA
# =============================================================================

DATABASE_SCHEMA = """
Table: "General Machine 2000" (PostgreSQL - ALWAYS use double quotes!)

Columns:
- id (INTEGER) - Auto-incrementing primary key
- machine_id (TEXT) - Unique machine identifier (e.g., 'M001', 'M002')
- machine_name (TEXT) - Human-readable name (e.g., 'Machine Alpha', 'Machine Beta')
- order_number (TEXT) - Production order reference
- product_name (TEXT) - Name of product being manufactured
- qty (NUMERIC) - Quantity produced
- date (TEXT) - Production date (YYYY-MM-DD format)
- temperature_celsius (NUMERIC) - Machine temperature reading
- humidity_percent (NUMERIC) - Ambient humidity percentage
- speed_m_per_min (NUMERIC) - Machine speed in meters per minute
- vibration_mm_per_s (NUMERIC) - Vibration measurement in mm/s
- rpm (INTEGER) - Rotations per minute
- pressure_bar (NUMERIC) - Pressure in bar
- power_load_kw (NUMERIC) - Power consumption in kilowatts

IMPORTANT:
- Table has ~20 million rows - ALWAYS use aggregations
- NEVER use SELECT * for analysis queries
- Use COALESCE() to handle NULL values
- Use GROUP BY with aggregation functions
- Always LIMIT results appropriately
"""

# =============================================================================
# SQL EXAMPLES
# =============================================================================

SQL_EXAMPLES = """
-- Top producing machines:
SELECT machine_name, SUM(qty) as total_production 
FROM "General Machine 2000" 
GROUP BY machine_name 
ORDER BY total_production DESC 
LIMIT 10;

-- Daily production trend:
SELECT date, SUM(qty) as daily_total 
FROM "General Machine 2000" 
GROUP BY date 
ORDER BY date DESC
LIMIT 30;

-- Machine health metrics:
SELECT machine_name,
       ROUND(AVG(temperature_celsius)::numeric, 2) as avg_temp,
       ROUND(AVG(vibration_mm_per_s)::numeric, 2) as avg_vibration,
       ROUND(AVG(rpm)::numeric, 0) as avg_rpm,
       ROUND(AVG(power_load_kw)::numeric, 2) as avg_power
FROM "General Machine 2000"
GROUP BY machine_name
ORDER BY avg_vibration DESC
LIMIT 10;

-- Product summary:
SELECT product_name, 
       COUNT(*) as record_count,
       SUM(qty) as total_qty,
       ROUND(AVG(qty)::numeric, 2) as avg_qty
FROM "General Machine 2000"
GROUP BY product_name
ORDER BY total_qty DESC
LIMIT 10;

-- Machine efficiency (production per power):
SELECT machine_name,
       SUM(qty) as total_production,
       ROUND(AVG(power_load_kw)::numeric, 2) as avg_power,
       ROUND((SUM(qty) / NULLIF(SUM(power_load_kw), 0))::numeric, 2) as efficiency_ratio
FROM "General Machine 2000"
GROUP BY machine_name
HAVING SUM(power_load_kw) > 0
ORDER BY efficiency_ratio DESC
LIMIT 10;

-- Monthly production trend:
SELECT 
    SUBSTRING(date, 1, 7) as month,
    SUM(qty) as monthly_production,
    COUNT(DISTINCT machine_id) as active_machines
FROM "General Machine 2000"
GROUP BY SUBSTRING(date, 1, 7)
ORDER BY month DESC
LIMIT 12;

-- Machine criticality score (high vibration + high temp = critical):
SELECT machine_name,
       ROUND(AVG(vibration_mm_per_s)::numeric, 2) as avg_vibration,
       ROUND(AVG(temperature_celsius)::numeric, 2) as avg_temp,
       ROUND((
           COALESCE(AVG(vibration_mm_per_s), 0) * 0.5 + 
           COALESCE(AVG(temperature_celsius), 0) / 100 * 0.5
       )::numeric, 2) as criticality_score
FROM "General Machine 2000"
GROUP BY machine_name
ORDER BY criticality_score DESC
LIMIT 10;

-- Production by order:
SELECT order_number,
       product_name,
       SUM(qty) as total_qty,
       COUNT(*) as batch_count,
       MIN(date) as start_date,
       MAX(date) as end_date
FROM "General Machine 2000"
WHERE order_number IS NOT NULL
GROUP BY order_number, product_name
ORDER BY total_qty DESC
LIMIT 20;
"""

# =============================================================================
# CHART TYPE GUIDELINES
# =============================================================================

CHART_GUIDELINES = """
Chart Type Selection:
- LINE: Time-series data (date/time on x-axis, trends over time)
- BAR: Categorical comparisons (machines, products, orders)
- PIE: Distribution/percentage (max 6-8 slices, single metric)
- TABLE: Detailed multi-column data, many rows

When to suggest each:
- "Show me production over time" → LINE chart
- "Compare machines" → BAR chart
- "What's the distribution of..." → PIE chart
- "List all..." → TABLE or CSV export
"""

# =============================================================================
# ANALYSIS PATTERNS
# =============================================================================

ANALYSIS_PATTERNS = {
    "best_machine": {
        "description": "Find the best performing machine",
        "sql": """
            SELECT machine_name, SUM(qty) as total_production
            FROM "General Machine 2000"
            GROUP BY machine_name
            ORDER BY total_production DESC
            LIMIT 1
        """,
        "chart": "bar"
    },
    "production_trend": {
        "description": "Show production over time",
        "sql": """
            SELECT date, SUM(qty) as daily_production
            FROM "General Machine 2000"
            GROUP BY date
            ORDER BY date DESC
            LIMIT 30
        """,
        "chart": "line"
    },
    "machine_health": {
        "description": "Analyze machine health metrics",
        "sql": """
            SELECT machine_name,
                   AVG(temperature_celsius) as avg_temp,
                   AVG(vibration_mm_per_s) as avg_vibration,
                   AVG(power_load_kw) as avg_power
            FROM "General Machine 2000"
            GROUP BY machine_name
            ORDER BY avg_vibration DESC
            LIMIT 10
        """,
        "chart": "bar"
    }
}
