"""
Sharvi AI - Data Analyst Agent (Cursor-style streaming)
=======================================================
A ReAct agent that streams its reasoning, SQL queries, and analysis in real-time.
Designed to feel like Cursor AI but for data analysis instead of coding.
"""
import os
import json
import re
from typing import Dict, Any, List, Optional, AsyncGenerator
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer

from tools import query_aggregate_data, export_large_dataset, validate_sql_syntax


# =============================================================================
# SYSTEM PROMPT - Smart, concise, only use tools when needed
# =============================================================================
SYSTEM_PROMPT = """You are Sharvi AI, a friendly data analyst assistant for machine production data.

## CRITICAL: RESPONSE LENGTH LIMIT
- **Maximum response: 2000 tokens** - Be concise!
- Answer directly and to the point
- Only elaborate if user explicitly asks for more detail
- No unnecessary explanations or filler text

## CRITICAL: WHEN TO USE TOOLS
- ONLY use tools when the user asks for actual data analysis
- For greetings (hi, hello, hey, etc.) - just respond naturally, NO tools needed
- For general questions about what you can do - just explain briefly, NO tools needed
- For data questions (best machine, trends, comparisons, exports) - USE tools

## RESPONSE FORMATTING (Use Markdown)
Format your responses to be easy to read:
- Use **bold** for key values and machine names
- Use bullet points for lists
- Use tables for comparing multiple items (keep tables small, max 10 rows)
- Keep paragraphs short (2-3 sentences max)
- Add emoji sparingly for visual appeal (📊 for data, 🏆 for top performers, etc.)

## RESPONSE STYLE
- Be CONCISE - get to the point quickly
- For greetings: Short friendly response (1-2 sentences max)
- For data analysis: Lead with the key insight, then supporting details
- Never show SQL to users unless they ask
- Never list example queries unless asked
- Don't over-explain - trust the user understands

## WHEN ANALYZING DATA
1. Run the query (don't announce it beforehand)
2. Present findings concisely:
   - Lead with the key insight/answer in 1-2 sentences
   - Use a small table or bullet list for details (max 10 items)
   - Format numbers nicely (7,201.19 not 7201.19)
3. Brief context only if needed

## EXAMPLE GOOD RESPONSE FORMAT
"🏆 **Machine Alpha** is your top performer with **7,201 units** produced.

| Rank | Machine | Production |
|------|---------|------------|
| 1 | Machine Alpha | 7,201 |
| 2 | Machine Beta | 5,432 |

📊 Chart included above."

## DATABASE
Table: "General Machine 2000" (~20M rows)
Columns: machine_id, machine_name, qty, date, temperature_celsius, humidity_percent, speed_m_per_min, vibration_mm_per_s, rpm, pressure_bar, power_load_kw, order_number, product_name

## SQL RULES
- Quote table: FROM "General Machine 2000"
- Use COALESCE for NULLs
- Always use aggregations (never SELECT *)
- **LIMIT results to 25 rows maximum** (use LIMIT 10-25)
- **ROUND all decimals to 3 places max**: ROUND(column, 3)
- If value is 0 or near 0, display as 0 (not 0.000000000000)

## DECIMAL FORMATTING EXAMPLE
```sql
SELECT 
    machine_name,
    ROUND(AVG(rpm), 3) AS avg_rpm,
    ROUND(AVG(power_load_kw), 3) AS avg_power
FROM "General Machine 2000"
GROUP BY machine_name
LIMIT 25;
```

## CSV EXPORT FILE NAMING
When using export_to_csv, provide a descriptive filename based on the request:
- "Overheating Machines by Rank" for temperature analysis
- "Top Producers by Output" for production data
- "High Vibration Machines" for vibration analysis
- Use clear, descriptive names that reflect the data content

## DATA ACCURACY
- ONLY report values from tool responses - NEVER make up data
- If null, say "not recorded"
- Format numbers with commas
"""


# =============================================================================
# TOOLS - With streaming progress updates
# =============================================================================

@tool
def run_sql_query(sql: str) -> str:
    """
    Execute a SQL query to analyze machine production data.
    
    Args:
        sql: PostgreSQL query. Table name MUST be quoted: "General Machine 2000"
             IMPORTANT: Always use LIMIT 25 or less. Use ROUND(column, 3) for decimals.
    
    Returns:
        JSON with query results (max 25 rows)
    """
    writer = get_stream_writer()
    
    # Stream: Show the query being executed
    writer({
        "type": "tool_start",
        "tool": "run_sql_query",
        "message": "Executing SQL query...",
        "sql": sql
    })
    
    # Validate SQL first
    validation = validate_sql_syntax(sql)
    if not validation.get("valid", True) and validation.get("error"):
        writer({
            "type": "tool_error",
            "tool": "run_sql_query", 
            "message": f"SQL validation failed: {validation['error']}"
        })
        return json.dumps({
            "error": validation["error"],
            "hint": "Check table name is quoted: \"General Machine 2000\""
        })
    
    # Execute query
    result = query_aggregate_data(sql)
    
    if result.get("error"):
        writer({
            "type": "tool_error",
            "tool": "run_sql_query",
            "message": f"Query failed: {result['error']}"
        })
        return json.dumps({"error": result["error"]})
    
    data = result.get("data", [])
    row_count = len(data)
    
    # Stream: Show results summary
    writer({
        "type": "tool_result",
        "tool": "run_sql_query",
        "message": f"Query returned {row_count} rows",
        "row_count": row_count,
        "preview": data[:3] if data else []  # Preview first 3 rows
    })
    
    return json.dumps({
        "success": True,
        "row_count": row_count,
        "data": data,
        "note": "Use these EXACT values in your response. Report null values as 'not recorded'."
    }, default=str)


@tool
def export_to_csv(sql: str, filename: str = "", description: str = "") -> str:
    """
    Export large dataset to CSV file for download.
    Use this when the user needs more than 25 rows or wants to download data.
    
    Args:
        sql: PostgreSQL query. Table name MUST be quoted: "General Machine 2000"
        filename: Descriptive name for the file (e.g., "Overheating Machines by Rank", "Top Producers by Output")
        description: Brief description of what's being exported
    
    Returns:
        JSON with download URL and filename
    """
    writer = get_stream_writer()
    
    display_name = filename or description or "data export"
    
    writer({
        "type": "tool_start",
        "tool": "export_to_csv",
        "message": f"Preparing CSV export: {display_name}",
        "sql": sql
    })
    
    # Remove LIMIT for full export
    clean_sql = re.sub(r'\s+LIMIT\s+\d+\s*;?\s*$', '', sql, flags=re.IGNORECASE)
    if not clean_sql.strip().endswith(';'):
        clean_sql = clean_sql.strip() + ';'
    
    result = export_large_dataset(clean_sql, custom_filename=filename)
    
    if result.get("error"):
        writer({
            "type": "tool_error",
            "tool": "export_to_csv",
            "message": f"Export failed: {result['error']}"
        })
        return json.dumps({"error": result["error"]})
    
    writer({
        "type": "tool_result",
        "tool": "export_to_csv",
        "message": f"Exported {result['row_count']:,} rows to CSV",
        "row_count": result["row_count"],
        "download_url": result["download_url"],
        "filename": result.get("filename", "export.csv")
    })
    
    return json.dumps({
        "success": True,
        "row_count": result["row_count"],
        "download_url": result["download_url"],
        "filename": result.get("filename", "export.csv"),
        "message": f"Successfully exported {result['row_count']:,} rows"
    })


# =============================================================================
# AGENT CREATION
# =============================================================================

def create_agent():
    """Create the ReAct agent with streaming support."""
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
    
    model = ChatOpenAI(
        model=model_name,
        temperature=0,
        streaming=True,  # Enable token streaming
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    
    tools = [run_sql_query, export_to_csv]
    memory = InMemorySaver()
    
    agent = create_react_agent(
        model=model,
        tools=tools,
        prompt=SYSTEM_PROMPT,
        checkpointer=memory,
    )
    
    print(f"[AGENT] Created with model: {model_name}, streaming enabled")
    return agent


# Global agent instance
agent = create_agent()


# =============================================================================
# SYNCHRONOUS EXECUTION (for non-streaming endpoint)
# =============================================================================

def run_agent(query: str, thread_id: str = "default") -> Dict[str, Any]:
    """Run the agent synchronously and return final result."""
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config=config
        )
        
        messages = result.get("messages", [])
        
        # Extract final response
        response_text = ""
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                response_text = msg.content
                break
        
        if not response_text:
            response_text = "I couldn't generate a response. Please try again."
        
        # Extract visualization and download URL from tool messages
        visualization = None
        download_url = None
        
        for msg in messages:
            if isinstance(msg, ToolMessage):
                try:
                    data = json.loads(msg.content)
                    if data.get("download_url"):
                        download_url = data["download_url"]
                    elif data.get("data") and data.get("success"):
                        visualization = build_visualization(data["data"])
                except (json.JSONDecodeError, TypeError):
                    pass
        
        return {
            "answer": response_text,
            "visualization": visualization,
            "download_url": download_url,
            "error": None
        }
        
    except Exception as e:
        return {
            "answer": f"Error: {str(e)}",
            "visualization": None,
            "download_url": None,
            "error": str(e)
        }


# =============================================================================
# STREAMING EXECUTION (Cursor-style real-time updates)
# =============================================================================

async def run_agent_stream(query: str, thread_id: str = "default") -> AsyncGenerator[Dict[str, Any], None]:
    """
    Stream agent execution with real-time updates.
    
    Yields events in this format:
    - {"type": "thinking", "content": "..."} - Agent's reasoning
    - {"type": "tool_call", "tool": "...", "sql": "..."} - Tool being called
    - {"type": "tool_result", "tool": "...", "data": {...}} - Tool result
    - {"type": "token", "content": "..."} - Streaming token from LLM
    - {"type": "message", "content": "..."} - Complete message chunk
    - {"type": "visualization", "data": {...}} - Chart data
    - {"type": "download", "url": "..."} - CSV download URL
    - {"type": "done", "answer": "..."} - Final complete response
    - {"type": "error", "message": "..."} - Error occurred
    """
    import asyncio
    
    config = {"configurable": {"thread_id": thread_id}}
    
    # Track state for building final response
    final_answer = ""
    visualization_data = None
    download_url = None
    tool_results = []
    
    try:
        # Stream with multiple modes for rich updates
        async for stream_type, chunk in agent.astream(
            {"messages": [HumanMessage(content=query)]},
            config=config,
            stream_mode=["messages", "updates", "custom"]
        ):
            
            if stream_type == "messages":
                # LLM token streaming
                message_chunk, metadata = chunk
                if hasattr(message_chunk, 'content') and message_chunk.content:
                    yield {
                        "type": "token",
                        "content": message_chunk.content,
                        "node": metadata.get("langgraph_node", "")
                    }
                    final_answer += message_chunk.content
                
                # Check for tool calls in the message
                if hasattr(message_chunk, 'tool_calls') and message_chunk.tool_calls:
                    for tc in message_chunk.tool_calls:
                        yield {
                            "type": "tool_call",
                            "tool": tc.get("name", "unknown"),
                            "args": tc.get("args", {})
                        }
            
            elif stream_type == "updates":
                # Node updates (agent reasoning, tool execution)
                for node_name, node_data in chunk.items():
                    if node_name == "agent":
                        # Agent is thinking/responding
                        messages = node_data.get("messages", [])
                        for msg in messages:
                            if isinstance(msg, AIMessage):
                                if msg.content and not msg.tool_calls:
                                    # Final response content
                                    yield {
                                        "type": "message",
                                        "content": msg.content,
                                        "node": "agent"
                                    }
                                    final_answer = msg.content
                                elif msg.tool_calls:
                                    # Agent decided to use a tool
                                    for tc in msg.tool_calls:
                                        yield {
                                            "type": "thinking",
                                            "content": f"I'll query the database to find this information..."
                                        }
                    
                    elif node_name == "tools":
                        # Tool execution completed
                        messages = node_data.get("messages", [])
                        for msg in messages:
                            if isinstance(msg, ToolMessage):
                                try:
                                    result = json.loads(msg.content)
                                    tool_results.append(result)
                                    
                                    if result.get("download_url"):
                                        download_url = result["download_url"]
                                        yield {
                                            "type": "download",
                                            "url": download_url,
                                            "row_count": result.get("row_count", 0)
                                        }
                                    
                                    if result.get("data") and result.get("success"):
                                        viz = build_visualization(result["data"])
                                        if viz:
                                            visualization_data = viz
                                            yield {
                                                "type": "visualization",
                                                "data": viz
                                            }
                                        
                                        # Stream data summary
                                        yield {
                                            "type": "data_received",
                                            "row_count": result.get("row_count", 0),
                                            "preview": result["data"][:5] if result["data"] else []
                                        }
                                    
                                    if result.get("error"):
                                        yield {
                                            "type": "tool_error",
                                            "message": result["error"]
                                        }
                                        
                                except (json.JSONDecodeError, TypeError):
                                    pass
            
            elif stream_type == "custom":
                # Custom tool progress updates
                yield chunk
            
            await asyncio.sleep(0)  # Allow other tasks to run
        
        # Final done event with complete response
        yield {
            "type": "done",
            "answer": final_answer,
            "visualization": visualization_data,
            "download_url": download_url
        }
        
    except Exception as e:
        yield {
            "type": "error",
            "message": str(e)
        }


# =============================================================================
# VISUALIZATION BUILDER
# =============================================================================

def build_visualization(data: List[Dict]) -> Optional[Dict[str, Any]]:
    """Build chart-ready data from query results."""
    if not data or len(data) == 0:
        return None
    
    first_row = data[0]
    keys = list(first_row.keys())
    
    # Identify label column and numeric columns
    label_col = None
    numeric_cols = []
    
    for key in keys:
        val = first_row.get(key)
        if label_col is None and (isinstance(val, str) or val is None):
            label_col = key
        elif val is not None:
            try:
                float(val)
                numeric_cols.append(key)
            except (ValueError, TypeError):
                if label_col is None:
                    label_col = key
    
    if not label_col and keys:
        label_col = keys[0]
    
    if not numeric_cols:
        return None
    
    # Build chart data (max 20 items for readability)
    max_items = 20
    labels = []
    for row in data[:max_items]:
        val = row.get(label_col)
        if val is None:
            labels.append("(not recorded)")
        else:
            labels.append(str(val)[:40])  # Truncate long labels
    
    datasets = []
    colors = [
        "rgba(59, 130, 246, 0.8)",   # Blue
        "rgba(16, 185, 129, 0.8)",   # Green  
        "rgba(245, 158, 11, 0.8)",   # Amber
        "rgba(239, 68, 68, 0.8)",    # Red
        "rgba(139, 92, 246, 0.8)",   # Purple
    ]
    
    for i, col in enumerate(numeric_cols[:5]):  # Max 5 datasets
        values = []
        for row in data[:max_items]:
            try:
                v = row.get(col)
                values.append(round(float(v), 2) if v is not None else 0)
            except (ValueError, TypeError):
                values.append(0)
        
        datasets.append({
            "label": col.replace('_', ' ').title(),
            "data": values,
            "backgroundColor": colors[i % len(colors)],
            "borderColor": colors[i % len(colors)].replace("0.8", "1"),
            "borderWidth": 1
        })
    
    # Determine chart type based on data characteristics
    chart_type = "bar"
    
    # Time series → line chart
    if label_col and any(x in label_col.lower() for x in ['date', 'time', 'month', 'year', 'day', 'week']):
        chart_type = "line"
        for ds in datasets:
            ds["fill"] = False
            ds["tension"] = 0.1
    
    # Few items with single metric → pie chart
    elif len(data) <= 8 and len(numeric_cols) == 1:
        chart_type = "pie"
        datasets[0]["backgroundColor"] = colors[:len(data)]
    
    return {
        "type": chart_type,
        "labels": labels,
        "datasets": datasets,
        "options": {
            "responsive": True,
            "plugins": {
                "legend": {"position": "top"},
                "title": {"display": False}
            }
        }
    }
