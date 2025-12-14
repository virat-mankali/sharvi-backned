# Sharvi AI - Cursor-Style Data Analyst Agent

## What This Agent Does

Sharvi AI is a streaming data analyst agent that works like Cursor AI IDE, but instead of helping with code, it helps analyze machine production data. It:

1. **Streams reasoning in real-time** - Shows its thought process as it works
2. **Executes SQL queries** - Converts natural language to PostgreSQL
3. **Provides visualizations** - Auto-generates charts when relevant
4. **Exports large datasets** - Creates downloadable CSV files

## Tech Stack

- **Backend**: FastAPI (Python)
- **Agent Framework**: LangGraph ReAct Agent
- **LLM**: GPT-4o (streaming enabled)
- **Database**: Supabase (PostgreSQL, ~20M rows)
- **Frontend**: Next.js with SSE streaming

## How Streaming Works

The agent uses LangGraph's multi-mode streaming:

```python
async for stream_type, chunk in agent.astream(
    {"messages": [HumanMessage(content=query)]},
    config=config,
    stream_mode=["messages", "updates", "custom"]
):
    # messages: Token-by-token LLM output
    # updates: Node completion events  
    # custom: Tool progress updates (via get_stream_writer)
```

### Stream Modes Explained

1. **messages** - Streams individual tokens as the LLM generates text
2. **updates** - Streams when nodes complete (agent thinking, tool execution)
3. **custom** - Streams custom data from inside tools using `get_stream_writer()`

## ReAct Loop

The agent follows Reason-Act-Observe:

```
User: "What's the best performing machine?"
     │
     ▼
┌─────────────────────────────────────────┐
│ REASON: "I need to find the machine     │
│ with highest total production"          │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│ ACT: run_sql_query(                     │
│   "SELECT machine_name, SUM(qty)..."    │
│ )                                       │
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│ OBSERVE: Got results:                   │
│ [{"machine_name": "Alpha", "total": 7201}]│
└─────────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│ RESPOND: "Machine Alpha is your top     │
│ performer with 7,201 total units"       │
└─────────────────────────────────────────┘
```

## Tools

### run_sql_query
Executes SQL and returns up to **25 rows maximum**. Streams progress via `get_stream_writer()`.

**IMPORTANT RULES:**
1. **Row Limit**: Always use `LIMIT 25` or less in queries. Never return more than 25 rows.
2. **Decimal Formatting**: Round all decimal numbers to maximum 3 decimal places. Use `ROUND(column, 3)` in SQL.
3. **Zero Values**: If a value is 0 or very close to 0, display it as `0` not `0.000000000000`.

```python
@tool
def run_sql_query(sql: str) -> str:
    writer = get_stream_writer()
    writer({"type": "tool_start", "sql": sql})
    # ... execute query ...
    writer({"type": "tool_result", "row_count": len(data)})
    return json.dumps({"data": data})
```

**Example SQL with proper formatting:**
```sql
SELECT 
    machine_name,
    ROUND(AVG(rpm), 3) AS avg_rpm,
    ROUND(AVG(power_load_kw), 3) AS avg_power,
    ROUND(AVG(vibration_mm_per_s), 3) AS avg_vibration
FROM "General Machine 2000"
GROUP BY machine_name
ORDER BY avg_rpm DESC
LIMIT 25;
```

### export_to_csv
Exports large datasets to Supabase Storage with signed download URL. Use this when user needs more than 25 rows.

**File Naming**: The agent provides a descriptive filename based on the request (e.g., "Overheating_Machines_by_Rank", "Top_Producers_by_Output").

## Frontend Integration

The frontend should:

1. **Connect via fetch with streaming**:
```typescript
const response = await fetch('/chat/stream', {
  method: 'POST',
  body: JSON.stringify({ query, thread_id })
});

const reader = response.body.getReader();
// Read SSE events...
```

2. **Handle event types**:
- `token` → Append to current message
- `thinking` → Show reasoning indicator
- `tool_start` → Show SQL being executed
- `visualization` → Render chart
- `download` → Show download button
- `done` → Mark complete

3. **Render progressively**:
- Show tokens as they arrive (like Cursor)
- Display SQL queries in code blocks
- Auto-render charts when data arrives

## Database Schema

```sql
Table: "General Machine 2000"

Columns:
- machine_id (TEXT)
- machine_name (TEXT)
- qty (NUMERIC)
- date (TEXT)
- temperature_celsius (NUMERIC)
- humidity_percent (NUMERIC)
- speed_m_per_min (NUMERIC)
- vibration_mm_per_s (NUMERIC)
- rpm (INTEGER)
- pressure_bar (NUMERIC)
- power_load_kw (NUMERIC)
- order_number (TEXT)
- product_name (TEXT)
```

## Environment Variables

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=eyJ...
DATABASE_URL=postgresql://...
```

## Running

```bash
cd backend
pip install -r requirements.txt
uvicorn server:app --reload --port 8000
```

## Example Queries

- "What's the best performing machine?"
- "Show me production trends over the last month"
- "Compare machine efficiency by power consumption"
- "Export all data for Machine Alpha to CSV"
- "Which machines have the highest vibration levels?"
- "What's the average temperature across all machines?"
