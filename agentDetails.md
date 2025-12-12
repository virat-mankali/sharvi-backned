This is a critical update. Handling **20 million rows** changes the architecture from "simple script" to "enterprise engineering." If you try to load 20 million rows into Python memory (Pandas), your Render instance will crash immediately.

Here is the updated, highly technical prompt for Kiro. It includes the logic for **Server-Side Aggregation** (for charts) and **Stream-to-Storage** (for exports).

-----

### Latest Update: Error Recovery & Intelligent Retry System

The agent now includes:

1. **Error Memory**: Tracks all failed SQL attempts with error analysis
2. **SQL Validation**: Validates queries using EXPLAIN before execution
3. **Intelligent Retry**: Up to 3 retries with learning from past failures
4. **Auto-Export Detection**: Automatically routes complex queries (ranking, all machines, criticality scoring) to CSV export
5. **Server-Side Calculations**: All scoring/ranking done in SQL, not Python

**Key Features:**
- `error_history`: List of `{sql, error, analysis}` for each failed attempt
- `validate_sql_syntax()`: Pre-execution validation using EXPLAIN
- `analyze_sql_error()`: Provides guidance for retry based on error type
- `requires_export`: Auto-detects queries needing CSV export (ranking, all machines, etc.)

**Flow for Complex Queries (e.g., "rank all machines by criticality"):**
1. Intent classified as "analysis"
2. `requires_export` flag set to `True` (detected keywords: rank, all machines, criticality)
3. SQL generated with server-side scoring calculation
4. SQL validated with EXPLAIN
5. If valid → exported to CSV → signed URL returned
6. If invalid → error recorded → retry with error context → repeat up to 3 times

-----

-----

### The Prompt for Kiro (Copy & Paste this)

**Role:**
You are a Lead Data Engineer and AI Architect.

**Goal:**
Build a production-grade **FastAPI backend** containing a **LangGraph Agent** responsible for analyzing and exporting large-scale data (20 million rows) from **Supabase**.

**The Stack:**

  * **Backend:** Python (FastAPI) hosted on Render.
  * **Orchestrator:** LangGraph + LangChain.
  * **Database:** Supabase (PostgreSQL).
  * **Frontend:** Next.js (You are providing the API for this).

**Critical Constraints (Scale & Performance):**

1.  **Volume:** The production DB has **20 million rows**.
2.  **No `SELECT *`:** The agent must **never** pull all raw rows into memory for analysis. It must use SQL aggregations (`SUM`, `COUNT`, `AVG`, `GROUP BY`) to reduce data *before* fetching it into Python.
3.  **Memory Safety:** Large exports must be streamed, not loaded into RAM.

-----

### Agent Architecture & Workflow

Create a `StateGraph` with a conditional entry point that classifies the user intent into one of two flows: **(A) Analysis/Visuals** or **(B) Data Export**.

#### Flow A: Analysis & Visualization (The "Chat" Flow)

  * **Goal:** Answer questions and provide data for frontend charts.
  * **Node Logic:**
    1.  **SQL Generator:** Converts user questions into SQL. *Strict Rule:* Must use aggregations. If the user asks for "Trend of sales," generate `SELECT date, SUM(amount)...`, not raw rows.
    2.  **Validator:** Checks if the query returns \> 100 rows. If so, refine the query to limit/group data.
    3.  **Formatter:** Instead of returning Markdown, return a **Structured JSON** object that the frontend can use to render Recharts/Chart.js.
    <!-- end list -->
      * *Output Schema:*
        ```json
        {
          "answer": "Sales peaked in Q3...",
          "visualization": {
            "type": "bar", // or line, pie
            "labels": ["Q1", "Q2", "Q3"],
            "datasets": [{ "label": "Sales", "data": [100, 150, 300] }]
          }
        }
        ```

#### Flow B: The "CSV Export" Flow (The "Download" Flow)

  * **Goal:** Allow users to download large datasets (e.g., "Give me all 50k transactions from NYC") without crashing the browser or server.
  * **Node Logic:**
    1.  **SQL Generator:** Generate the filtering query (e.g., `SELECT * FROM sales WHERE city = 'NYC'`).
    2.  **Stream-to-Storage Tool:**
          * **Do not** return this data to the chat.
          * Execute the query using a **Server-Side Cursor** (stream results in chunks).
          * Write chunks to a CSV buffer.
          * Upload the stream directly to a **Supabase Storage Bucket** (e.g., `exports`).
          * Generate a **Signed URL** (valid for 1 hour).
    3.  **Final Output:** Return a message: "I've prepared your dataset. You can download it here: [Link]."

-----

### Technical Implementation Requirements

**1. The Tools (`tools.py`)**

  * `query_aggregate_data`: For Flow A. Fetches small, aggregated dataframes.
  * `export_large_dataset`: For Flow B. Uses `cursor.fetchmany(size=1000)` loops to write to a file-like object and upload to Supabase Storage (`supabase.storage.from_('exports').upload(...)`).

**2. The Graph (`agent.py`)**

  * Use a Pydantic model for the output structure so the frontend always receives consistent JSON.

**3. The API (`server.py`)**

  * Endpoint: `POST /agent/analyze`
  * Input: `{"query": "Show me sales by region", "thread_id": "123"}`
  * Output: The structured JSON described in Flow A.

**4. Environment Variables**

  * Ensure the code uses `os.getenv` for: `SUPABASE_URL`, `SUPABASE_KEY`, `OPENAI_API_KEY`.

**Deliverables:**

  * Complete code for `agent.py` (Graph definition), `tools.py` (Supabase interaction), and `server.py` (FastAPI).
  * A specific System Prompt for the LLM instructing it on when to choose Bar vs Pie vs Line charts based on data types (e.g., "Use Line charts for time-series data").

-----

### Why this prompt works for you:

1.  **Prevents Crashing:** By explicitly forbidding `SELECT *` for analysis and enforcing "Stream-to-Storage" for exports, you solve the 20 million row problem.
2.  **Frontend Ready:** It forces the agent to return JSON for charts, meaning you can easily use libraries like **Recharts** or **Tremor** in Next.js to make it look professional, rather than just displaying text.
3.  **Scalable:** The "Cursor" logic ensures that even if you export 1 million rows, your Python memory usage remains low.