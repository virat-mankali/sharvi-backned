The Prompt for Kiro (Copy & Paste this)
Role: You are a Senior Python Backend Engineer specializing in AI Agents, LangGraph, and Data Analytics.

Goal: Create a robust FastAPI backend that hosts a LangGraph Data Analyst Agent. This backend will be hosted on Render and must expose a REST API for my Next.js frontend to consume.

The Context: I have a database hosted on Supabase. The agent needs to answer user questions by converting natural language into SQL, querying the database, analyzing the results using Pandas, and returning a helpful insight.

Tech Stack Requirements:

Framework: FastAPI

Agent Orchestration: LangGraph & LangChain

LLM: ChatOpenAI (gpt-4o or gpt-4-turbo)

Database: supabase-py (or psycopg2 via SQLAlchemy for SQL generation)

Data Processing: Pandas

Detailed Agent Workflow (The "Graph"): Please implement a StateGraph with the following Nodes and logic:

State Definition: A typed dictionary containing messages (chat history) and data_context (to hold retrieved SQL results).

Node 1: Schema Loader: A tool or step that understands the database structure (tables and columns) so the LLM knows what to query.

Node 2: SQL Generator: The agent interprets the user's question and writes a valid SQL query for Supabase (PostgreSQL).

Node 3: SQL Executor: A tool that executes the SQL query against Supabase and returns the raw data.

Constraint: Handle empty results or SQL errors gracefully (allow the agent to retry if the SQL fails).

Node 4: Data Analyst: Takes the raw data (convert it to a Pandas DataFrame internally) and generates a natural language answer/insight based on the user's question.

API Endpoints Required:

POST /chat: Accepts a JSON body {"query": "User question", "thread_id": "optional-session-id"}. It should run the graph and return the final agent response.

GET /health: A simple health check returning {"status": "ok"}.

Environment Variables: Please set up the code to load these from a .env file:

OPENAI_API_KEY

SUPABASE_URL

SUPABASE_KEY (Service Role or Anon key depending on permissions needed)

Deliverables:

requirements.txt with all dependencies.

agent.py: The logic containing the LangGraph definition, tools, and nodes.

server.py: The FastAPI app setup.

Include comments explaining how to run it locally (uvicorn server:app --reload).

Here is the SQL schema of the tables in my Supabase database:

```sql
-- Table: "General Machine 2000"
-- Description: Machine production and sensor data
CREATE TABLE "General Machine 2000" (
    id INTEGER PRIMARY KEY,              -- Auto-incrementing ID
    machine_id TEXT,                     -- Unique machine identifier
    machine_name TEXT,                   -- Human-readable machine name
    order_number TEXT,                   -- Production order number
    product_name TEXT,                   -- Name of product being manufactured
    qty NUMERIC,                         -- Quantity produced
    date TEXT,                           -- Production date
    temperature_celsius NUMERIC,         -- Machine temperature in Celsius
    humidity_percent NUMERIC,            -- Ambient humidity percentage
    speed_m_per_min NUMERIC,             -- Machine speed in meters per minute
    vibration_mm_per_s NUMERIC,          -- Vibration measurement in mm/s
    rpm INTEGER,                         -- Rotations per minute
    pressure_bar NUMERIC,                -- Pressure in bar
    power_load_kw NUMERIC                -- Power consumption in kilowatts
);
```