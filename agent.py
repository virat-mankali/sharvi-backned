"""LangGraph Agent for Data Analysis with memory, conditional routing, and error recovery."""
import os
import json
from typing import Literal, Dict, Any, List, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import RetryPolicy
from typing_extensions import TypedDict

from schema import SYSTEM_PROMPT, DATABASE_SCHEMA
from tools import query_aggregate_data, export_large_dataset, validate_sql_syntax


# Initialize LLM
def get_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


# ============== STATE DEFINITION WITH ERROR MEMORY ==============

class AgentState(TypedDict):
    """State for the LangGraph agent with conversation memory and error tracking."""
    messages: Annotated[List[BaseMessage], add_messages]  # Conversation history
    query: str
    intent: str | None
    sql_query: str | None
    query_result: List[Dict[str, Any]] | None
    final_response: Dict[str, Any] | None
    error: str | None
    retry_count: int
    # Error memory for learning from failures
    error_history: List[Dict[str, str]] | None  # [{sql, error, analysis}]
    requires_export: bool | None  # Flag for queries that need CSV export


# ============== HELPER FUNCTIONS ==============

def get_conversation_context(messages: List[BaseMessage], max_messages: int = 10) -> str:
    """Extract recent conversation context for the LLM."""
    recent = messages[-max_messages:] if len(messages) > max_messages else messages
    context_parts = []
    for msg in recent:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        context_parts.append(f"{role}: {msg.content}")
    return "\n".join(context_parts)


# ============== NODE FUNCTIONS ==============

def classify_intent(state: AgentState) -> Dict[str, Any]:
    """Classify user intent: greeting, analysis, export, or followup."""
    llm = get_llm()
    
    # Get conversation context
    context = get_conversation_context(state.get("messages", []))

    classification_prompt = f"""Classify the user's intent into one of four categories:

1. "greeting" - User is saying hello, greeting, asking about themselves, or asking general questions not about data
   Examples: "Hi", "Hello", "Hey there", "How are you?", "What can you do?", "Help", "What's my name?", "Do you remember me?"

2. "followup" - User is asking a follow-up question about a PREVIOUS response, asking for clarification, or asking HOW/WHY something was done
   Examples: "How did you calculate that?", "Why did you use those parameters?", "Can you explain the formula?", "What does that score mean?", "How did u calculate the criticality score?", "Explain the methodology", "What weights did you use?", "Why is machine X ranked higher?"
   IMPORTANT: If the user is asking about something that was just discussed or explained, this is a followup!

3. "analysis" - User wants NEW insights, charts, summaries, trends, comparisons, or answers about the data
   Examples: "Show me sales by region", "What's the average temperature?", "Trend of production", "Compare machine A vs B"

4. "export" - User wants to download/export raw data as CSV/file
   Examples: "Download all records from NYC", "Export transactions", "Give me the raw data"

Recent conversation:
{context}

Current user query: {state['query']}

IMPORTANT: If the user is asking about HOW something was calculated or WHY something was done in a previous response, classify as "followup", NOT "analysis".

Respond with ONLY one word: "greeting", "followup", "analysis", or "export"
"""

    response = llm.invoke([HumanMessage(content=classification_prompt)])
    intent = response.content.strip().lower()

    if intent not in ["greeting", "followup", "analysis", "export"]:
        intent = "analysis"

    return {"intent": intent}


def handle_greeting(state: AgentState) -> Dict[str, Any]:
    """Handle greeting/conversational messages with memory context."""
    llm = get_llm()
    
    # Get conversation context
    context = get_conversation_context(state.get("messages", []))

    greeting_prompt = f"""You are Veda AI, a friendly and helpful data analyst assistant for machine production data.

Previous conversation:
{context}

Current message from user: "{state['query']}"

Respond warmly and helpfully. Remember details from the conversation (like the user's name if they told you).
You can help them:
- Analyze machine performance (RPM, temperature, pressure, etc.)
- Compare machines and find the best/worst performers
- Show trends over time
- Generate visualizations (bar charts, line charts, pie charts)
- Export data to CSV for download

Keep your response concise, friendly, and personal. Reference previous conversation if relevant.
Don't use markdown formatting.
"""

    response = llm.invoke([HumanMessage(content=greeting_prompt)])
    answer = response.content.strip()

    return {
        "final_response": {
            "answer": answer,
            "visualization": None,
            "error": None,
        },
        "messages": [AIMessage(content=answer)],
    }


def handle_followup(state: AgentState) -> Dict[str, Any]:
    """Handle follow-up questions about previous responses - explain methodology, calculations, etc."""
    llm = get_llm()
    
    # Get full conversation context
    context = get_conversation_context(state.get("messages", []), max_messages=20)

    followup_prompt = f"""You are Veda AI, a data analyst assistant. The user is asking a follow-up question about something you previously discussed or calculated.

FULL CONVERSATION HISTORY:
{context}

CURRENT FOLLOW-UP QUESTION: "{state['query']}"

Your task is to EXPLAIN and CLARIFY based on what was discussed before. 

If the user asks about how a criticality score or ranking was calculated, explain:
1. The formula/methodology used
2. The weights assigned to each parameter
3. Why those parameters matter for machine criticality
4. How to interpret the results

For machine criticality scoring, the typical formula is:
- Criticality Score = (Vibration × 0.25) + (Power Load × 0.25) + (Temperature/100 × 0.25) + (RPM/1000 × 0.25)
- Higher scores indicate machines that need more attention
- Vibration: High vibration often indicates wear or misalignment
- Power Load: Higher load means the machine is working harder
- Temperature: Elevated temps can indicate friction or cooling issues
- RPM: Higher speeds can accelerate wear

If the user asks about MTBF/MTTR:
- MTBF (Mean Time Between Failures): Average time a machine runs before failing
- MTTR (Mean Time To Repair): Average time to fix a machine after failure
- Note: These metrics require failure/maintenance data which may not be in the current dataset

Provide a clear, helpful explanation. Be conversational and reference the specific context of what was discussed.
Don't generate new SQL or data - just explain what was already done.
"""

    response = llm.invoke([HumanMessage(content=followup_prompt)])
    answer = response.content.strip()

    return {
        "final_response": {
            "answer": answer,
            "visualization": None,
            "error": None,
        },
        "messages": [AIMessage(content=answer)],
    }


def generate_sql(state: AgentState) -> Dict[str, Any]:
    """Generate SQL query based on user question and intent, learning from past errors."""
    llm = get_llm()
    
    # Get conversation context for better understanding
    context = get_conversation_context(state.get("messages", []))
    
    # Build error context if we have previous failures
    error_context = ""
    error_history = state.get("error_history") or []
    if error_history:
        error_context = "\n\nPREVIOUS FAILED ATTEMPTS (learn from these):\n"
        for i, err in enumerate(error_history, 1):
            error_context += f"""
Attempt {i}:
- SQL: {err.get('sql', 'N/A')}
- Error: {err.get('error', 'N/A')}
- Analysis: {err.get('analysis', 'N/A')}
"""
        error_context += "\nDO NOT repeat these mistakes. Generate a corrected query.\n"

    # Check if this is a complex query that needs export (ranking, all rows, etc.)
    query_lower = state['query'].lower()
    needs_export = any(keyword in query_lower for keyword in [
        'rank', 'all machines', 'all rows', 'csv', 'download', 'export',
        'criticality', 'score', 'calculate', 'based on parameters'
    ])

    if state["intent"] == "analysis" and not needs_export:
        sql_instructions = """Generate a SQL query that:
- Uses aggregations (SUM, COUNT, AVG, GROUP BY)
- Returns at most 100 rows
- Is optimized for visualization
- Returns data suitable for charts

Return ONLY the SQL query, no explanation."""
    else:
        # For export or complex ranking queries
        sql_instructions = """Generate a SQL query that:
- Performs all calculations SERVER-SIDE in SQL (not in Python)
- For ranking/scoring: Use window functions like RANK(), ROW_NUMBER(), or calculated columns
- Include all relevant columns needed for the analysis
- Use COALESCE for NULL handling
- For criticality scoring, calculate a composite score using available metrics

Example for machine criticality ranking:
SELECT 
    machine_id,
    machine_name,
    AVG(rpm) as avg_rpm,
    AVG(power_load_kw) as avg_load,
    AVG(vibration_mm_per_s) as avg_vibration,
    COUNT(*) as total_records,
    -- Criticality score (higher = more critical)
    (COALESCE(AVG(vibration_mm_per_s), 0) * 0.3 + 
     COALESCE(AVG(power_load_kw), 0) * 0.2 + 
     COALESCE(AVG(temperature_celsius), 0) * 0.2 +
     COALESCE(AVG(rpm), 0) / 100 * 0.3) as criticality_score
FROM "General Machine 2000"
GROUP BY machine_id, machine_name
ORDER BY criticality_score DESC;

Return ONLY the SQL query, no explanation."""

    prompt = f"""{SYSTEM_PROMPT}

Previous conversation context:
{context}
{error_context}
Current User Question: {state['query']}

{sql_instructions}
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    sql_query = response.content.strip()

    # Clean up SQL (remove markdown code blocks if present)
    if sql_query.startswith("```"):
        sql_query = sql_query.split("```")[1]
        if sql_query.startswith("sql"):
            sql_query = sql_query[3:]
        sql_query = sql_query.strip()

    return {"sql_query": sql_query, "requires_export": needs_export}


def validate_sql(state: AgentState) -> Dict[str, Any]:
    """Validate SQL query syntax and structure before execution."""
    sql_query = state["sql_query"]
    
    # First, validate SQL syntax using EXPLAIN
    validation_result = validate_sql_syntax(sql_query)
    
    if validation_result.get("error"):
        # SQL has syntax errors - record for retry
        error_history = state.get("error_history") or []
        error_history.append({
            "sql": sql_query,
            "error": validation_result["error"],
            "analysis": "SQL syntax validation failed before execution"
        })
        return {
            "error": validation_result["error"],
            "error_history": error_history,
            "retry_count": state.get("retry_count", 0) + 1
        }
    
    # For analysis queries without export flag, ensure aggregation or limit
    if state["intent"] == "analysis" and not state.get("requires_export"):
        sql_upper = sql_query.upper()
        has_aggregation = any(
            agg in sql_upper
            for agg in ["SUM(", "COUNT(", "AVG(", "MIN(", "MAX(", "GROUP BY", "RANK(", "ROW_NUMBER("]
        )

        if not has_aggregation and "LIMIT" not in sql_upper:
            sql_query = sql_query.rstrip(";") + " LIMIT 100;"

    return {"sql_query": sql_query, "error": None}


def execute_analysis_query(state: AgentState) -> Dict[str, Any]:
    """Execute SQL for analysis and store results with intelligent error handling."""
    sql_query = state["sql_query"]
    
    # If this requires export (complex ranking, all machines, etc.), route to export
    if state.get("requires_export"):
        result = export_large_dataset(sql_query)
        
        if "error" in result and result.get("error"):
            # Record error for learning
            error_history = state.get("error_history") or []
            error_history.append({
                "sql": sql_query,
                "error": result["error"],
                "analysis": analyze_sql_error(result["error"], sql_query)
            })
            
            if state.get("retry_count", 0) < 3:
                return {
                    "error": result["error"],
                    "error_history": error_history,
                    "retry_count": state.get("retry_count", 0) + 1,
                }
            return {"error": result["error"], "query_result": None, "error_history": error_history}
        
        # Success - return download URL
        answer = f"I've calculated the machine rankings based on the parameters you specified. The dataset contains {result['row_count']:,} rows with criticality scores. You can download it using the link below (valid for 1 hour)."
        
        return {
            "final_response": {
                "answer": answer,
                "download_url": result["download_url"],
                "visualization": None,
            },
            "messages": [AIMessage(content=answer)],
            "error": None,
        }
    
    # Standard analysis query
    result = query_aggregate_data(sql_query)

    if "error" in result and result.get("error"):
        # Record error with analysis for learning
        error_history = state.get("error_history") or []
        error_history.append({
            "sql": sql_query,
            "error": result["error"],
            "analysis": analyze_sql_error(result["error"], sql_query)
        })
        
        if state.get("retry_count", 0) < 3:
            return {
                "error": result["error"],
                "error_history": error_history,
                "retry_count": state.get("retry_count", 0) + 1,
            }
        return {"error": result["error"], "query_result": None, "error_history": error_history}

    return {"query_result": result["data"], "error": None}


def analyze_sql_error(error: str, sql: str) -> str:
    """Analyze SQL error to provide guidance for retry."""
    error_lower = error.lower()
    
    if "column" in error_lower and "does not exist" in error_lower:
        return "Column name is incorrect. Check the schema for exact column names. Remember table is 'General Machine 2000' with columns like machine_id, machine_name, rpm, vibration_mm_per_s, power_load_kw, temperature_celsius, etc."
    
    if "relation" in error_lower and "does not exist" in error_lower:
        return "Table name is incorrect. Use exact name: \"General Machine 2000\" (with quotes and spaces)."
    
    if "syntax error" in error_lower:
        return "SQL syntax error. Check for missing commas, parentheses, or incorrect keywords."
    
    if "division by zero" in error_lower:
        return "Division by zero. Use NULLIF or COALESCE to handle zero values."
    
    if "too many rows" in error_lower:
        return "Query returns too many rows. Add GROUP BY aggregation or use export flow for large datasets."
    
    if "permission" in error_lower or "denied" in error_lower:
        return "Permission issue. Ensure query only accesses allowed tables."
    
    return f"Execution failed. Review the SQL structure and try a simpler approach."


def execute_export_query(state: AgentState) -> Dict[str, Any]:
    """Execute SQL for export and upload to storage."""
    result = export_large_dataset(state["sql_query"])

    if "error" in result:
        return {"error": result["error"]}

    answer = f"I've prepared your dataset with {result['row_count']:,} rows. You can download it using the link below (valid for 1 hour)."
    
    return {
        "final_response": {
            "answer": answer,
            "download_url": result["download_url"],
            "visualization": None,
        },
        "messages": [AIMessage(content=answer)],
    }


def format_analysis_response(state: AgentState) -> Dict[str, Any]:
    """Format query results into structured response with visualization data."""
    # If we already have a final response (from export), just return it
    if state.get("final_response"):
        return {}
    
    if state.get("error"):
        # Include error history in the response for debugging
        error_history = state.get("error_history") or []
        error_details = state['error']
        if error_history:
            error_details += f" (Attempted {len(error_history)} times)"
        
        answer = f"I encountered an error while processing your request: {error_details}. Please try rephrasing your question or simplifying the query."
        return {
            "final_response": {
                "answer": answer,
                "visualization": None,
                "error": state["error"],
            },
            "messages": [AIMessage(content=answer)],
        }

    llm = get_llm()
    data = state["query_result"]
    context = get_conversation_context(state.get("messages", []))

    format_prompt = f"""Based on this data and the user's question, create a response.

Previous conversation:
{context}

User Question: {state['query']}
SQL Query: {state['sql_query']}
Data (first 10 rows): {json.dumps(data[:10] if data else [], default=str)}
Total rows: {len(data) if data else 0}

Respond with a JSON object containing:
1. "answer": A natural language insight/summary (2-3 sentences). Be conversational and reference the user by name if known.
2. "chart_type": One of "bar", "line", "pie", or "table"
3. "labels": Array of x-axis labels (from the data)
4. "datasets": Array of objects with "label" (string) and "data" (array of numbers)

Rules for chart selection:
- LINE: Time-series data (dates on x-axis)
- BAR: Categorical comparisons
- PIE: Distribution/percentages (max 8 items)
- TABLE: When visualization doesn't make sense

Return ONLY valid JSON, no markdown or explanation.
"""

    response = llm.invoke([HumanMessage(content=format_prompt)])

    try:
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        parsed = json.loads(content)

        visualization = {
            "type": parsed.get("chart_type", "table"),
            "labels": parsed.get("labels", []),
            "datasets": parsed.get("datasets", []),
        }
        answer = parsed.get("answer", "Here are the results.")

        return {
            "final_response": {
                "answer": answer,
                "visualization": visualization,
                "error": None,
            },
            "messages": [AIMessage(content=answer)],
        }
    except Exception:
        answer = f"Here are the query results ({len(data) if data else 0} rows)."
        return {
            "final_response": {
                "answer": answer,
                "visualization": {
                    "type": "table",
                    "labels": list(data[0].keys()) if data else [],
                    "datasets": [{"label": "Results", "data": data[:50] if data else []}],
                },
                "error": None,
            },
            "messages": [AIMessage(content=answer)],
        }


def handle_retry(state: AgentState) -> Dict[str, Any]:
    """Handle SQL retry with error memory - learns from past failures."""
    llm = get_llm()
    
    # Build comprehensive error history context
    error_history = state.get("error_history") or []
    error_context = ""
    if error_history:
        error_context = "\n=== PREVIOUS FAILED ATTEMPTS (DO NOT REPEAT THESE MISTAKES) ===\n"
        for i, err in enumerate(error_history, 1):
            error_context += f"""
--- Attempt {i} ---
SQL Query: {err.get('sql', 'N/A')}
Error: {err.get('error', 'N/A')}
Analysis: {err.get('analysis', 'N/A')}
"""
        error_context += "\n=== END OF ERROR HISTORY ===\n"

    retry_prompt = f"""You are fixing a failed SQL query. Learn from the previous errors and generate a CORRECT query.

{error_context}

CURRENT ERROR: {state['error']}
FAILED QUERY: {state['sql_query']}

ORIGINAL USER QUESTION: {state['query']}

{SYSTEM_PROMPT}

IMPORTANT RULES FOR RETRY:
1. The table name MUST be quoted: "General Machine 2000"
2. Use exact column names from schema: machine_id, machine_name, rpm, vibration_mm_per_s, power_load_kw, temperature_celsius, humidity_percent, speed_m_per_min, pressure_bar, qty, date, order_number, product_name
3. Use COALESCE for NULL handling in calculations
4. For ranking/scoring, calculate everything in SQL using window functions or computed columns
5. If the query is complex (ranking all machines), it will be exported to CSV automatically

Generate a CORRECTED SQL query. Return ONLY the SQL, no explanation.
"""

    response = llm.invoke([HumanMessage(content=retry_prompt)])
    sql_query = response.content.strip()

    if sql_query.startswith("```"):
        sql_query = sql_query.split("```")[1]
        if sql_query.startswith("sql"):
            sql_query = sql_query[3:]
        sql_query = sql_query.strip()

    return {"sql_query": sql_query, "error": None}


# ============== ROUTING FUNCTIONS ==============

def route_by_intent(state: AgentState) -> Literal["greeting", "followup", "analysis", "export"]:
    """Route to greeting, followup, analysis, or export flow based on intent."""
    return state["intent"]


def should_retry(state: AgentState) -> Literal["retry", "format", "end"]:
    """Determine if we should retry, format results, or end with error."""
    # Check if we already have a final response (from export success)
    if state.get("final_response"):
        return "end"
    if state.get("error") and state.get("retry_count", 0) < 3:
        return "retry"
    elif state.get("error"):
        return "end"
    return "format"


def route_after_validation(state: AgentState) -> Literal["analysis", "export", "retry"]:
    """Route after SQL validation - check for validation errors first."""
    # If validation found errors, go to retry
    if state.get("error"):
        return "retry"
    return "export" if state["intent"] == "export" else "analysis"


# ============== BUILD THE GRAPH WITH MEMORY ==============

def create_agent_graph():
    """Create and compile the LangGraph agent with memory."""

    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("handle_greeting", handle_greeting)
    workflow.add_node("handle_followup", handle_followup)
    workflow.add_node("generate_sql", generate_sql)
    workflow.add_node("validate_sql", validate_sql)
    workflow.add_node("execute_analysis", execute_analysis_query)
    workflow.add_node("execute_export", execute_export_query)
    workflow.add_node("format_response", format_analysis_response)
    workflow.add_node("handle_retry", handle_retry)

    # Set entry point
    workflow.set_entry_point("classify_intent")

    # Conditional routing after intent classification
    workflow.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "greeting": "handle_greeting",
            "followup": "handle_followup",
            "analysis": "generate_sql",
            "export": "generate_sql",
        },
    )

    # Greeting and followup flows end directly
    workflow.add_edge("handle_greeting", END)
    workflow.add_edge("handle_followup", END)

    # SQL generation flows to validation
    workflow.add_edge("generate_sql", "validate_sql")

    # Conditional routing after validation - includes retry path for validation errors
    workflow.add_conditional_edges(
        "validate_sql",
        route_after_validation,
        {
            "analysis": "execute_analysis",
            "export": "execute_export",
            "retry": "handle_retry",
        },
    )

    # Export flow ends directly
    workflow.add_edge("execute_export", END)

    # Analysis flow with retry logic
    workflow.add_conditional_edges(
        "execute_analysis",
        should_retry,
        {
            "retry": "handle_retry",
            "format": "format_response",
            "end": END,
        },
    )

    workflow.add_edge("handle_retry", "execute_analysis")
    workflow.add_edge("format_response", END)

    # Compile with memory checkpointer
    checkpointer = InMemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# Create the compiled graph with memory
agent = create_agent_graph()


def generate_helpful_error_message(query: str, error: str, error_history: list, retry_count: int) -> str:
    """Generate a helpful, user-friendly error message explaining why the request failed."""
    llm = get_llm()
    
    # Build error context
    error_summary = ""
    if error_history:
        error_summary = "I tried the following approaches:\n"
        for i, err in enumerate(error_history, 1):
            error_summary += f"- Attempt {i}: {err.get('analysis', err.get('error', 'Unknown error'))}\n"
    
    prompt = f"""You are Veda AI. The user asked a question but you couldn't fulfill it after {retry_count} attempts.

USER'S QUESTION: "{query}"

FINAL ERROR: {error}

{error_summary}

Generate a helpful, friendly response that:
1. Apologizes for not being able to help
2. Explains WHY you couldn't complete the request (in simple terms)
3. Suggests what the user could do differently or what data might be missing
4. Offers alternative questions they could ask

Common reasons for failure:
- The requested data/columns don't exist in the database (e.g., "breakdown logs", "failure history", "MTBF", "MTTR" - these require maintenance tracking data)
- The query was too complex for the available data
- The table only contains: machine_id, machine_name, order_number, product_name, qty, date, temperature_celsius, humidity_percent, speed_m_per_min, vibration_mm_per_s, rpm, pressure_bar, power_load_kw

Be conversational and helpful. Don't be technical about SQL errors.
Keep the response concise (3-4 sentences max).
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def run_agent(query: str, thread_id: str = "default") -> Dict[str, Any]:
    """Run the agent with a user query and conversation memory."""
    
    # Config with thread_id for memory persistence
    config = {"configurable": {"thread_id": thread_id}}
    
    # Input state - add user message to conversation with error tracking
    input_state = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "intent": None,
        "sql_query": None,
        "query_result": None,
        "final_response": None,
        "error": None,
        "retry_count": 0,
        "error_history": [],  # Track errors for learning
        "requires_export": False,  # Flag for export queries
    }

    # Invoke with config for memory
    result = agent.invoke(input_state, config=config)

    if result.get("final_response"):
        return result["final_response"]
    elif result.get("error"):
        # Generate a helpful error message using LLM
        error_history = result.get("error_history", [])
        retry_count = result.get("retry_count", 0)
        helpful_message = generate_helpful_error_message(
            query=query,
            error=result["error"],
            error_history=error_history,
            retry_count=retry_count
        )
        return {
            "answer": helpful_message,
            "visualization": None,
            "error": result["error"],
        }
    else:
        # No response and no error - generate explanation
        helpful_message = generate_helpful_error_message(
            query=query,
            error="The analysis completed but produced no results",
            error_history=[],
            retry_count=0
        )
        return {
            "answer": helpful_message,
            "visualization": None,
            "error": "No response generated",
        }


# Node name to user-friendly step mapping
NODE_STEPS = {
    "classify_intent": {"step": "Understanding your question", "icon": "🧠"},
    "handle_greeting": {"step": "Preparing response", "icon": "💬"},
    "handle_followup": {"step": "Explaining methodology", "icon": "💡"},
    "generate_sql": {"step": "Generating database query", "icon": "📝"},
    "validate_sql": {"step": "Validating query", "icon": "✅"},
    "execute_analysis": {"step": "Executing analysis", "icon": "⚡"},
    "execute_export": {"step": "Preparing export", "icon": "📦"},
    "format_response": {"step": "Formatting results", "icon": "📊"},
    "handle_retry": {"step": "Retrying query", "icon": "🔄"},
}


async def run_agent_stream(query: str, thread_id: str = "default"):
    """
    Run the agent with streaming updates.
    Yields events as the agent progresses through nodes.
    """
    import asyncio

    config = {"configurable": {"thread_id": thread_id}}

    input_state = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "intent": None,
        "sql_query": None,
        "query_result": None,
        "final_response": None,
        "error": None,
        "retry_count": 0,
        "error_history": [],
        "requires_export": False,
    }

    # Stream with updates mode to get node-by-node progress
    final_result = None

    for event in agent.stream(input_state, config=config, stream_mode="updates"):
        for node_name, node_output in event.items():
            # Get user-friendly step info
            step_info = NODE_STEPS.get(node_name, {"step": node_name, "icon": "⏳"})

            # Yield progress event
            yield {
                "type": "progress",
                "data": {
                    "node": node_name,
                    "step": step_info["step"],
                    "icon": step_info["icon"],
                },
            }

            # Check if we have intent info to share
            if node_name == "classify_intent" and node_output.get("intent"):
                yield {
                    "type": "intent",
                    "data": {"intent": node_output["intent"]},
                }

            # Check if we have the final response
            if node_output.get("final_response"):
                final_result = node_output["final_response"]

            # Small delay to make streaming visible
            await asyncio.sleep(0.1)

    # Yield the final result
    if final_result:
        yield {
            "type": "result",
            "data": final_result,
        }
    else:
        # Get the final state to check for errors
        final_state = agent.get_state(config)
        state_values = final_state.values if final_state else {}
        
        error = state_values.get("error")
        error_history = state_values.get("error_history", [])
        retry_count = state_values.get("retry_count", 0)
        
        if error:
            # Generate helpful error message
            helpful_message = generate_helpful_error_message(
                query=query,
                error=error,
                error_history=error_history,
                retry_count=retry_count
            )
            yield {
                "type": "result",
                "data": {
                    "answer": helpful_message,
                    "visualization": None,
                    "error": error,
                },
            }
        else:
            # Fallback for truly unknown errors
            yield {
                "type": "result",
                "data": {
                    "answer": "I wasn't able to complete your request. This might be because the data you're looking for isn't available in the current dataset. Try asking about machine performance metrics like RPM, temperature, vibration, or power load.",
                    "visualization": None,
                    "error": "No response generated",
                },
            }
