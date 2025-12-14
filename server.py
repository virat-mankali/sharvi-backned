"""
FastAPI Server for Veda AI - Cursor-style Streaming Data Analyst
================================================================
Provides SSE streaming endpoints for real-time agent updates.
"""
import os
import json
import asyncio
from typing import Optional, AsyncGenerator
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Import agent after env vars are loaded
from agent import run_agent, run_agent_stream

# =============================================================================
# FASTAPI APP SETUP
# =============================================================================

app = FastAPI(
    title="Veda AI - Data Analyst Agent",
    description="Cursor-style streaming data analyst powered by LangGraph",
    version="2.0.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class ChatRequest(BaseModel):
    """Request body for chat endpoints."""
    query: str
    thread_id: Optional[str] = "default"


class VisualizationData(BaseModel):
    """Chart visualization data."""
    type: str
    labels: list
    datasets: list
    options: Optional[dict] = None


class ChatResponse(BaseModel):
    """Response from non-streaming chat endpoint."""
    answer: str
    visualization: Optional[dict] = None
    download_url: Optional[str] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    model: str


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "model": os.getenv("OPENAI_MODEL", "gpt-4o")
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Non-streaming chat endpoint.
    Returns complete response after agent finishes.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        result = run_agent(
            query=request.query,
            thread_id=request.thread_id or "default"
        )
        
        return ChatResponse(
            answer=result.get("answer", ""),
            visualization=result.get("visualization"),
            download_url=result.get("download_url"),
            error=result.get("error")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint using Server-Sent Events (SSE).
    
    Streams real-time updates as the agent thinks and executes:
    
    Event Types:
    - thinking: Agent's reasoning process
    - tool_call: Tool being invoked with arguments
    - tool_start: Tool execution starting
    - tool_result: Tool execution completed
    - tool_error: Tool execution failed
    - token: Individual LLM token
    - message: Complete message chunk
    - data_received: Query results received
    - visualization: Chart data ready
    - download: CSV download URL ready
    - done: Agent finished, final response
    - error: Error occurred
    
    Frontend should handle each event type appropriately for
    Cursor-style real-time display.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events from agent stream."""
        try:
            # Send initial event
            yield format_sse({
                "type": "start",
                "message": "Processing your question...",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Stream agent events
            async for event in run_agent_stream(
                query=request.query,
                thread_id=request.thread_id or "default"
            ):
                yield format_sse(event)
                
                # Small delay to prevent overwhelming the client
                await asyncio.sleep(0.01)
            
            # Signal completion
            yield "data: [DONE]\n\n"
            
        except asyncio.CancelledError:
            # Client disconnected
            yield format_sse({
                "type": "cancelled",
                "message": "Request cancelled"
            })
        except Exception as e:
            yield format_sse({
                "type": "error",
                "message": str(e)
            })
            yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "Content-Type": "text/event-stream",
        }
    )


def format_sse(data: dict) -> str:
    """Format data as Server-Sent Event."""
    return f"data: {json.dumps(data, default=str)}\n\n"


# =============================================================================
# ALTERNATIVE ENDPOINTS (for compatibility)
# =============================================================================

@app.post("/agent/analyze", response_model=ChatResponse)
async def analyze(request: ChatRequest):
    """Alternative endpoint matching original spec."""
    return await chat(request)


@app.post("/agent/stream")
async def agent_stream(request: ChatRequest):
    """Alternative streaming endpoint."""
    return await chat_stream(request)


# =============================================================================
# STARTUP/SHUTDOWN
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Validate environment on startup."""
    required_vars = ["OPENAI_API_KEY", "SUPABASE_URL", "SUPABASE_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        print(f"⚠️  WARNING: Missing environment variables: {missing}")
    else:
        print("✅ All required environment variables present")
    
    print(f"🤖 Model: {os.getenv('OPENAI_MODEL', 'gpt-4o')}")
    print("🚀 Veda AI server ready")


# =============================================================================
# RUN INSTRUCTIONS
# =============================================================================
# Development: uvicorn server:app --reload --port 8000
# Production:  uvicorn server:app --host 0.0.0.0 --port 8000
