"""FastAPI server for the LangGraph Data Analyst Agent with streaming."""
import os
import json
import asyncio
from typing import Optional, AsyncGenerator
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Import agent after env vars are loaded
from agent import run_agent, run_agent_stream

# Initialize FastAPI app
app = FastAPI(
    title="Data Analyst Agent API",
    description="LangGraph-powered agent for analyzing machine production data",
    version="1.0.0",
)

# CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== REQUEST/RESPONSE MODELS ==============


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""
    query: str
    thread_id: Optional[str] = "default"


class VisualizationResponse(BaseModel):
    """Visualization data structure."""
    type: str
    labels: list
    datasets: list


class ChatResponse(BaseModel):
    """Response from chat endpoint."""
    answer: str
    visualization: Optional[VisualizationResponse] = None
    download_url: Optional[str] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str


# ============== ENDPOINTS ==============


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Non-streaming chat endpoint."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        result = run_agent(query=request.query, thread_id=request.thread_id)

        return ChatResponse(
            answer=result.get("answer", ""),
            visualization=result.get("visualization"),
            download_url=result.get("download_url"),
            error=result.get("error"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint using Server-Sent Events (SSE).
    Streams node updates as the agent processes the query.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            async for event in run_agent_stream(
                query=request.query, thread_id=request.thread_id
            ):
                # Format as SSE
                yield f"data: {json.dumps(event)}\n\n"

            # Signal completion
            yield "data: [DONE]\n\n"

        except Exception as e:
            error_event = {"type": "error", "data": {"message": str(e)}}
            yield f"data: {json.dumps(error_event)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/agent/analyze", response_model=ChatResponse)
async def analyze(request: ChatRequest):
    """Alternative endpoint matching the spec."""
    return await chat(request)


# ============== STARTUP ==============


@app.on_event("startup")
async def startup_event():
    """Validate environment variables on startup."""
    required_vars = ["OPENAI_API_KEY", "SUPABASE_URL", "SUPABASE_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        print(f"WARNING: Missing environment variables: {missing}")


# ============== RUN INSTRUCTIONS ==============
# uvicorn server:app --reload --port 8000
