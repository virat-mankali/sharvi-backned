"""
State and Type Definitions for Veda AI
======================================
Pydantic models and TypedDict definitions for the agent.
"""
from typing import TypedDict, Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field


# =============================================================================
# VISUALIZATION TYPES
# =============================================================================

class ChartDataset(BaseModel):
    """Single dataset for a chart."""
    label: str
    data: List[float]
    backgroundColor: Optional[str] = None
    borderColor: Optional[str] = None
    borderWidth: Optional[int] = 1
    fill: Optional[bool] = None
    tension: Optional[float] = None


class VisualizationData(BaseModel):
    """Chart-ready visualization data for frontend."""
    type: Literal["bar", "line", "pie", "doughnut", "scatter"]
    labels: List[str]
    datasets: List[ChartDataset]
    options: Optional[Dict[str, Any]] = None


# =============================================================================
# RESPONSE TYPES
# =============================================================================

class AgentResponse(BaseModel):
    """Complete response from the agent."""
    answer: str = Field(description="Natural language answer/insight")
    visualization: Optional[VisualizationData] = Field(
        default=None, 
        description="Chart data for frontend rendering"
    )
    download_url: Optional[str] = Field(
        default=None, 
        description="Signed URL for CSV download"
    )
    error: Optional[str] = Field(
        default=None, 
        description="Error message if something went wrong"
    )


# =============================================================================
# STREAMING EVENT TYPES
# =============================================================================

class StreamEvent(BaseModel):
    """Base streaming event."""
    type: str
    timestamp: Optional[str] = None


class ThinkingEvent(StreamEvent):
    """Agent is reasoning."""
    type: Literal["thinking"] = "thinking"
    content: str


class ToolCallEvent(StreamEvent):
    """Agent is calling a tool."""
    type: Literal["tool_call"] = "tool_call"
    tool: str
    args: Dict[str, Any]


class ToolStartEvent(StreamEvent):
    """Tool execution starting."""
    type: Literal["tool_start"] = "tool_start"
    tool: str
    message: str
    sql: Optional[str] = None


class ToolResultEvent(StreamEvent):
    """Tool execution completed."""
    type: Literal["tool_result"] = "tool_result"
    tool: str
    message: str
    row_count: Optional[int] = None
    preview: Optional[List[Dict]] = None


class ToolErrorEvent(StreamEvent):
    """Tool execution failed."""
    type: Literal["tool_error"] = "tool_error"
    tool: Optional[str] = None
    message: str


class TokenEvent(StreamEvent):
    """LLM token streamed."""
    type: Literal["token"] = "token"
    content: str
    node: Optional[str] = None


class MessageEvent(StreamEvent):
    """Complete message chunk."""
    type: Literal["message"] = "message"
    content: str
    node: Optional[str] = None


class DataReceivedEvent(StreamEvent):
    """Query data received."""
    type: Literal["data_received"] = "data_received"
    row_count: int
    preview: List[Dict]


class VisualizationEvent(StreamEvent):
    """Visualization data ready."""
    type: Literal["visualization"] = "visualization"
    data: Dict[str, Any]


class DownloadEvent(StreamEvent):
    """CSV download ready."""
    type: Literal["download"] = "download"
    url: str
    row_count: Optional[int] = None


class DoneEvent(StreamEvent):
    """Agent finished processing."""
    type: Literal["done"] = "done"
    answer: str
    visualization: Optional[Dict[str, Any]] = None
    download_url: Optional[str] = None


class ErrorEvent(StreamEvent):
    """Error occurred."""
    type: Literal["error"] = "error"
    message: str


# =============================================================================
# AGENT STATE (for custom graph if needed)
# =============================================================================

class AgentState(TypedDict):
    """State for custom LangGraph agent."""
    messages: List[Dict[str, Any]]
    query: str
    thread_id: str
    intent: Optional[Literal["analysis", "export", "comparison", "trend"]]
    sql_query: Optional[str]
    query_result: Optional[List[Dict[str, Any]]]
    visualization: Optional[Dict[str, Any]]
    download_url: Optional[str]
    final_response: Optional[str]
    error: Optional[str]
    retry_count: int
