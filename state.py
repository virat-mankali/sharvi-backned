"""State definitions for the LangGraph Data Analyst Agent."""
from typing import TypedDict, Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field


class VisualizationData(BaseModel):
    """Structured visualization data for frontend charts."""
    type: Literal["bar", "line", "pie", "table"] = Field(description="Chart type")
    labels: List[str] = Field(default_factory=list, description="X-axis labels")
    datasets: List[Dict[str, Any]] = Field(default_factory=list, description="Chart datasets")


class AgentResponse(BaseModel):
    """Structured response from the agent."""
    answer: str = Field(description="Natural language answer/insight")
    visualization: Optional[VisualizationData] = Field(default=None, description="Chart data for frontend")
    download_url: Optional[str] = Field(default=None, description="Signed URL for CSV export")
    error: Optional[str] = Field(default=None, description="Error message if any")


class AgentState(TypedDict):
    """State for the LangGraph agent."""
    messages: List[Dict[str, Any]]
    query: str
    thread_id: str
    intent: Optional[Literal["analysis", "export"]]
    sql_query: Optional[str]
    query_result: Optional[List[Dict[str, Any]]]
    final_response: Optional[Dict[str, Any]]
    error: Optional[str]
    retry_count: int
