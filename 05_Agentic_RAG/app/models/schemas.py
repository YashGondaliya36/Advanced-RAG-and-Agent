from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str
    max_retries: int = Field(default=2, description="How many times the agent should retry if it finds bad documents")

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    agent_trajectory: List[str]

class IngestRequest(BaseModel):
    text: str
    source_name: Optional[str] = "unknown_source"

class IngestResponse(BaseModel):
    message: str
    chunks_processed: int