from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class QueryRequest(BaseModel):
    query: str
    use_cache: bool = Field(default=True, description="Enable Semantic Caching to bypass LLM generation for similar queries.")

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    cache_hit: bool
    latency_ms: Optional[float] = None

class IngestRequest(BaseModel):
    text: str
    source_name: Optional[str] = "unknown_source"

class IngestResponse(BaseModel):
    message: str
    chunks_processed: int

class EvalRequest(BaseModel):
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None

class EvalResponse(BaseModel):
    faithfulness_score: float
    answer_relevancy_score: float