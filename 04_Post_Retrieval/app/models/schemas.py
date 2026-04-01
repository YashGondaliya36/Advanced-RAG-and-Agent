from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str
    use_reranker: bool = Field(default=True, description="Use FlashRank Cross-Encoder to re-score chunks")
    use_reordering: bool = Field(default=True, description="Combat 'Lost in the Middle' by reordering context")
    initial_k: int = Field(default=10, description="Chunks to fetch initially before compression/reranking")
    final_k: int = Field(default=3, description="Final chunks to pass to the LLM")

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    optimization_steps_applied: List[str]

class IngestRequest(BaseModel):
    text: str
    source_name: Optional[str] = "unknown_source"

class IngestResponse(BaseModel):
    message: str
    chunks_processed: int