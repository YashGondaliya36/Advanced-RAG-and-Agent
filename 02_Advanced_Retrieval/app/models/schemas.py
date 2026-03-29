from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str
    strategy: str = Field(
        default="hybrid", 
        description="Retrieval strategy: 'dense', 'sparse', or 'hybrid'"
    )
    k: int = Field(default=4, description="Number of chunks to retrieve")

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    strategy_used: str

class IngestRequest(BaseModel):
    text: str
    source_name: Optional[str] = "unknown_source"

class IngestResponse(BaseModel):
    message: str
    chunks_processed: int