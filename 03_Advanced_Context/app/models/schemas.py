from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str
    use_hyde: bool = Field(
        default=False, 
        description="Whether to use Hypothetical Document Embeddings (HyDE) before retrieving."
    )
    k: int = Field(default=4, description="Number of child chunks to retrieve (will resolve to fewer parent chunks)")

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    hypothetical_answer_generated: Optional[str] = None

class IngestRequest(BaseModel):
    text: str
    source_name: Optional[str] = "unknown_source"

class IngestResponse(BaseModel):
    message: str
    parent_chunks: int
    child_chunks: int