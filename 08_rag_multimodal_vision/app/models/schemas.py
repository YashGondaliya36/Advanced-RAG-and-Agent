from pydantic import BaseModel
from typing import List

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]

class IngestResponse(BaseModel):
    message: str
    documents_ingested: int

