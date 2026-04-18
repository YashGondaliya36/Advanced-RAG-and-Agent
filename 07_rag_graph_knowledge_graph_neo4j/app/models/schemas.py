from pydantic import BaseModel, Field
from typing import List, Optional

# --- API Request Models ---

class IngestRequest(BaseModel):
    text: str = Field(..., description="The raw text to extract knowledge from.")
    source: str = Field(default="internal_wiki", description="The source of the document.")

class QueryRequest(BaseModel):
    query: str = Field(..., description="The natural language question to ask the Graph.")
    
class QueryResponse(BaseModel):
    answer: str
    cypher_query: str
    latency_ms: float

# --- Knowledge Graph Ontology ---
# We define exactly what the LLM is allowed to extract.

ALLOWED_NODES = [
    "Person",
    "Team",
    "Service",
    "Database",
    "Infrastructure",
    "Incident"
]

ALLOWED_RELATIONSHIPS = [
    "MANAGES",       # Person -> Person
    "BELONGS_TO",    # Person -> Team
    "OWNS",          # Team -> Service/Database
    "DEPENDS_ON",    # Service -> Service
    "CONNECTS_TO",   # Service -> Database
    "DEPLOYED_ON",   # Service -> Infrastructure
    "AFFECTS"        # Incident -> Service/Database/Infrastructure
]
