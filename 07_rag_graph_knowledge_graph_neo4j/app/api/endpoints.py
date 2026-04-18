from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any

from app.models.schemas import IngestRequest, QueryRequest, QueryResponse
from app.services.graph_rag_service import GraphRAGService

router = APIRouter()

# Dependency to get the service instance
def get_graph_service():
    # In a real app, you might use a Singleton or proper dependency injection
    # to avoid reconnecting to Neo4j on every request.
    return GraphRAGService()

@router.post("/ingest", response_model=Dict[str, Any])
async def ingest_document(request: IngestRequest, service: GraphRAGService = Depends(get_graph_service)):
    """
    Takes raw text, extracts entities/relationships using Gemini, and pushes them to Neo4j.
    """
    try:
        result = await service.ingest_text(request.text, request.source)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query", response_model=QueryResponse)
async def query_graph(request: QueryRequest, service: GraphRAGService = Depends(get_graph_service)):
    """
    Translates natural language to Cypher, queries the Neo4j Knowledge Graph, and returns the answer.
    """
    try:
        result = await service.query_graph(request.query)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
