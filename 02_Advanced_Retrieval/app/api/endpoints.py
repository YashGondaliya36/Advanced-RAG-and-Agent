from fastapi import APIRouter, HTTPException
from app.models.schemas import QueryRequest, QueryResponse, IngestRequest, IngestResponse
from app.services.rag_service import HybridRAGService
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

rag_service = None

def get_rag_service() -> HybridRAGService:
    global rag_service
    if rag_service is None:
        try:
            rag_service = HybridRAGService()
        except Exception as e:
            logger.error(f"Failed to initialize RAG Service: {e}")
            raise HTTPException(status_code=500, detail="RAG Service config error. Check API keys.")
    return rag_service

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: IngestRequest):
    """
    Ingest raw text into the Hybrid RAG system.
    This chunks the text and stores it in BOTH ChromaDB (Dense) and BM25 (Sparse) indexes.
    """
    service = get_rag_service()
    try:
        chunks = await service.ingest_text(request.text, request.source_name)
        return IngestResponse(
            message="Successfully ingested document into Dense and Sparse indexes.",
            chunks_processed=chunks
        )
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query", response_model=QueryResponse)
async def query_system(request: QueryRequest):
    """
    Query the system using Advanced Hybrid Routing.
    You can specify 'dense' (semantic), 'sparse' (exact keywords), or 'hybrid' (combined).
    """
    service = get_rag_service()
    try:
        result = await service.query(
            question=request.query,
            strategy=request.strategy,
            k=request.k
        )
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            strategy_used=result["strategy"]
        )
    except Exception as e:
        logger.error(f"Error during querying: {e}")
        raise HTTPException(status_code=500, detail=str(e))