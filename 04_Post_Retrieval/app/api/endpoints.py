from fastapi import APIRouter, HTTPException
from app.models.schemas import QueryRequest, QueryResponse, IngestRequest, IngestResponse
from app.services.rag_service import PostRetrievalService
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

rag_service = None

def get_rag_service() -> PostRetrievalService:
    global rag_service
    if rag_service is None:
        try:
            rag_service = PostRetrievalService()
        except Exception as e:
            logger.error(f"Failed to initialize RAG Service: {e}")
            raise HTTPException(status_code=500, detail="RAG Service config error. Check API keys.")
    return rag_service

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: IngestRequest):
    service = get_rag_service()
    try:
        chunks = await service.ingest_text(request.text, request.source_name)
        return IngestResponse(
            message="Successfully ingested document.",
            chunks_processed=chunks
        )
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query", response_model=QueryResponse)
async def query_system(request: QueryRequest):
    service = get_rag_service()
    try:
        result = await service.query(
            question=request.query,
            use_reranker=request.use_reranker,
            use_reordering=request.use_reordering,
            initial_k=request.initial_k,
            final_k=request.final_k
        )
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            optimization_steps_applied=result["optimization_steps_applied"]
        )
    except Exception as e:
        logger.error(f"Error during querying: {e}")
        raise HTTPException(status_code=500, detail=str(e))