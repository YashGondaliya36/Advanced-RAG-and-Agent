from fastapi import APIRouter, HTTPException
from app.models.schemas import QueryRequest, QueryResponse, IngestRequest, IngestResponse
from app.services.rag_service import ContextRAGService
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

rag_service = None

def get_rag_service() -> ContextRAGService:
    global rag_service
    if rag_service is None:
        try:
            rag_service = ContextRAGService()
        except Exception as e:
            logger.error(f"Failed to initialize RAG Service: {e}")
            raise HTTPException(status_code=500, detail="RAG Service config error. Check API keys.")
    return rag_service

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: IngestRequest):
    """
    Ingest text using Small-to-Big (Hierarchical) Chunking.
    Tiny child chunks go to ChromaDB. Massive parent chunks go to the ByteStore.
    """
    service = get_rag_service()
    try:
        stats = await service.ingest_text(request.text, request.source_name)
        return IngestResponse(
            message="Successfully ingested document using Parent-Child relationships.",
            parent_chunks=stats["parent_chunks"],
            child_chunks=stats["child_chunks"]
        )
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query", response_model=QueryResponse)
async def query_system(request: QueryRequest):
    """
    Query the system using Small-to-Big retrieval.
    Optionally toggle `use_hyde: true` to generate a hypothetical answer before searching.
    """
    service = get_rag_service()
    try:
        result = await service.query(
            question=request.query,
            use_hyde=request.use_hyde,
            k=request.k
        )
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            hypothetical_answer_generated=result["hypothetical_answer_generated"]
        )
    except Exception as e:
        logger.error(f"Error during querying: {e}")
        raise HTTPException(status_code=500, detail=str(e))