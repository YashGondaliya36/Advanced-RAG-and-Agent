from fastapi import APIRouter, HTTPException
from app.models.schemas import QueryRequest, QueryResponse, IngestRequest, IngestResponse
from app.services.rag_service import NaiveRAGService
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

# Lazy initialization of the RAG service to ensure env vars are loaded first
rag_service = None

def get_rag_service() -> NaiveRAGService:
    global rag_service
    if rag_service is None:
        try:
            rag_service = NaiveRAGService()
        except Exception as e:
            logger.error(f"Failed to initialize RAG Service: {e}")
            raise HTTPException(status_code=500, detail="RAG Service configuration error. Check API keys.")
    return rag_service

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: IngestRequest):
    """
    Ingest raw text into the naive RAG system.
    This chunks the text and stores dense vector embeddings in ChromaDB.
    """
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
    """
    Query the naive RAG system.
    Performs a semantic search and passes context to Google Gemini to generate an answer.
    """
    service = get_rag_service()
    try:
        result = await service.query(request.query)
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        logger.error(f"Error during querying: {e}")
        raise HTTPException(status_code=500, detail=str(e))
