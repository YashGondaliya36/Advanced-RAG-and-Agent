from fastapi import APIRouter, HTTPException
from app.models.schemas import QueryRequest, QueryResponse, IngestRequest, IngestResponse, EvalRequest, EvalResponse
from app.services.rag_service import EnterpriseRAGService
from app.services.eval_service import RagasEvaluator
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

rag_service = None

def get_rag_service() -> EnterpriseRAGService:
    global rag_service
    if rag_service is None:
        try:
            rag_service = EnterpriseRAGService()
        except Exception as e:
            logger.error(f"Failed to initialize RAG Service: {e}")
            raise HTTPException(status_code=500, detail="RAG Service config error.")
    return rag_service

@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: IngestRequest):
    service = get_rag_service()
    chunks = await service.ingest_text(request.text, request.source_name)
    return IngestResponse(message="Ingested successfully.", chunks_processed=chunks)

@router.post("/query", response_model=QueryResponse)
async def query_system(request: QueryRequest):
    service = get_rag_service()
    result = await service.query(request.query, request.use_cache)
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        cache_hit=result["cache_hit"],
        latency_ms=result["latency_ms"]
    )

@router.post("/evaluate", response_model=EvalResponse)
async def evaluate_rag(request: EvalRequest):
    """
    Evaluate a specific RAG response using RAGAS metrics.
    Requires proper RAGAS environment configuration (OpenAI/LangChain LLM setup).
    """
    scores = RagasEvaluator.evaluate_response(
        question=request.question,
        answer=request.answer,
        contexts=request.contexts
    )
    return EvalResponse(
        faithfulness_score=scores["faithfulness_score"],
        answer_relevancy_score=scores["answer_relevancy_score"]
    )