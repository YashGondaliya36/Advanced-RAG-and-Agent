from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from typing import Optional
from app.models.schemas import QueryRequest, QueryResponse, IngestResponse
from app.services.multimodal_rag_service import MultimodalRAGService
from app.services.hybrid_rag_service import HybridMultimodalRAGService

router = APIRouter()

# Dependency to get the Pure Gemini RAG Service
def get_rag_service():
    return MultimodalRAGService()

# Dependency to get the Hybrid RAG Service
def get_hybrid_service():
    return HybridMultimodalRAGService()

# ==========================================
# PURE GEMINI ENDPOINTS (Rate Limit Prone)
# ==========================================

@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    service: MultimodalRAGService = Depends(get_rag_service)
):
    """
    Ingest a file (PDF, PNG, JPEG, text) into the Multimodal Vector Store.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
        
    try:
        contents = await file.read()
        num_ingested = await service.ingest_file(contents, file.filename)
        
        return IngestResponse(
            message=f"Successfully ingested {file.filename}",
            documents_ingested=num_ingested
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest/text", response_model=IngestResponse)
async def ingest_text(
    text: str = Form(...),
    source: str = Form("user_input"),
    service: MultimodalRAGService = Depends(get_rag_service)
):
    """
    Ingest raw text into the Multimodal Vector Store.
    """
    try:
        num_ingested = await service.ingest_text(text, source)
        
        return IngestResponse(
            message=f"Successfully ingested text from {source}",
            documents_ingested=num_ingested
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    service: MultimodalRAGService = Depends(get_rag_service)
):
    """
    Query the Multimodal Vector Store and generate an answer using Gemini Vision.
    """
    try:
        result = await service.query(request.question, request.top_k)
        
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# HYBRID ENDPOINTS (Local Embeddings + Cloud Generation)
# ==========================================

@router.post("/hybrid/ingest/file", response_model=IngestResponse)
async def hybrid_ingest_file(
    file: UploadFile = File(...),
    service: HybridMultimodalRAGService = Depends(get_hybrid_service)
):
    """
    Ingest a file using Local Open-Source CLIP embeddings (No API Rate Limits).
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
        
    try:
        contents = await file.read()
        num_ingested = await service.ingest_file(contents, file.filename)
        
        return IngestResponse(
            message=f"Successfully ingested {file.filename} via Hybrid Pipeline",
            documents_ingested=num_ingested
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/hybrid/ingest/text", response_model=IngestResponse)
async def hybrid_ingest_text(
    text: str = Form(...),
    source: str = Form("user_input"),
    service: HybridMultimodalRAGService = Depends(get_hybrid_service)
):
    """
    Ingest text using Local Open-Source CLIP embeddings.
    """
    try:
        num_ingested = await service.ingest_text(text, source)
        
        return IngestResponse(
            message=f"Successfully ingested text from {source} via Hybrid Pipeline",
            documents_ingested=num_ingested
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/hybrid/query", response_model=QueryResponse)
async def hybrid_query(
    request: QueryRequest,
    service: HybridMultimodalRAGService = Depends(get_hybrid_service)
):
    """
    Query using Local Open-Source embeddings, and generate answer via Gemini.
    """
    try:
        result = await service.query(request.question, request.top_k)
        
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


