from fastapi import FastAPI
from app.api.endpoints import router
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Module 5: Agentic & Autonomous RAG using LangGraph for Corrective RAG (CRAG) loops.",
    version="5.0.0"
)

app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "docs": "/docs",
        "features": ["LangGraph State Orchestration", "LLM Document Grader", "Query Rewriting Loops"]
    }