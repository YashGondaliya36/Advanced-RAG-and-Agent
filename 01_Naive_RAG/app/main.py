from fastapi import FastAPI
from app.api.endpoints import router
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="A foundational implementation of Naive RAG using FastAPI, LangChain, ChromaDB, and Google Gemini.",
    version="1.0.0"
)

# Include our modular API endpoints
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "docs": "/docs",
        "health": "ok"
    }
