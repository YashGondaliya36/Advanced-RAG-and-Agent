from fastapi import FastAPI
from app.api.endpoints import router
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Module 4: Post-Retrieval Optimization with FlashRank Reranking and Long Context Reordering.",
    version="4.0.0"
)

app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "docs": "/docs",
        "features": ["FlashRank Cross-Encoder Reranking", "Lost-in-the-Middle Reordering"]
    }