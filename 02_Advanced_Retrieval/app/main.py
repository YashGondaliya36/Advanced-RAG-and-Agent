from fastapi import FastAPI
from app.api.endpoints import router
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Module 2: Advanced Retrieval showing Dense vs Sparse vs Hybrid Search with Reciprocal Rank Fusion.",
    version="2.0.0"
)

app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "docs": "/docs",
        "strategies_available": ["dense", "sparse", "hybrid"]
    }