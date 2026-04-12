from fastapi import FastAPI
from app.api.endpoints import router
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Module 6: Enterprise Operations with Semantic Caching (Cost/Latency) and RAGAS Evaluation (Quality).",
    version="6.0.0"
)

app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "docs": "/docs",
        "features": ["Semantic Caching", "RAGAS Framework Integration"]
    }