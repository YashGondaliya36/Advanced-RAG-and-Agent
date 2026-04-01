from fastapi import FastAPI
from app.api.endpoints import router
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Module 3: Advanced Context handling showing Small-to-Big Retrieval (Parent-Child) and HyDE (Hypothetical Document Embeddings).",
    version="3.0.0"
)

app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "docs": "/docs",
        "features": ["Small-to-Big Retrieval", "HyDE Strategy"]
    }