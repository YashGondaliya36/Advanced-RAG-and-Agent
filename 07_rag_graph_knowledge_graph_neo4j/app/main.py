import logging
from fastapi import FastAPI
from app.api.endpoints import router
from app.core.config import settings

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Knowledge Graph RAG using Neo4j and Gemini"
)

# Include the router
app.include_router(router, prefix="/api/v1")

@app.get("/")
async def root():
    return {
        "message": "Welcome to the GraphRAG API (Module 07).",
        "docs_url": "/docs"
    }
