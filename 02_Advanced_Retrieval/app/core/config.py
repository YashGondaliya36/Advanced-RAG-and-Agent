from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Advanced Retrieval (Hybrid RAG)"
    GOOGLE_API_KEY: str = ""
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    BM25_PERSIST_PATH: str = "./bm25_index.pkl"

    class Config:
        env_file = "../.env"

settings = Settings()