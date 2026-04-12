from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Enterprise RAG (Semantic Caching & Eval)"
    GOOGLE_API_KEY: str = ""
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CACHE_PERSIST_DIR: str = "./semantic_cache.sqlite3"

    class Config:
        env_file = "../.env"

settings = Settings()