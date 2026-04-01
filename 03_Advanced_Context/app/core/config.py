from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Advanced Context RAG (HyDE & Parent-Child)"
    GOOGLE_API_KEY: str = ""
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    BYTE_STORE_DIR: str = "./byte_store"

    class Config:
        env_file = "../.env"

settings = Settings()