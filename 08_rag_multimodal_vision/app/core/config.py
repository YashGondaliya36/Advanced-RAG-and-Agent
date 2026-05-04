from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    PROJECT_NAME: str = "Multimodal RAG API"
    VERSION: str = "1.0.0"
    
    # API Keys
    GOOGLE_API_KEY: str = Field(..., description="Google API Key for Gemini")
    
    # Chroma Config
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    
    class Config:
        env_file = "../.env"
        extra = "ignore"

settings = Settings()


