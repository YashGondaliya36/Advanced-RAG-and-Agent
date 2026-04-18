from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "OmniCorp GraphRAG API"
    VERSION: str = "1.0.0"
    
    # Root level .env file
    GOOGLE_API_KEY: str
    
    # Neo4j Credentials
    NEO4J_URI: str
    NEO4J_USERNAME: str
    NEO4J_PASSWORD: str
    NEO4J_DATABASE: str = "fa3ca9cd" # Fallback to Aura ID from screenshot if not in .env

    class Config:
        env_file = "../.env"
        extra = "ignore"

settings = Settings()
