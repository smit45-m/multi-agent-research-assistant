"""
Configuration settings for the Multi-Agent Research Assistant.
Uses Groq's OpenAI-compatible API for fast LLM inference.
"""
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or .env file.
    Groq provides an OpenAI-compatible API, so we reuse the OPENAI_* keys.
    """
    # LLM Configuration (Groq via OpenAI-compatible endpoint)
    OPENAI_API_KEY: str = "your-groq-api-key"
    OPENAI_BASE_URL: str = "https://api.groq.com/openai/v1"
    OPENAI_MODEL_NAME: str = "llama-3.3-70b-versatile"
    
    # Embedding Model (local HuggingFace - free, no API key needed)
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Vector Store
    VECTOR_STORE_PATH: str = "data/vector_store"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # API Server
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    MAX_CONCURRENT_REQUESTS: int = 50
    
    # Optional API Key Authentication
    API_KEY: Optional[str] = None
    
    # Optional Web Search API Key
    SEARCH_API_KEY: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    """
    Returns cached application settings.
    Settings are loaded once from environment/.env and reused.
    """
    return Settings()
