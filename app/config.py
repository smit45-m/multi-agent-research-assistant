"""
Configuration settings for the Multi-Agent Research Assistant.
Uses any OpenAI-compatible API (Groq, OpenAI, vLLM, Ollama) for LLM inference.
"""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

# Placeholder values that mean "no real key was configured".
_PLACEHOLDER_KEYS = {
    "",
    "your-groq-api-key",
    "your-groq-or-openai-api-key-here",
    "your-api-key",
    "sk-test-key",
    "changeme",
}


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or .env file.
    Groq provides an OpenAI-compatible API, so we reuse the OPENAI_* keys.
    """

    # LLM Configuration (any OpenAI-compatible endpoint)
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.groq.com/openai/v1"
    OPENAI_MODEL_NAME: str = "llama-3.3-70b-versatile"
    LLM_TIMEOUT_S: float = 30.0
    LLM_MAX_RETRIES: int = 1

    # Embedding Model (local HuggingFace - free, no API key needed).
    # If unavailable, the system falls back to OpenAI embeddings (when a key
    # is configured) and finally to deterministic offline hashing embeddings.
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
    RATE_LIMIT_PER_MINUTE: int = 120

    # External retrieval toggles
    WEB_SEARCH_ENABLED: bool = True
    EXTERNAL_SOURCES_TIMEOUT_S: float = 6.0

    # Optional shared task store / rate limiting backend for horizontal scaling.
    # When unset, an in-process store is used (single-replica deployments only).
    REDIS_URL: Optional[str] = None

    # Optional API Key Authentication
    API_KEY: Optional[str] = None

    # Optional Web Search API Key
    SEARCH_API_KEY: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def llm_available(self) -> bool:
        """True when a real (non-placeholder) LLM API key is configured."""
        return self.OPENAI_API_KEY.strip() not in _PLACEHOLDER_KEYS


@lru_cache()
def get_settings() -> Settings:
    """
    Returns cached application settings.
    Settings are loaded once from environment/.env and reused.
    """
    return Settings()
