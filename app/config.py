"""Configuration anchored to this project, never to the server's working directory."""
from functools import lru_cache
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.groq.com/openai/v1"
    OPENAI_MODEL_NAME: str = "llama-3.3-70b-versatile"
    FAST_MODEL_NAME: Optional[str] = None
    RESEARCH_MODEL_NAME: Optional[str] = None
    LLM_TIMEOUT_SECONDS: float = Field(default=35, ge=1, le=120)
    FAST_BUDGET_SECONDS: float = Field(default=25, ge=3, le=120)
    BALANCED_BUDGET_SECONDS: float = Field(default=75, ge=5, le=240)
    RESEARCH_BUDGET_SECONDS: float = Field(default=180, ge=10, le=600)
    WEB_SEARCH_ENABLED: bool = True
    SEARCH_API_KEY: Optional[str] = None
    SEARCH_TIMEOUT_SECONDS: float = 7
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_LOCAL_ONLY: bool = True
    VECTOR_STORE_PATH: str = str(PROJECT_ROOT / "data" / "vector_store")
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    LOW_LATENCY_CHUNK_SIZE: int = 450
    LOW_LATENCY_CHUNK_OVERLAP: int = 60
    RESEARCHED_CHUNK_SIZE: int = 1200
    RESEARCHED_CHUNK_OVERLAP: int = 250
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    MAX_CONCURRENT_REQUESTS: int = Field(default=50, ge=1, le=128)
    API_KEY: Optional[str] = None
    VISION_MODEL_NAME: Optional[str] = None
    AUDIO_MODEL_NAME: Optional[str] = None
    MEDIA_API_KEY: Optional[str] = None
    MEDIA_BASE_URL: Optional[str] = None
    MEDIA_MAX_MB: int = Field(default=20, ge=1, le=25)
    MEDIA_TTL_SECONDS: int = 3600
    ENABLE_LIVE_BENCHMARK_API: bool = False
    RETRIEVAL_MIN_RELEVANCE: float = Field(default=0.18, ge=0, le=1)

    GEMINI_API_KEY: str = ""
    GEMINI_MODEL_NAME: str = "gemini-flash-lite-latest"
    GEMINI_FAST_MODEL: str = "gemini-flash-lite-latest"
    GEMINI_RESEARCH_MODEL: str = "gemini-3.5-flash"

    # TypeSafe AI Jev & autotrust/JEV-27B (System One AI Decision Model)
    # Hugging Face: https://huggingface.co/autotrust/JEV-27B
    TYPESAFE_API_KEY: Optional[str] = None
    TYPESAFE_API_URL: str = "https://api.typesafe.ai/v1/systemone"
    HF_TOKEN: Optional[str] = None
    HUGGINGFACE_API_KEY: Optional[str] = None
    JEV_MODEL_ID: str = "autotrust/JEV-27B"
    JEV_ENDPOINT_URL: Optional[str] = None
    JEV_ENABLED: bool = True
    FAST_MODE_MAX_OUTPUT_TOKENS: int = 280
    FAST_MODE_SEARCH_TIMEOUT_SECONDS: float = 0.8

    # Local Open-Source LLM & Privacy Configuration (Air-Gapped)
    LOCAL_LLM_URL: str = "http://127.0.0.1:11434"
    LOCAL_LLM_MODEL: str = "llama3.2"
    LOCAL_OPENAI_URL: str = "http://127.0.0.1:1234/v1"
    PRIVACY_BUDGET_SECONDS: float = Field(default=30, ge=3, le=120)

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    @property
    def gemini_configured(self) -> bool:
        return bool(self.GEMINI_API_KEY and not self.GEMINI_API_KEY.startswith(("your-", "changeme")))

    @property
    def llm_configured(self) -> bool:
        return self.gemini_configured or bool(self.OPENAI_API_KEY and not self.OPENAI_API_KEY.startswith(("your-", "changeme")))

    @property
    def store_path(self) -> Path:
        path = Path(self.VECTOR_STORE_PATH)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def model_for(self, mode: str) -> str:
        """Returns the optimal model for low-latency (<5s) vs researched depth vs privacy."""
        if mode == "privacy":
            return self.LOCAL_LLM_MODEL
        if self.gemini_configured:
            if mode in ("fast", "quick"):
                return self.GEMINI_FAST_MODEL
            if mode in ("research", "deep"):
                return self.GEMINI_RESEARCH_MODEL
            return self.GEMINI_MODEL_NAME
        # Groq / OpenAI provider selection
        if mode in ("fast", "quick"):
            return self.FAST_MODEL_NAME or "llama-3.1-8b-instant"
        if mode in ("research", "deep"):
            return self.RESEARCH_MODEL_NAME or "llama-3.3-70b-versatile"
        return self.OPENAI_MODEL_NAME

    def chunk_config_for(self, mode: str) -> tuple[int, int]:
        """Returns (chunk_size, chunk_overlap) optimized for the selected mode."""
        if mode in ("fast", "quick"):
            return self.LOW_LATENCY_CHUNK_SIZE, self.LOW_LATENCY_CHUNK_OVERLAP
        return self.RESEARCHED_CHUNK_SIZE, self.RESEARCHED_CHUNK_OVERLAP


@lru_cache()
def get_settings() -> Settings:
    return Settings()
