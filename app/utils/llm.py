"""
Shared LLM factory.

Builds a ChatOpenAI client only when a real API key is configured, so agents
can skip network calls entirely (and fall back to deterministic logic) instead
of waiting on doomed requests.
"""

from typing import Optional

from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")


def build_chat_llm(temperature: float = 0.2) -> Optional[ChatOpenAI]:
    """
    Build a ChatOpenAI client, or return None when no usable key is configured.

    Args:
        temperature: Sampling temperature for the model.

    Returns:
        A configured ChatOpenAI instance, or None if the LLM is unavailable.
    """
    settings = get_settings()
    if not settings.llm_available:
        return None

    kwargs: dict = {
        "model": settings.OPENAI_MODEL_NAME,
        "api_key": settings.OPENAI_API_KEY,
        "temperature": temperature,
        "timeout": settings.LLM_TIMEOUT_S,
        "max_retries": settings.LLM_MAX_RETRIES,
    }
    if settings.OPENAI_BASE_URL:
        kwargs["base_url"] = settings.OPENAI_BASE_URL
    return ChatOpenAI(**kwargs)


def extract_json_block(content: str) -> str:
    """
    Strip Markdown code fences from an LLM response so it can be JSON-parsed.

    Args:
        content: Raw LLM response text.

    Returns:
        The content with any ```json ... ``` fences removed.
    """
    text = content.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()
