"""
Embedding model configuration and helper functions.

Fallback chain:
1. Local HuggingFace embeddings (langchain-huggingface, needs
   sentence-transformers installed — see requirements-embeddings.txt).
2. OpenAI-compatible embeddings (needs a real API key).
3. Deterministic offline hashing embeddings (keyword-level similarity,
   no dependencies) so the system stays functional everywhere.
"""

import hashlib
import math
import re
from typing import Any, List

from langchain_core.embeddings import Embeddings
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import get_settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")

_WORD_RE = re.compile(r"\b\w+\b")


class HashingEmbeddings(Embeddings):
    """
    Deterministic offline embedding model using feature hashing.

    Provides keyword-level similarity (comparable to sparse retrieval) when
    neither local nor API embedding models are available. Fully offline and
    reproducible.
    """

    def __init__(self, dimensions: int = 512):
        self.dimensions = dimensions

    def _embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dimensions
        for token in _WORD_RE.findall(text.lower()):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


def _try_huggingface(model_name: str) -> Any:
    """Try the maintained langchain-huggingface package, then legacy."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=model_name)
    except ImportError:
        from langchain_community.embeddings import (
            HuggingFaceEmbeddings as LegacyHFE,
        )

        return LegacyHFE(model_name=model_name)


def get_embedding_model() -> Any:
    """
    Initialize and return the best available embedding model.

    Returns:
        Configured embedding model instance (never raises).
    """
    settings = get_settings()
    model_name = settings.EMBEDDING_MODEL

    is_local_model = (
        model_name.startswith("all-")
        or "sentence-transformers" in model_name
        or "bge-" in model_name
    )

    if is_local_model:
        try:
            embeddings = _try_huggingface(model_name)
            logger.info("Initialized HuggingFace embeddings: %s", model_name)
            return embeddings
        except Exception as exc:  # noqa: BLE001 - continue down the chain
            logger.warning(
                "HuggingFace embeddings unavailable (%s); trying next option.",
                exc,
            )

    if settings.llm_available and not is_local_model:
        try:
            from langchain_openai import OpenAIEmbeddings
            from pydantic import SecretStr

            embeddings = OpenAIEmbeddings(
                api_key=SecretStr(settings.OPENAI_API_KEY), model=model_name
            )
            logger.info("Initialized OpenAI embeddings: %s", model_name)
            return embeddings
        except Exception as exc:  # noqa: BLE001
            logger.warning("OpenAI embeddings unavailable (%s).", exc)

    logger.info(
        "Using offline hashing embeddings (keyword-level similarity). "
        "Install requirements-embeddings.txt for semantic embeddings."
    )
    return HashingEmbeddings()


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
def batch_embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of texts using the configured embedding model,
    with exponential backoff retry logic.

    Args:
        texts: A list of texts to embed.

    Returns:
        A list of embedding vectors.
    """
    model = get_embedding_model()
    vectors: List[List[float]] = model.embed_documents(texts)
    return vectors
