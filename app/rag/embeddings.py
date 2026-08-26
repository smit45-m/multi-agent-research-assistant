"""
Embedding model configuration and helper functions.
"""
from typing import List, Any
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from langchain_openai import OpenAIEmbeddings

from app.config import get_settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")

def get_embedding_model() -> Any:
    """
    Initializes and returns the embedding model configured via settings.
    Supports both OpenAIEmbeddings and HuggingFace/local embeddings.
    
    Returns:
        Configured embedding model instance.
    """
    settings = get_settings()
    model_name = settings.EMBEDDING_MODEL
    
    if model_name.startswith("all-") or "sentence-transformers" in model_name or "bge-" in model_name:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name=model_name)
            logger.info(f"Initialized HuggingFace embeddings with model: {model_name}")
            return embeddings
        except Exception as e:
            logger.warning(f"Failed to load HuggingFaceEmbeddings ({e}), falling back to OpenAIEmbeddings")

    embeddings = OpenAIEmbeddings(
        openai_api_key=settings.OPENAI_API_KEY,
        model=model_name
    )
    logger.debug(f"Initialized OpenAI embeddings with model: {model_name}")
    return embeddings


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
def batch_embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embeds a list of texts into vectors using the configured embedding model,
    with exponential backoff retry logic.
    
    Args:
        texts (List[str]): A list of string texts to embed.
        
    Returns:
        List[List[float]]: A list of embedding vectors corresponding to the input texts.
        
    Raises:
        Exception: Reraises exceptions if all retries fail.
    """
    logger.info(f"Embedding batch of {len(texts)} texts...")
    try:
        embeddings_model = get_embedding_model()
        vectors = embeddings_model.embed_documents(texts)
        logger.info(f"Successfully embedded {len(texts)} texts.")
        return vectors
    except Exception as e:
        logger.error(f"Error during batch embedding: {str(e)}")
        raise
