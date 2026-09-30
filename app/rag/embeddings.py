"""Optional local semantic embeddings. Never send local-model names to an unrelated API."""
from functools import lru_cache
from langchain_core.embeddings import Embeddings
from app.config import get_settings


class LocalEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()

    def embed_query(self, text):
        return self.model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0].tolist()


@lru_cache(maxsize=1)
def get_embedding_model():
    settings = get_settings()
    if settings.EMBEDDING_MODEL == "disabled":
        raise RuntimeError("Dense embeddings are disabled; BM25 remains available.")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(settings.EMBEDDING_MODEL,
                                    local_files_only=settings.EMBEDDING_LOCAL_ONLY,
                                    trust_remote_code=False, device="cpu")
        return LocalEmbeddings(model)
    except Exception:
        raise RuntimeError("Local semantic model is not installed or cached. BM25 remains available; install sentence-transformers and cache EMBEDDING_MODEL to enable dense retrieval.") from None


def batch_embed_texts(texts):
    return get_embedding_model().embed_documents(texts)
