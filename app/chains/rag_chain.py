"""
Core RAG chain integrating retrieval and generation using LCEL.
"""

from typing import Any, List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnablePassthrough,
)

from app.chains.prompts import get_prompt_template
from app.rag.vector_store import VectorStoreManager
from app.utils.llm import build_chat_llm
from app.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")


def format_docs(docs: List[Document]) -> str:
    """
    Formats a list of documents into a single string with source metadata.

    Args:
        docs: List of retrieved documents.

    Returns:
        Formatted context string.
    """
    formatted_chunks = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get(
            "source_path", doc.metadata.get("source", f"Doc {i + 1}")
        )
        content = doc.page_content.strip()
        formatted_chunks.append(f"--- SOURCE: {source} ---\n{content}\n")
    return "\n".join(formatted_chunks)


def create_rag_chain(
    vector_store_manager: VectorStoreManager,
    model_name: Optional[str] = None,
) -> Runnable:
    """
    Creates a Retrieval-Augmented Generation (RAG) chain using LCEL.
    The chain takes a 'query' as input, retrieves relevant documents,
    and generates an answer.

    Args:
        vector_store_manager: Initialized vector store manager.
        model_name: Unused override kept for API compatibility; the model
            comes from settings via the shared LLM factory.

    Returns:
        An LCEL Runnable chain.

    Raises:
        RuntimeError: If no LLM is configured.
    """
    llm = build_chat_llm(temperature=0.2)
    if llm is None:
        raise RuntimeError(
            "create_rag_chain requires a configured LLM API key (set OPENAI_API_KEY)."
        )

    logger.info("Creating RAG chain with configured LLM")

    retriever: Runnable
    if vector_store_manager.vector_store is None:
        logger.warning("Vector store is not initialized. Chain will retrieve nothing.")
        retriever = RunnableLambda(lambda _q: [])
    else:
        retriever = vector_store_manager.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        )

    prompt = get_prompt_template("writer")

    def _remap(x: Any) -> dict:
        return {"query": x["query"], "analysis": x["context"]}

    rag_chain: Runnable = (
        {"context": retriever | format_docs, "query": RunnablePassthrough()}
        | RunnableLambda(_remap)
        | prompt
        | llm
        | StrOutputParser()
    )

    logger.info("RAG chain successfully assembled using LCEL.")
    return rag_chain
