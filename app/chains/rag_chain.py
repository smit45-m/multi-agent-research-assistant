"""
Core RAG chain integrating retrieval and generation using LCEL.
"""
from typing import Optional, Any, Dict

from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.utils.logger import setup_logger
from app.rag.vector_store import VectorStoreManager
from app.chains.prompts import get_prompt_template

logger = setup_logger(__name__, "INFO")

def format_docs(docs: list[Document]) -> str:
    """
    Formats a list of documents into a single string with source metadata.
    
    Args:
        docs (list[Document]): List of retrieved documents.
        
    Returns:
        str: Formatted context string.
    """
    formatted_chunks = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source_path", doc.metadata.get("source", f"Doc {i+1}"))
        content = doc.page_content.strip()
        formatted_chunks.append(f"--- SOURCE: {source} ---\n{content}\n")
    return "\n".join(formatted_chunks)


def create_rag_chain(
    vector_store_manager: VectorStoreManager, 
    model_name: Optional[str] = None
) -> Runnable:
    """
    Creates a Retrieval-Augmented Generation (RAG) chain using LCEL.
    The chain takes a 'query' as input, retrieves relevant documents, and generates an answer.
    
    Args:
        vector_store_manager (VectorStoreManager): Initialized vector store manager.
        model_name (Optional[str]): Override for the LLM model name. Defaults to settings.
        
    Returns:
        Runnable: An LCEL Runnable chain.
    """
    settings = get_settings()
    llm_model = model_name or settings.OPENAI_MODEL_NAME
    
    logger.info(f"Creating RAG chain with LLM model: {llm_model}")
    
    # Initialize the LLM
    kwargs = {
        "model_name": llm_model,
        "api_key": settings.OPENAI_API_KEY,
        "temperature": 0.2
    }
    if settings.OPENAI_BASE_URL:
        kwargs["base_url"] = settings.OPENAI_BASE_URL
    llm = ChatOpenAI(**kwargs)
    
    # Check if vector store is initialized
    if vector_store_manager.vector_store is None:
        logger.warning("Vector store is not initialized. Chain may fail on execution.")
        retriever = lambda q: []
    else:
        retriever = vector_store_manager.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
    
    prompt = get_prompt_template("writer")
    
    rag_chain = (
        {"context": retriever | format_docs, "query": RunnablePassthrough()}
        | (lambda x: {"query": x["query"], "analysis": x["context"]}) 
        | prompt
        | llm
        | StrOutputParser()
    )
    
    logger.info("RAG chain successfully assembled using LCEL.")
    return rag_chain
