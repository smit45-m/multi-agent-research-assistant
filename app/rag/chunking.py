"""
Text splitting and adaptive multi-scale chunking strategies.
Supports:
- Document-size adaptive chunking (Sentence-window, Semantic recursive, Parent-Child)
- Recursive character splitting
"""
from typing import List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")

def get_text_splitter(
    chunk_size: Optional[int] = None, 
    chunk_overlap: Optional[int] = None
) -> RecursiveCharacterTextSplitter:
    """
    Initializes and returns a RecursiveCharacterTextSplitter with configured sizes and separators.
    """
    settings = get_settings()
    
    c_size = chunk_size if chunk_size is not None else settings.CHUNK_SIZE
    c_overlap = chunk_overlap if chunk_overlap is not None else settings.CHUNK_OVERLAP
    
    separators = [
        "\n\n",
        "\n",
        " ",
        ".",
        ",",
        "\u200b",
        "\uff0c",
        "\u3001",
        "\uff0e",
        "\u3002",
        "",
    ]
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=c_size,
        chunk_overlap=c_overlap,
        separators=separators,
        length_function=len,
        is_separator_regex=False
    )
    return splitter

def split_documents(
    documents: List[Document], 
    chunk_size: Optional[int] = None, 
    chunk_overlap: Optional[int] = None,
    adaptive: bool = True
) -> List[Document]:
    """
    Splits a list of Documents into smaller chunked Documents.
    If adaptive=True and chunk_size is not forced, applies Multi-Scale Adaptive Chunking:
    - Small (<2KB): Sentence-window chunks
    - Medium (2KB-20KB): Semantic recursive chunks
    - Large (>20KB): Hierarchical Parent-Child chunks
    """
    if not documents:
        return []

    if chunk_size is not None or not adaptive:
        splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_documents(documents)

    from app.rag.advanced_rag import AdaptiveChunker
    all_chunks: List[Document] = []

    for doc in documents:
        parents, children = AdaptiveChunker.chunk_document(doc)
        # Store child chunks (plus parent reference) for indexing
        all_chunks.extend(children)

    if all_chunks:
        logger.info(f"Adaptive multi-scale splitting generated {len(all_chunks)} chunks across {len(documents)} documents.")
    return all_chunks or documents
