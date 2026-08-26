"""
Text splitting and chunking strategies.
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
    
    Args:
        chunk_size (Optional[int]): Maximum size of chunks to return. Defaults to settings.CHUNK_SIZE.
        chunk_overlap (Optional[int]): Overlap in characters between chunks. Defaults to settings.CHUNK_OVERLAP.
        
    Returns:
        RecursiveCharacterTextSplitter: The configured text splitter instance.
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
    
    logger.debug(f"Initialized RecursiveCharacterTextSplitter (chunk_size={c_size}, chunk_overlap={c_overlap})")
    return splitter

def split_documents(
    documents: List[Document], 
    chunk_size: Optional[int] = None, 
    chunk_overlap: Optional[int] = None
) -> List[Document]:
    """
    Splits a list of Documents into smaller chunked Documents.
    
    Args:
        documents (List[Document]): The documents to split.
        chunk_size (Optional[int]): Maximum size of chunks. Defaults to settings.CHUNK_SIZE.
        chunk_overlap (Optional[int]): Overlap between chunks. Defaults to settings.CHUNK_OVERLAP.
        
    Returns:
        List[Document]: The list of chunked Document objects.
    """
    if not documents:
        logger.warning("No documents provided to split.")
        return []
        
    logger.info(f"Splitting {len(documents)} documents...")
    
    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    
    num_chunks = len(chunks)
    if num_chunks > 0:
        avg_size = sum(len(chunk.page_content) for chunk in chunks) / num_chunks
        logger.info(f"Split completed: Generated {num_chunks} chunks with an average size of {avg_size:.2f} characters.")
    else:
        logger.warning("Splitting resulted in 0 chunks.")
        
    return chunks
