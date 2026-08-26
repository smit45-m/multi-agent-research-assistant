"""
Multi-format document loading utilities.
"""
import datetime
from pathlib import Path
from typing import List, Union, Dict, Type

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    JSONLoader,
    BSHTMLLoader,
    UnstructuredMarkdownLoader,
    WebBaseLoader
)

from app.utils.logger import setup_logger
from app.utils.exceptions import DocumentProcessingError

logger = setup_logger(__name__, "INFO")

SUPPORTED_FORMATS: Dict[str, Type] = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
    ".json": JSONLoader,
    ".html": BSHTMLLoader,
    ".md": UnstructuredMarkdownLoader,
}

def _add_metadata(documents: List[Document], source: str, format_type: str) -> List[Document]:
    """
    Enriches the metadata of loaded documents.
    """
    load_timestamp = datetime.datetime.now().isoformat()
    for doc in documents:
        doc.metadata["source_path"] = source
        doc.metadata["format"] = format_type
        doc.metadata["load_timestamp"] = load_timestamp
    return documents

def load_document(file_path: Union[str, Path]) -> List[Document]:
    """
    Auto-detects format from file extension and loads a single document.
    
    Args:
        file_path (Union[str, Path]): Path to the file to load.
        
    Returns:
        List[Document]: Loaded document(s).
        
    Raises:
        DocumentProcessingError: If the file format is not supported or loading fails.
    """
    file_path_obj = Path(file_path)
    if not file_path_obj.exists() or not file_path_obj.is_file():
        err_msg = f"File not found or is not a valid file: {file_path}"
        logger.error(err_msg)
        raise DocumentProcessingError(err_msg)
        
    extension = file_path_obj.suffix.lower()
    
    if extension not in SUPPORTED_FORMATS:
        err_msg = f"Unsupported file format: {extension}. Supported formats: {list(SUPPORTED_FORMATS.keys())}"
        logger.error(err_msg)
        raise DocumentProcessingError(err_msg)
        
    loader_cls = SUPPORTED_FORMATS[extension]
    logger.info(f"Loading {file_path} using {loader_cls.__name__}...")
    
    try:
        if loader_cls == JSONLoader:
            loader = loader_cls(str(file_path), jq_schema=".", text_content=False)
        else:
            loader = loader_cls(str(file_path))
            
        documents = loader.load()
        logger.info(f"Successfully loaded {len(documents)} document pages/sections from {file_path}.")
        
        return _add_metadata(documents, str(file_path), extension)
        
    except Exception as e:
        err_msg = f"Failed to load document {file_path}: {str(e)}"
        logger.error(err_msg)
        raise DocumentProcessingError(err_msg) from e

def load_from_url(url: str) -> List[Document]:
    """
    Fetches and parses a web page from a URL.
    """
    logger.info(f"Loading content from URL: {url}...")
    try:
        loader = WebBaseLoader(url)
        documents = loader.load()
        logger.info(f"Successfully loaded {len(documents)} document(s) from {url}.")
        return _add_metadata(documents, url, "url")
    except Exception as e:
        err_msg = f"Failed to load URL {url}: {str(e)}"
        logger.error(err_msg)
        raise DocumentProcessingError(err_msg) from e

def load_directory(dir_path: Union[str, Path], glob: str = "*") -> List[Document]:
    """
    Loads all supported documents from a directory based on a glob pattern.
    """
    dir_path_obj = Path(dir_path)
    if not dir_path_obj.exists() or not dir_path_obj.is_dir():
        err_msg = f"Directory not found or is not a valid directory: {dir_path}"
        logger.error(err_msg)
        raise DocumentProcessingError(err_msg)
        
    logger.info(f"Scanning directory {dir_path} with glob '{glob}'...")
    all_documents = []
    failed_files = []
    
    for file_path in dir_path_obj.rglob(glob):
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_FORMATS:
            try:
                docs = load_document(file_path)
                all_documents.extend(docs)
            except DocumentProcessingError as e:
                failed_files.append(str(file_path))
                logger.warning(f"Skipping file due to error: {e}")
                
    logger.info(f"Loaded a total of {len(all_documents)} document parts from directory {dir_path}.")
    if failed_files:
        logger.warning(f"Failed to load {len(failed_files)} files in directory scan.")
        
    return all_documents
