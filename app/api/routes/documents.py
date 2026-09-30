"""
Document management routes supporting 15+ multi-format sources.
"""
import os
import uuid
import aiofiles
from datetime import datetime, timezone
from typing import List
from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from app.api.schemas.responses import DocumentResponse, DocumentListResponse
from app.rag.document_loader import load_document, SUPPORTED_LOCAL_FORMATS
from app.rag.chunking import split_documents
from app.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")
router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])

DOCUMENTS: List[DocumentResponse] = []

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    request: Request,
    file: UploadFile = File(...)
):
    """Uploads and processes a document across 15+ supported formats."""
    ext = ""
    if file.filename:
        ext = os.path.splitext(file.filename)[1].lower()
    
    if ext not in SUPPORTED_LOCAL_FORMATS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format '{ext}'. Supported formats: {SUPPORTED_LOCAL_FORMATS}"
        )

    import tempfile
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}{ext}")
    doc_id = str(uuid.uuid4())
    try:
        async with aiofiles.open(temp_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)

        docs = load_document(temp_path)
        for d in docs:
            d.metadata["filename"] = file.filename or os.path.basename(temp_path)
            d.metadata["title"] = file.filename or os.path.basename(temp_path)
            d.metadata["document_id"] = doc_id
            d.metadata["format"] = ext.lstrip(".")
        chunks = split_documents(docs, chunk_size=1000, chunk_overlap=100)
        for c in chunks:
            c.metadata["filename"] = file.filename or os.path.basename(temp_path)
            c.metadata["title"] = file.filename or os.path.basename(temp_path)
            c.metadata["document_id"] = doc_id
            c.metadata["format"] = ext.lstrip(".")
        
        vector_store = getattr(request.app.state, "vector_store", None)
        if vector_store:
            vector_store.add_documents(chunks)
            try:
                vector_store.save()
            except Exception as save_err:
                logger.warning(f"Could not persist vector store immediately: {save_err}")
            
        response = DocumentResponse(
            document_id=doc_id,
            filename=file.filename or "unknown",
            format=ext.replace(".", ""),
            chunk_count=len(chunks),
            status="processed",
            uploaded_at=datetime.now(timezone.utc)
        )
        DOCUMENTS.append(response)
        logger.info(f"Successfully processed and indexed document '{file.filename}' ({len(chunks)} chunks, id={doc_id})")
        return response
    except Exception as e:
        logger.error(f"Document processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

@router.get("/", response_model=DocumentListResponse)
async def list_documents(request: Request):
    """Lists all indexed documents from vector store and memory."""
    global DOCUMENTS
    vector_store = getattr(request.app.state, "vector_store", None)
    if vector_store:
        all_docs = vector_store.get_all_documents()
        doc_map = {}
        for d in all_docs:
            meta = d.metadata or {}
            d_id = meta.get("document_id") or meta.get("filename") or "indexed_doc"
            fname = meta.get("filename") or meta.get("title") or "document"
            fmt = str(meta.get("format") or os.path.splitext(fname)[1]).lstrip(".")
            if d_id not in doc_map:
                doc_map[d_id] = DocumentResponse(
                    document_id=d_id,
                    filename=fname,
                    format=fmt or "doc",
                    chunk_count=1,
                    status="processed",
                    uploaded_at=datetime.now(timezone.utc)
                )
            else:
                doc_map[d_id].chunk_count += 1
        
        merged_docs = list(doc_map.values())
        return DocumentListResponse(
            documents=merged_docs,
            total_count=len(merged_docs)
        )
    return DocumentListResponse(
        documents=DOCUMENTS,
        total_count=len(DOCUMENTS)
    )

@router.delete("/{document_id}")
async def delete_document(document_id: str, request: Request):
    """Deletes a document from the tracker and vector store."""
    global DOCUMENTS
    vector_store = getattr(request.app.state, "vector_store", None)
    if vector_store:
        deleted = vector_store.delete_by_document_id(document_id)
        if deleted == 0:
            ids_to_del = [key for key, doc in vector_store._documents.items() if doc.metadata.get("filename") == document_id]
            vector_store.delete_documents(ids_to_del)
        try:
            vector_store.save()
        except Exception as e:
            logger.warning(f"Could not persist vector store after deletion: {e}")
    DOCUMENTS = [d for d in DOCUMENTS if d.document_id != document_id and d.filename != document_id]
    return {"message": f"Document {document_id} deleted successfully"}
