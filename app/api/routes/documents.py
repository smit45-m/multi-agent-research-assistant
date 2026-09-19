"""
Document management routes supporting 15+ multi-format sources.
"""

import os
import uuid
from datetime import datetime, timezone
from typing import List

import aiofiles
from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.api.schemas.responses import DocumentListResponse, DocumentResponse
from app.rag.chunking import split_documents
from app.rag.document_loader import SUPPORTED_LOCAL_FORMATS, load_document
from app.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")
router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])

DOCUMENTS: List[DocumentResponse] = []


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    request: Request, file: UploadFile = File(...)
) -> DocumentResponse:
    """Uploads and processes a document across 15+ supported formats."""
    ext = ""
    if file.filename:
        ext = os.path.splitext(file.filename)[1].lower()

    if ext not in SUPPORTED_LOCAL_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file format '{ext}'. "
                f"Supported formats: {SUPPORTED_LOCAL_FORMATS}"
            ),
        )

    import tempfile

    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}{ext}")
    try:
        async with aiofiles.open(temp_path, "wb") as out_file:
            content = await file.read()
            await out_file.write(content)

        docs = load_document(temp_path)
        chunks = split_documents(docs, chunk_size=1000, chunk_overlap=100)

        vector_store = getattr(request.app.state, "vector_store", None)
        if vector_store:
            vector_store.add_documents(chunks)

        doc_id = str(uuid.uuid4())
        response = DocumentResponse(
            document_id=doc_id,
            filename=file.filename or "unknown",
            format=ext.replace(".", ""),
            chunk_count=len(chunks),
            status="processed",
            uploaded_at=datetime.now(timezone.utc),
        )
        DOCUMENTS.append(response)
        logger.info(
            f"Successfully processed and indexed document "
            f"'{file.filename}' ({len(chunks)} chunks)"
        )
        return response
    except Exception as e:
        logger.error(f"Document processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error processing document: {str(e)}"
        )
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@router.get("/", response_model=DocumentListResponse)
async def list_documents() -> DocumentListResponse:
    """Lists all indexed documents."""
    return DocumentListResponse(documents=DOCUMENTS, total_count=len(DOCUMENTS))


@router.delete("/{document_id}")
async def delete_document(document_id: str) -> dict:
    """Deletes a document from the tracker."""
    global DOCUMENTS
    doc_to_delete = next((d for d in DOCUMENTS if d.document_id == document_id), None)
    if not doc_to_delete:
        raise HTTPException(status_code=404, detail="Document not found")

    DOCUMENTS = [d for d in DOCUMENTS if d.document_id != document_id]
    return {"message": f"Document {document_id} deleted successfully"}
