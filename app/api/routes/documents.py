import os
import uuid
import aiofiles
from datetime import datetime
from typing import List
from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from app.api.schemas.responses import DocumentResponse, DocumentListResponse
from app.rag.document_loader import load_document, SUPPORTED_FORMATS
from app.rag.chunking import split_documents

router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])

DOCUMENTS: List[DocumentResponse] = []

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    request: Request,
    file: UploadFile = File(...)
):
    """Uploads and processes a document."""
    ext = ""
    if file.filename:
        ext = os.path.splitext(file.filename)[1].lower()
    
    if ext not in SUPPORTED_FORMATS:
        raise HTTPException(status_code=400, detail=f"Unsupported file format. Supported formats: {SUPPORTED_FORMATS}")

    import tempfile
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}{ext}")
    try:
        async with aiofiles.open(temp_path, 'wb') as out_file:
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
            chunk_count=len(chunks),
            status="processed",
            uploaded_at=datetime.utcnow()
        )
        DOCUMENTS.append(response)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@router.get("/", response_model=DocumentListResponse)
async def list_documents():
    """Lists all uploaded documents."""
    return DocumentListResponse(
        documents=DOCUMENTS,
        total_count=len(DOCUMENTS)
    )

@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """Deletes a document from the tracker."""
    global DOCUMENTS
    doc_to_delete = next((d for d in DOCUMENTS if d.document_id == document_id), None)
    if not doc_to_delete:
        raise HTTPException(status_code=404, detail="Document not found")
        
    DOCUMENTS = [d for d in DOCUMENTS if d.document_id != document_id]
    
    return {"message": f"Document {document_id} deleted successfully"}
