"""
Document management routes supporting 15+ multi-format sources with strict per-user MongoDB isolation.
"""
import hashlib
import os
import uuid
import aiofiles
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Depends, Request, UploadFile, File, Form, HTTPException, status

from app.api.schemas.responses import DocumentResponse, DocumentListResponse
from app.auth.schemas import UserResponse
from app.auth.security import get_optional_user
from app.db.user_repository import UserWorkspaceRepository, get_user_vector_store
from app.rag.document_loader import load_document, SUPPORTED_LOCAL_FORMATS
from app.rag.chunking import split_documents
from app.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")
router = APIRouter(prefix="/api/v1/documents", tags=["Documents"])

# Ephemeral guest list used only when no authenticated user is signed in
DOCUMENTS: List[DocumentResponse] = []


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    privacy_mode: bool = Form(False),
    current_user: Optional[UserResponse] = Depends(get_optional_user),
):
    """Uploads and processes a document cleanly scoped to the authenticated user's private DB & vector store."""
    ext = ""
    if file.filename:
        ext = os.path.splitext(file.filename)[1].lower()

    if ext not in SUPPORTED_LOCAL_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Supported formats: {SUPPORTED_LOCAL_FORMATS}",
        )

    import tempfile
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}{ext}")
    doc_id = str(uuid.uuid4())
    try:
        async with aiofiles.open(temp_path, "wb") as out_file:
            content = await file.read()
            await out_file.write(content)

        size_bytes = len(content)
        file_sha = hashlib.sha256(content).hexdigest()
        docs = load_document(temp_path, privacy_mode=privacy_mode)
        clean_name = file.filename or os.path.basename(temp_path)
        for d in docs:
            d.metadata["filename"] = clean_name
            d.metadata["title"] = clean_name
            d.metadata["document_id"] = doc_id
            d.metadata["format"] = ext.lstrip(".")
            if current_user:
                d.metadata["user_id"] = current_user.id
                d.metadata["db_name"] = current_user.db_name

        chunks = split_documents(docs, chunk_size=1000, chunk_overlap=100)
        for c in chunks:
            c.metadata["filename"] = clean_name
            c.metadata["title"] = clean_name
            c.metadata["document_id"] = doc_id
            c.metadata["format"] = ext.lstrip(".")
            if current_user:
                c.metadata["user_id"] = current_user.id
                c.metadata["db_name"] = current_user.db_name

        if current_user:
            vector_store = get_user_vector_store(current_user.db_name)
            vector_store.add_documents(chunks)
            try:
                vector_store.save()
            except Exception as save_err:
                logger.warning(f"Could not persist user vector store immediately: {save_err}")

            preview = " ".join((c.page_content or "") for c in chunks[:2])[:500]
            saved = UserWorkspaceRepository(current_user.db_name, current_user.id).record_file_upload(
                document_id=doc_id,
                filename=clean_name,
                fmt=ext.replace(".", ""),
                chunk_count=len(chunks),
                size_bytes=size_bytes,
                sha256=file_sha,
                preview_text=preview,
                privacy_mode=privacy_mode,
            )
            response = DocumentResponse(
                document_id=doc_id,
                filename=clean_name,
                format=ext.replace(".", ""),
                chunk_count=len(chunks),
                size_bytes=size_bytes,
                qa_count=0,
                qa_links=[],
                status="processed",
                uploaded_at=datetime.fromisoformat(saved["uploaded_at"]),
            )
        else:
            vector_store = getattr(request.app.state, "vector_store", None)
            if vector_store:
                vector_store.add_documents(chunks)
                try:
                    vector_store.save()
                except Exception as save_err:
                    logger.warning(f"Could not persist vector store immediately: {save_err}")
            response = DocumentResponse(
                document_id=doc_id,
                filename=clean_name,
                format=ext.replace(".", ""),
                chunk_count=len(chunks),
                size_bytes=size_bytes,
                status="processed",
                uploaded_at=datetime.now(timezone.utc),
            )
            DOCUMENTS.append(response)

        try:
            from app.api.routes.research import _SYNTHESIS_CACHE, _SYNTHESIS_CACHE_LOCK
            with _SYNTHESIS_CACHE_LOCK:
                _SYNTHESIS_CACHE.clear()
        except Exception:
            pass

        logger.info(
            f"Indexed document '{file.filename}' ({len(chunks)} chunks, id={doc_id}, scope={current_user.db_name if current_user else 'guest'})"
        )
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
async def list_documents(
    request: Request,
    current_user: Optional[UserResponse] = Depends(get_optional_user),
):
    """Lists uploaded documents strictly isolated to the authenticated user (or guest sandbox when unauthenticated)."""
    if current_user:
        files = UserWorkspaceRepository(current_user.db_name, current_user.id).list_files()
        docs = [
            DocumentResponse(
                document_id=f["document_id"],
                filename=f["filename"],
                format=f.get("format", "document"),
                chunk_count=int(f.get("chunk_count", 0)),
                size_bytes=int(f.get("size_bytes", 0)),
                qa_count=int(f.get("qa_count", len(f.get("qa_links") or []))),
                qa_links=list(f.get("qa_links") or []),
                status=f.get("status", "processed"),
                uploaded_at=datetime.fromisoformat(f["uploaded_at"]),
            )
            for f in files
        ]
        return DocumentListResponse(documents=docs, total_count=len(docs))

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
                    uploaded_at=datetime.now(timezone.utc),
                )
            else:
                doc_map[d_id].chunk_count += 1

        merged_docs = list(doc_map.values())
        return DocumentListResponse(documents=merged_docs, total_count=len(merged_docs))
    return DocumentListResponse(documents=DOCUMENTS, total_count=len(DOCUMENTS))


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    request: Request,
    current_user: Optional[UserResponse] = Depends(get_optional_user),
):
    """Deletes a document strictly from the authenticated user's isolated MongoDB & vector store."""
    if current_user:
        ok = UserWorkspaceRepository(current_user.db_name, current_user.id).delete_file(document_id)
        if not ok:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found in your personal workspace.",
            )
        try:
            from app.api.routes.research import _SYNTHESIS_CACHE, _SYNTHESIS_CACHE_LOCK
            with _SYNTHESIS_CACHE_LOCK:
                _SYNTHESIS_CACHE.clear()
        except Exception:
            pass
        return {"message": f"Document {document_id} deleted from your private workspace."}

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
    try:
        from app.api.routes.research import _SYNTHESIS_CACHE, _SYNTHESIS_CACHE_LOCK
        with _SYNTHESIS_CACHE_LOCK:
            _SYNTHESIS_CACHE.clear()
    except Exception:
        pass
    DOCUMENTS = [d for d in DOCUMENTS if d.document_id != document_id and d.filename != document_id]
    return {"message": f"Document {document_id} deleted successfully"}
