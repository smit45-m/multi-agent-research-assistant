"""
Multimodal analysis routes — the Interactive Analyst agent's API surface.

Accepts live camera photos, microphone recordings (WAV), tabular datasets,
and text documents; measures them with app.tools.media_analyzer and returns
an interactive emoji/table Markdown briefing plus the raw measurements.
"""

import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.agents.interactive_analyst import InteractiveAnalystAgent
from app.tools.media_analyzer import (
    analyze_audio_bytes,
    analyze_image_bytes,
    analyze_tabular_bytes,
    analyze_text,
)
from app.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")
router = APIRouter(prefix="/api/v1/media", tags=["Multimodal Analysis"])

_ANALYST = InteractiveAnalystAgent()

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
AUDIO_EXTS = {".wav"}
TABULAR_EXTS = {".csv", ".tsv", ".xlsx", ".json"}
TEXT_EXTS = {
    ".txt",
    ".md",
    ".py",
    ".js",
    ".ts",
    ".html",
    ".htm",
    ".yaml",
    ".xml",
    ".sh",
}


async def _read_upload(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.",
        )
    return data


def _briefing_or_422(measurements: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return _ANALYST.render(measurements)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/analyze")
async def analyze_media(
    request: Request, file: UploadFile = File(...)
) -> Dict[str, Any]:
    """
    Analyze any supported media file and return an interactive briefing.

    Routes by extension/content-type:
    images → pixel analysis; WAV audio → PCM analysis; CSV/TSV/XLSX/JSON →
    dataset profiling; text/code/markdown → document analytics.
    """
    filename = file.filename or "upload"
    ext = os.path.splitext(filename)[1].lower()
    ctype = (file.content_type or "").lower()
    data = await _read_upload(file)

    try:
        if ext in IMAGE_EXTS or ctype.startswith("image/"):
            measurements = analyze_image_bytes(data, filename)
        elif ext in AUDIO_EXTS or ctype in ("audio/wav", "audio/x-wav", "audio/wave"):
            measurements = analyze_audio_bytes(data, filename)
        elif ext in TABULAR_EXTS:
            measurements = analyze_tabular_bytes(data, filename, ext)
        elif ext in TEXT_EXTS or ctype.startswith("text/"):
            measurements = analyze_text(
                data.decode("utf-8", errors="replace"), filename
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported media type '{ext or ctype}'. Supported: "
                    "images (png/jpg/webp/gif/bmp), WAV audio, "
                    "CSV/TSV/XLSX/JSON datasets, and text/markdown/code files."
                ),
            )
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - map parse failures to 422
        logger.warning("Media analysis failed for %s: %s", filename, exc)
        raise HTTPException(
            status_code=422, detail=f"Could not analyze '{filename}': {exc}"
        ) from exc

    return _briefing_or_422(measurements)


@router.post("/capture/photo")
async def analyze_live_photo(
    request: Request, file: UploadFile = File(...)
) -> Dict[str, Any]:
    """Analyze a live camera capture (browser getUserMedia snapshot)."""
    data = await _read_upload(file)
    try:
        measurements = analyze_image_bytes(data, file.filename or "live-photo.png")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=422, detail=f"Could not decode the captured photo: {exc}"
        ) from exc
    return _briefing_or_422(measurements)


@router.post("/capture/audio")
async def analyze_live_audio(
    request: Request, file: UploadFile = File(...)
) -> Dict[str, Any]:
    """Analyze a live microphone recording (WAV PCM from the browser)."""
    data = await _read_upload(file)
    try:
        measurements = analyze_audio_bytes(data, file.filename or "live-audio.wav")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=422,
            detail=(
                "Could not decode the recording as WAV PCM: "
                f"{exc}. The recorder must submit 16-bit WAV."
            ),
        ) from exc
    return _briefing_or_422(measurements)


@router.post("/analyze/index")
async def analyze_and_index(
    request: Request, file: UploadFile = File(...)
) -> Dict[str, Any]:
    """
    Analyze a text/tabular file AND index its content into the knowledge
    base so follow-up research queries can be grounded in it.
    """
    filename = file.filename or "upload"
    ext = os.path.splitext(filename)[1].lower()
    data = await _read_upload(file)

    briefing: Optional[Dict[str, Any]] = None
    text_content: Optional[str] = None

    if ext in TABULAR_EXTS:
        measurements = analyze_tabular_bytes(data, filename, ext)
        briefing = _briefing_or_422(measurements)
        text_content = (
            data.decode("utf-8", errors="replace") if ext != ".xlsx" else None
        )
    elif ext in TEXT_EXTS:
        text_content = data.decode("utf-8", errors="replace")
        briefing = _briefing_or_422(analyze_text(text_content, filename))
    else:
        raise HTTPException(
            status_code=400,
            detail="Only text and tabular files can be indexed via this endpoint.",
        )

    chunks_indexed = 0
    vector_store = getattr(request.app.state, "vector_store", None)
    if vector_store is not None and text_content:
        from langchain_core.documents import Document

        from app.rag.chunking import split_documents

        docs = [Document(page_content=text_content, metadata={"source": filename})]
        chunks = split_documents(docs, chunk_size=1000, chunk_overlap=100)
        vector_store.add_documents(chunks)
        chunks_indexed = len(chunks)

    assert briefing is not None
    briefing["chunks_indexed"] = chunks_indexed
    return briefing
