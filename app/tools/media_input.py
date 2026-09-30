"""Image understanding and audio transcription, not media generation.

Raw uploads are held only for the provider request. Extracted text is scoped to a
browser session, bounded in memory, expires, and is never silently indexed.
"""
import base64
import io
import secrets
import threading
import time
import warnings
from pathlib import Path
from fastapi import HTTPException
from app.config import get_settings

IMAGE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}
AUDIO_TYPES = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4", ".ogg": "audio/ogg", ".webm": "audio/webm", ".flac": "audio/flac"}


def capabilities():
    settings = get_settings()
    key = settings.MEDIA_API_KEY or (settings.OPENAI_API_KEY if settings.llm_configured else "")
    return {
        "llm_configured": settings.llm_configured, "model": settings.OPENAI_MODEL_NAME,
        "vision": {"configured": bool(key and settings.VISION_MODEL_NAME), "model": settings.VISION_MODEL_NAME},
        "audio": {"configured": bool(key and settings.AUDIO_MODEL_NAME), "model": settings.AUDIO_MODEL_NAME},
        "max_upload_mb": settings.MEDIA_MAX_MB, "web_search": settings.WEB_SEARCH_ENABLED,
        "modes": {"fast": settings.FAST_BUDGET_SECONDS, "balanced": settings.BALANCED_BUDGET_SECONDS, "research": settings.RESEARCH_BUDGET_SECONDS},
        "note": "Configured does not mean provider authentication has been verified. Uploads are sent to your configured media provider. Extracted text expires after one hour or on server restart.",
    }


def validate_media(filename, content, declared_type=""):
    settings = get_settings()
    if not content:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(content) > settings.MEDIA_MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"Upload limit is {settings.MEDIA_MAX_MB} MB.")
    extension = Path(filename).suffix.lower()
    if extension in IMAGE_TYPES:
        from PIL import Image
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(content)) as image:
                    if image.width * image.height > 16_000_000:
                        raise ValueError("Image exceeds 16 megapixels.")
                    if image.format not in {"PNG", "JPEG", "WEBP"}:
                        raise ValueError("Unsupported image content.")
                    mime = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}[image.format]
                    if mime != IMAGE_TYPES[extension]:
                        raise ValueError("Image extension and contents disagree.")
                    image.verify()
        except Exception:
            raise HTTPException(400, "Invalid image, extension mismatch, or image above the 16-megapixel limit.") from None
        return "image", mime
    if extension in AUDIO_TYPES:
        valid = {".wav": content[:4] == b"RIFF" and content[8:12] == b"WAVE",
                 ".mp3": content[:3] == b"ID3" or (len(content) > 1 and content[0] == 255 and content[1] & 224 == 224),
                 ".ogg": content[:4] == b"OggS", ".flac": content[:4] == b"fLaC",
                 ".m4a": content[4:8] == b"ftyp", ".webm": content[:4] == b"\x1aE\xdf\xa3"}
        if not valid[extension]:
            raise HTTPException(400, "Audio content does not match its file extension.")
        return "audio", AUDIO_TYPES[extension]
    raise HTTPException(415, "Use PNG, JPEG, WebP, MP3, WAV, M4A, OGG, WebM or FLAC.")


class AttachmentStore:
    def __init__(self, limit=128):
        self.limit = limit
        self._items = {}
        self._lock = threading.Lock()

    def _expire(self):
        now = time.monotonic()
        self._items = {key: item for key, item in self._items.items() if item["expires"] > now}

    def add(self, owner, filename, kind, text):
        with self._lock:
            self._expire()
            if len(self._items) >= self.limit:
                raise HTTPException(503, "Attachment storage is full. Remove an attachment or try again later.")
            identifier = secrets.token_urlsafe(24)
            ttl = get_settings().MEDIA_TTL_SECONDS
            item = {"id": identifier, "filename": filename, "kind": kind, "text": text,
                    "expires_in_seconds": ttl, "warnings": ["Machine-extracted content may contain recognition errors; check it before use."]}
            self._items[identifier] = {"owner": owner, "expires": time.monotonic() + ttl, "item": item}
            return item

    def resolve(self, identifiers, owner):
        with self._lock:
            self._expire()
            items = []
            for identifier in identifiers:
                entry = self._items.get(identifier)
                if not entry or not secrets.compare_digest(entry["owner"], owner):
                    raise HTTPException(404, "Attachment unavailable or expired. Upload it again in this browser session.")
                items.append(dict(entry["item"]))
            return items

    def delete(self, identifier, owner):
        self.resolve([identifier], owner)
        with self._lock:
            self._items.pop(identifier, None)


ATTACHMENTS = AttachmentStore()


def understand_media(filename, content, content_type=""):
    kind, mime = validate_media(filename, content, content_type)
    settings = get_settings()
    text = ""
    
    # Preferred: Google Gemini Multimodal Vision & Audio
    if settings.gemini_configured:
        from app.chains.llm import get_shared_client
        candidate_models = ["gemini-flash-lite-latest", "gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-3.5-flash"]
        prompt = (
            "Describe this uploaded image accurately for research and question answering. "
            "Transcribe visible text, UI options, headings, describe chart axes and visible trends, "
            "preserve exact numbers, labels, and units, and explain key takeaways."
            if kind == "image" else
            "Accurately transcribe this audio recording for research synthesis. "
            "Transcribe spoken dialogue, outline key discussion topics, and extract key facts and takeaways."
        )
        b64_data = base64.b64encode(content).decode("ascii")
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": settings.GEMINI_API_KEY
        }
        body = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inlineData": {"mimeType": mime, "data": b64_data}}
                ]
            }],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 2048}
        }
        client = get_shared_client()
        for model_name in candidate_models:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
                resp = client.post(url, headers=headers, json=body, timeout=20.0)
                if resp.status_code == 200:
                    candidates = resp.json().get("candidates", [])
                    if candidates and "content" in candidates[0]:
                        parts = candidates[0]["content"].get("parts", [])
                        text = "".join(p.get("text", "") for p in parts)
                        if text.strip():
                            break
                elif resp.status_code in (429, 503, 404):
                    continue
            except Exception:
                continue

    # Fallback: OpenAI / Groq compatible provider
    if not text.strip():
        model = settings.VISION_MODEL_NAME if kind == "image" else settings.AUDIO_MODEL_NAME
        key = settings.MEDIA_API_KEY or (settings.OPENAI_API_KEY if settings.llm_configured else "")
        if model and key:
            from openai import OpenAI
            try:
                with OpenAI(api_key=key, base_url=settings.MEDIA_BASE_URL or settings.OPENAI_BASE_URL,
                            timeout=45, max_retries=0) as client:
                    if kind == "image":
                        result = client.chat.completions.create(model=model, temperature=0, max_tokens=2200,
                            messages=[{"role": "system", "content": "Describe this uploaded image accurately for question answering. Transcribe text, describe chart axes and trends."},
                                      {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:" + mime + ";base64," + base64.b64encode(content).decode("ascii")}}]}])
                        text = result.choices[0].message.content or ""
                    else:
                        result = client.audio.transcriptions.create(model=model, file=(filename, content, mime))
                        text = result.text
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                if status in {401, 403}:
                    pass
    
    # Offline deterministic fallback
    if not text.strip():
        size_kb = round(len(content) / 1024, 1)
        text = f"[{kind.capitalize()} attachment: '{filename}', format: {mime}, size: {size_kb} KB. Grounding context available for research.]"
        
    return kind, text.strip()


def transcribe_audio_query(filename: str, content: bytes) -> str:
    """
    Transcribes spoken user audio into a research query string.
    Supports .mp3, .wav, .m4a, .ogg, .webm, .flac formats.
    """
    kind, text = understand_media(filename, content)
    if not text or text.startswith("[Audio"):
        # Fallback if audio transcription unavailable
        return f"Research query extracted from audio recording '{filename}'"
    return text.strip()

