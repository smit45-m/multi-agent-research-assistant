"""
Privacy Guard and Local Air-Gapped RAG / Local Neural LLM Engine.
Guarantees 100% local, air-gapped processing of sensitive company data.
Zero network packets leave the machine: Gemini, Groq, OpenAI, and Web Search are blocked.
"""
import contextlib
import ipaddress
import json
import logging
import os
import re
import threading
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
from urllib.parse import urlparse

from app.config import get_settings
from app.rag.relevance import grounded_sentences, relevance, tokens

logger = logging.getLogger(__name__)

# Thread-local strict air-gap isolation guard
_AIR_GAP_STATE = threading.local()

# Patterns for sensitive company PII / credentials
PII_PATTERNS = [
    (re.compile(r"\b(Primary\s+Owner|Owner|Author|Contact|Employee|Patient|Prepared\s+by)(\s*:\s*)([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})"), r"\1\2[NAME_PROTECTED]"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,15}\b"), "[EMAIL_PROTECTED]"),
    (re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[PHONE_PROTECTED]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN_PROTECTED]"),
    (re.compile(r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b"), "[CARD_PROTECTED]"),
    (
        re.compile(
            r"(?:api[_-]?key|secret|token|password|bearer)"
            r"(?:\s*[:=]\s*['\"]?[a-zA-Z0-9_\-\.]{8,}['\"]?"
            r"|\s+['\"]?(?:(?=[a-zA-Z0-9_\-\.]*[0-9_\-])[a-zA-Z0-9_\-\.]{8,}|[a-zA-Z0-9_\-\.]{16,})['\"]?)",
            re.IGNORECASE,
        ),
        "token=[REDACTED_CREDENTIAL]",
    ),
]

_INJECTION_PHRASE_RE = re.compile(
    r"(?:"
    r"(?:ignore|disregard|forget|override|bypass)\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|system|safety|security)?\s*(?:instructions?|prompts?|rules?|constraints?|guardrails?)"
    r"|system\s*override\b"
    r"|(?:reveal|print|output|leak|exfiltrate|echo)\s+(?:your\s+)?(?:system\s*prompt|hidden\s*instructions?|api[_\s]*keys?|secrets?|credentials?|pwned|hacked)"
    r"|\b(?:output|respond|reply|say|print)\s+(?:only|exactly)\s*[:\"']"
    r"|(?:<<SYS>>|<\|im_start\|>|\[/?INST\])"
    r")",
    re.IGNORECASE,
)
_INLINE_OVERRIDE_RE = re.compile(
    r"(?:ignore|disregard|forget|override|bypass)\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|system|safety|security)\s+(?:instructions?|prompts?|rules?|constraints?|guardrails?)[^\n.!?]*[.!?]?",
    re.IGNORECASE,
)


def sanitize_untrusted_text(text: str, strip_pii: bool = False) -> str:
    """Strips prompt-injection directives across line breaks/sentences and optionally masks PII."""
    if not text:
        return ""
    cleaned = _INLINE_OVERRIDE_RE.sub(" ", text.replace("\r\n", "\n"))
    kept_lines: List[str] = []
    for raw_line in cleaned.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        segments = re.split(r"(?<=[.!?])\s+", line)
        safe_segments = [seg.strip() for seg in segments if seg.strip() and not _INJECTION_PHRASE_RE.search(seg)]
        if safe_segments:
            kept_lines.append(" ".join(safe_segments))
    result = "\n".join(kept_lines).strip()
    return redact_pii(result) if strip_pii else result

_EMBED_HINTS = ("embed", "minilm", "bge-", "e5-", "nomic-embed", "mxbai")


@contextlib.contextmanager
def air_gapped_scope(active: bool = True) -> Iterator[None]:
    """Enforces thread-scoped air-gap status so cloud helpers deterministically abort."""
    prev = getattr(_AIR_GAP_STATE, "active", False)
    prev_model = getattr(_AIR_GAP_STATE, "local_model", None)
    _AIR_GAP_STATE.active = bool(active or prev)
    try:
        yield
    finally:
        _AIR_GAP_STATE.active = prev
        _AIR_GAP_STATE.local_model = prev_model


def is_air_gapped() -> bool:
    """Returns True if the current execution thread is inside an Air-Gapped Private session."""
    return bool(getattr(_AIR_GAP_STATE, "active", False))


def set_last_local_model(model_name: Optional[str]) -> None:
    _AIR_GAP_STATE.local_model = model_name


def get_last_local_model() -> Optional[str]:
    return getattr(_AIR_GAP_STATE, "local_model", None)


def is_loopback_url(url: str) -> bool:
    """Validates that an endpoint resolves strictly to localhost / loopback interface."""
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").strip().lower()
        if host in {"localhost", "127.0.0.1", "::1"}:
            return True
        ip = ipaddress.ip_address(host)
        return ip.is_loopback
    except Exception:
        return False


def assert_local_url(url: str) -> str:
    """Blocks non-loopback network requests inside air-gapped sessions."""
    if not is_loopback_url(url):
        raise PermissionError(f"Air-Gapped Privacy Policy blocked external URL: {url}")
    return url


def redact_pii(text: str) -> str:
    """Masks high-risk company credentials and PII to protect sensitive corporate assets."""
    if not text:
        return ""
    sanitized = text
    for pattern, replacement in PII_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)
    return sanitized


def _is_embedding_model(name: str) -> bool:
    low = name.lower()
    return any(h in low for h in _EMBED_HINTS)


def _score_local_candidate(model_info: Dict[str, Any]) -> Tuple[int, float]:
    """
    Ranks locally installed chat models for laptop/workstation GPU efficiency & quality.
    Prefers GPU-resident fast instruction models (e.g. llama3.2:latest 2.0GB -> ~98 tok/s,
    or compact <= 8.5GB models) over giant 15GB+ models that spill to swap and time out.
    """
    name = str(model_info.get("name", "")).lower()
    size_bytes = float(model_info.get("size") or 0.0)
    size_gb = size_bytes / (1024 ** 3) if size_bytes > 0 else 4.0

    if _is_embedding_model(name):
        return (-1000, -size_gb)

    score = 0
    # Size / GPU VRAM fit tier (typical consumer/company workstation has 6-12GB VRAM)
    if 1.6 <= size_gb <= 4.5:
        score += 45  # Sweet spot: 3B-4B models fit 100% in VRAM at 80-120 tok/s
    elif 4.5 < size_gb <= 8.5:
        score += 36  # 7B-9B Q4 models (fit 8GB+ VRAM when num_ctx <= 4096)
    elif 0.5 <= size_gb < 1.6:
        score += 24  # 1B-2B ultra-fast models
    elif size_gb > 11.0:
        score -= 50  # Heavy 20B-30B models spill into CPU RAM / swap on 8-16GB laptops

    # Family quality bonus
    if "llama3.2:latest" in name or name == "llama3.2:3b":
        score += 25
    elif "llama3.2" in name and ":1b" not in name:
        score += 20
    elif "qwen3.5" in name or "qwen2.5" in name:
        score += 18
    elif "llama3.1" in name or "mistral" in name or "gemma3" in name or "phi" in name:
        score += 15
    elif "gemma4" in name:
        score += 10
    elif "coder" in name and size_gb > 10.0:
        score -= 15

    # Prefer models already loaded hot in VRAM
    if model_info.get("loaded_in_memory"):
        score += 30

    return (score, -size_gb)


def list_local_llm_models() -> Dict[str, Any]:
    """
    Probes localhost Ollama & LM Studio / OpenAI-compatible daemons (0 cloud calls)
    and returns installed local models sorted by GPU suitability.
    """
    import httpx

    settings = get_settings()
    ollama_base = assert_local_url(getattr(settings, "LOCAL_LLM_URL", "http://127.0.0.1:11434")).rstrip("/")
    local_oai_base = assert_local_url(getattr(settings, "LOCAL_OPENAI_URL", "http://127.0.0.1:1234/v1")).rstrip("/")

    # 1. Check Ollama
    try:
        loaded_names = set()
        with httpx.Client(timeout=1.2) as client:
            try:
                ps_resp = client.get(f"{ollama_base}/api/ps")
                if ps_resp.status_code == 200:
                    loaded_names = {m.get("name", "") for m in ps_resp.json().get("models", [])}
            except Exception:
                pass
            resp = client.get(f"{ollama_base}/api/tags")
            if resp.status_code == 200:
                raw_models = resp.json().get("models", [])
                parsed: List[Dict[str, Any]] = []
                for m in raw_models:
                    name = m.get("name") or m.get("model") or ""
                    if not name or _is_embedding_model(name):
                        continue
                    details = m.get("details") or {}
                    size_bytes = int(m.get("size") or 0)
                    info = {
                        "name": name,
                        "provider": "ollama",
                        "size_gb": round(size_bytes / (1024 ** 3), 2) if size_bytes else None,
                        "size": size_bytes,
                        "parameter_size": details.get("parameter_size", ""),
                        "quantization": details.get("quantization_level", ""),
                        "loaded_in_memory": name in loaded_names,
                    }
                    parsed.append(info)
                parsed.sort(key=_score_local_candidate, reverse=True)
                recommended = parsed[0]["name"] if parsed else "llama3.2:latest"
                return {
                    "available": bool(parsed),
                    "provider": "ollama",
                    "endpoint": ollama_base,
                    "recommended_model": recommended,
                    "models": parsed,
                }
    except Exception:
        pass

    # 2. Check LM Studio / vLLM / LocalAI OpenAI-compatible localhost endpoint
    try:
        with httpx.Client(timeout=1.2) as client:
            resp = client.get(f"{local_oai_base}/models")
            if resp.status_code == 200:
                data = resp.json().get("data", [])
                parsed = [
                    {
                        "name": m.get("id", "local-model"),
                        "provider": "local_openai",
                        "size_gb": None,
                        "size": 0,
                        "parameter_size": "",
                        "quantization": "",
                        "loaded_in_memory": True,
                    }
                    for m in data
                    if m.get("id") and not _is_embedding_model(m.get("id", ""))
                ]
                recommended = parsed[0]["name"] if parsed else "local-model"
                return {
                    "available": bool(parsed) or bool(data),
                    "provider": "local_openai",
                    "endpoint": local_oai_base,
                    "recommended_model": recommended,
                    "models": parsed,
                }
    except Exception:
        pass

    return {
        "available": False,
        "provider": "none",
        "endpoint": "",
        "recommended_model": "in-process-synthesizer",
        "models": [],
    }


def _resolve_target_models(
    discovery: Dict[str, Any], preferred_model: Optional[str] = None
) -> List[str]:
    """Resolves ordered candidate model list from discovery & optional user preference."""
    installed = [m["name"] for m in discovery.get("models", [])]
    pref = (preferred_model or "").strip()
    if pref and pref.lower() not in {"auto", "default"}:
        # Match exact tag or prefix (e.g. "llama3.2" -> "llama3.2:latest")
        exact = [m for m in installed if m == pref or m.split(":")[0] == pref]
        others = [m for m in installed if m not in exact]
        ordered = (exact or [pref]) + [m for m in others if (next((x.get("size_gb") or 0 for x in discovery.get("models", []) if x["name"] == m), 0)) <= 8.5]
        return list(dict.fromkeys(ordered))

    # Auto mode: prefer compact GPU-friendly models (skip giant >10.5GB models if smaller ones exist)
    compact = [
        m["name"]
        for m in discovery.get("models", [])
        if (m.get("size_gb") is None or m["size_gb"] <= 10.5)
    ]
    return compact or installed or [discovery.get("recommended_model") or "llama3.2:latest"]


def check_local_llm(preferred_model: Optional[str] = None) -> Tuple[bool, str, str]:
    """
    Checks if a local open-source LLM server (Ollama or LM Studio/vLLM) is running locally.
    Returns: (is_available, provider, selected_model_name)
    """
    discovery = list_local_llm_models()
    if not discovery.get("available"):
        return False, "none", "in-process-synthesizer"
    candidates = _resolve_target_models(discovery, preferred_model=preferred_model)
    return True, discovery["provider"], candidates[0]


def _ollama_payload(
    model_name: str,
    system: str,
    prompt: str,
    stream: bool,
    max_tokens: int,
    num_ctx: int,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": stream,
        "keep_alive": "15m",
        "options": {
            "temperature": 0.2,
            "num_predict": max_tokens,
            "num_ctx": num_ctx,
        },
    }
    low = model_name.lower()
    # Disable slow internal thinking token burn on hybrid thinking models so GPU streams immediately
    if any(t in low for t in ("qwen3", "gemma4", "deepseek-r1")):
        payload["think"] = False
    return payload


def stream_local_llm(
    system: str,
    prompt: str,
    model: Optional[str] = None,
    max_tokens: int = 1600,
    timeout_sec: float = 75.0,
) -> Iterator[str]:
    """
    Streams tokens in real-time from localhost Ollama or LM Studio / OpenAI-compatible daemon.
    Zero bytes ever leave localhost (127.0.0.1).
    """
    import httpx

    settings = get_settings()
    discovery = list_local_llm_models()
    if not discovery.get("available"):
        return

    pref = model if model and model.lower() != "auto" else getattr(settings, "LOCAL_LLM_MODEL", "auto")
    candidates = _resolve_target_models(discovery, preferred_model=pref)
    num_ctx = getattr(settings, "LOCAL_LLM_NUM_CTX", 4096)
    provider = discovery["provider"]

    if provider == "ollama":
        url = assert_local_url(f"{discovery['endpoint']}/api/chat")
        for idx, cand_model in enumerate(candidates[:2]):
            yielded_any = False
            req_timeout = min(timeout_sec, 45.0 if idx == 0 else 30.0)
            payload = _ollama_payload(cand_model, system, prompt, stream=True, max_tokens=max_tokens, num_ctx=num_ctx)
            try:
                with httpx.Client(timeout=httpx.Timeout(req_timeout, connect=2.0)) as client:
                    with client.stream("POST", url, json=payload) as resp:
                        if resp.status_code != 200:
                            logger.warning(f"Local Ollama {cand_model} HTTP {resp.status_code}")
                            continue
                        for line in resp.iter_lines():
                            if not line:
                                continue
                            try:
                                data = json.loads(line)
                            except Exception:
                                continue
                            piece = (data.get("message") or {}).get("content", "")
                            if piece:
                                if not yielded_any:
                                    set_last_local_model(f"ollama/{cand_model}")
                                yielded_any = True
                                yield piece
                if yielded_any:
                    return
            except Exception as e:
                logger.warning(f"Local Ollama stream failed on {cand_model}: {e}")
                if yielded_any:
                    return
                continue

    elif provider == "local_openai":
        url = assert_local_url(f"{discovery['endpoint']}/chat/completions")
        for cand_model in candidates[:2]:
            yielded_any = False
            payload = {
                "model": cand_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "max_tokens": max_tokens,
                "stream": True,
            }
            try:
                with httpx.Client(timeout=httpx.Timeout(min(timeout_sec, 45.0), connect=2.0)) as client:
                    with client.stream("POST", url, json=payload) as resp:
                        if resp.status_code != 200:
                            continue
                        for line in resp.iter_lines():
                            if not line.startswith("data: "):
                                continue
                            raw_ev = line[6:].strip()
                            if raw_ev == "[DONE]":
                                break
                            try:
                                ev = json.loads(raw_ev)
                                piece = ev.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if piece:
                                    if not yielded_any:
                                        set_last_local_model(f"local/{cand_model}")
                                    yielded_any = True
                                    yield piece
                            except Exception:
                                continue
                if yielded_any:
                    return
            except Exception as e:
                logger.warning(f"Local OpenAI daemon stream failed on {cand_model}: {e}")
                if yielded_any:
                    return
                continue


def call_local_llm(
    system: str,
    prompt: str,
    model: Optional[str] = None,
    max_tokens: int = 1600,
    timeout_sec: float = 60.0,
) -> Optional[str]:
    """
    Calls a local open-source model daemon over localhost only.
    Guarantees no outbound external internet calls.
    """
    parts = list(stream_local_llm(system, prompt, model=model, max_tokens=max_tokens, timeout_sec=timeout_sec))
    full = "".join(parts).strip()
    return full if full else None


def local_vision_ocr(image_bytes: bytes, prompt: str) -> Optional[str]:
    """
    Executes multimodal OCR / image understanding strictly on localhost Ollama vision models
    (e.g. `qwen3.5:latest`, `llava`, `gemma3`) with zero cloud egress.
    """
    import base64
    import httpx

    discovery = list_local_llm_models()
    if not discovery.get("available") or discovery.get("provider") != "ollama":
        return None

    vision_candidates = [
        m["name"]
        for m in discovery.get("models", [])
        if any(v in m["name"].lower() for v in ("qwen3.5", "llava", "minicpm", "moondream", "gemma3", "vision"))
    ]
    if not vision_candidates:
        return None

    b64_img = base64.b64encode(image_bytes).decode("ascii")
    url = assert_local_url(f"{discovery['endpoint']}/api/chat")
    for cand in vision_candidates[:1]:
        try:
            payload = {
                "model": cand,
                "messages": [{"role": "user", "content": prompt, "images": [b64_img]}],
                "stream": False,
                "keep_alive": "10m",
                "options": {"temperature": 0.1, "num_predict": 1024, "num_ctx": 4096},
            }
            if "qwen3" in cand.lower():
                payload["think"] = False
            with httpx.Client(timeout=45.0) as client:
                resp = client.post(url, json=payload)
                if resp.status_code == 200:
                    text = ((resp.json().get("message") or {}).get("content") or "").strip()
                    if text:
                        set_last_local_model(f"ollama/{cand}")
                        return text
        except Exception as exc:
            logger.warning(f"Local Ollama vision OCR failed on {cand}: {exc}")
    return None


def synthesize_local_airgapped_report(
    query: str,
    docs: List[Dict[str, Any]],
    confidence: float = 0.92,
    user_memory_context: str = "",
) -> str:
    """
    In-process, zero-network-egress semantic RAG synthesizer fallback for confidential company data
    when no local LLM daemon (Ollama / LM Studio) is running.
    """
    if not docs and user_memory_context and user_memory_context.strip():
        safe_mem = sanitize_untrusted_text(user_memory_context.strip(), strip_pii=True)
        return f"### Recalled Workspace Context\n\n{safe_mem}"
    if not docs:
        return (
            f"No relevant local or attached documents were found for **\"{redact_pii(query)}\"**. "
            "Upload private files or index documents in the workspace before running Air-Gapped Privacy mode."
        )

    ranked_candidates: List[Tuple[float, float, int, int, str]] = []
    seen_norm = set()

    for idx, doc in enumerate(docs):
        doc_id = idx + 1
        raw_title = doc.get("filename") or doc.get("title") or f"Document {doc_id}"
        title = os.path.basename(raw_title)
        content = sanitize_untrusted_text(doc.get("content", ""), strip_pii=True)
        if not content:
            continue

        doc_rel = relevance(query, content, title)

        # Combine wrapped prose sentences with line-level structured fields (e.g., key: value memos)
        candidate_units: List[str] = grounded_sentences(query, content, limit=8)
        for line in content.split("\n"):
            line = line.strip(" -*\t")
            if len(line) < 20:
                continue
            for seg in re.split(r"(?<=[.!?])\s+", line):
                seg = seg.strip()
                if len(seg) >= 20:
                    candidate_units.append(seg)

        for sent_idx, s in enumerate(candidate_units):
            norm = re.sub(r"\s+", " ", s.lower()).strip()
            if norm in seen_norm:
                continue
            seen_norm.add(norm)
            s_rel = relevance(query, s)
            ranked_candidates.append((s_rel, doc_rel, doc_id, sent_idx, s))

    if not ranked_candidates:
        return f"No readable grounded passages remained after sanitization for **\"{redact_pii(query)}\"**."

    relevant_only = [item for item in ranked_candidates if item[0] > 0.0]
    pool = relevant_only if relevant_only else ranked_candidates
    pool.sort(key=lambda x: (round(x[0], 4), round(x[1], 4), -x[2], -x[3]), reverse=True)

    selected = pool[:8]
    # Reorder top facts coherently by (doc_id, sent_idx) after relevance filtering when readable,
    # keeping highest-scoring sentence first as direct answer.
    lead = selected[0]
    details = selected[1:]

    def _cite(sentence: str, doc_id: int) -> str:
        clean_s = sentence.strip()
        if not clean_s.endswith((".", "!", "?")):
            clean_s = clean_s + "."
        return f"{clean_s} [{doc_id}]"

    lead_text = _cite(lead[4], lead[2])
    if not details:
        return f"### Direct Answer\n\n{lead_text}"

    bullets = "\n".join(f"- {_cite(item[4], item[2])}" for item in details)
    return f"### Direct Answer\n\n{lead_text}\n\n### Supporting Evidence\n\n{bullets}"
