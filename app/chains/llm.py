"""Request-scoped, bounded model calls with truthful failure messages and usage."""
import json
import re
import threading
import time
import httpx
from app.config import get_settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")

_INVISIBLE_CHARS_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]+")
_COOLDOWN_LOCK = threading.Lock()
_MODEL_COOLDOWN: dict[str, float] = {}


def _clean_model_text(text: str) -> str:
    """Strip zero-width / BOM invisible characters and surrounding whitespace from model outputs."""
    if not isinstance(text, str):
        return ""
    return _INVISIBLE_CHARS_RE.sub("", text).strip()


def _is_cooled_down(model_name: str) -> bool:
    now = time.monotonic()
    with _COOLDOWN_LOCK:
        until = _MODEL_COOLDOWN.get(model_name)
        if until is None:
            return False
        if now >= until:
            _MODEL_COOLDOWN.pop(model_name, None)
            return False
        return True


def _mark_cooldown(model_name: str, reason: object) -> None:
    msg = str(reason).lower()
    if "404" in msg or "model_not_found" in msg or "decommissioned" in msg:
        ttl = 300.0
    elif "429" in msg or "413" in msg or "rate_limit" in msg or "tokens per minute" in msg:
        ttl = 25.0
    else:
        ttl = 12.0
    with _COOLDOWN_LOCK:
        _MODEL_COOLDOWN[model_name] = max(_MODEL_COOLDOWN.get(model_name, 0.0), time.monotonic() + ttl)


def _is_auth_error(exc: Exception) -> bool:
    status = getattr(exc, "status_code", getattr(exc, "code", None))
    msg = str(exc).lower()
    return status in (401, 403) or "invalid_api_key" in msg or "authentication" in msg


def _groq_pool(mode: str, configured_model: str, max_tokens: int) -> tuple[list[str], int]:
    if mode in ("fast", "quick"):
        actual_tokens = min(max(max_tokens, 650), 1800)
        base_pool = ["qwen/qwen3.8-27b", "openai/gpt-oss-20b", "openai/gpt-oss-120b"]
    elif mode in ("research", "deep"):
        actual_tokens = min(max(max_tokens, 1600), 3200)
        base_pool = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
    else:
        actual_tokens = min(max(max_tokens, 1200), 2400)
        base_pool = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]

    raw_pool = [
        m for m in dict.fromkeys([*base_pool, configured_model])
        if m and "llama-3" not in m and "allam" not in m
    ]
    return [m for m in raw_pool if not _is_cooled_down(m)], actual_tokens


def _gemini_pool(mode: str) -> list[str]:
    if mode in ("fast", "quick"):
        raw = ["gemini-flash-lite-latest", "gemini-3.5-flash-lite", "gemini-3-flash-preview"]
    else:
        raw = ["gemini-flash-lite-latest", "gemini-3-flash-preview", "gemini-3.5-flash-lite", "gemini-3.5-flash"]
    deduped = list(dict.fromkeys([m for m in raw if m]))
    active = [m for m in deduped if not _is_cooled_down(m)]
    return active or deduped


_shared_http_client: httpx.Client = None

def get_shared_client() -> httpx.Client:
    global _shared_http_client
    if _shared_http_client is None or _shared_http_client.is_closed:
        transport = httpx.HTTPTransport(
            local_address="0.0.0.0",
            retries=1,
            limits=httpx.Limits(max_keepalive_connections=25, max_connections=60, keepalive_expiry=120.0),
        )
        _shared_http_client = httpx.Client(
            transport=transport,
            http2=False,
            timeout=30.0,
            limits=httpx.Limits(max_keepalive_connections=25, max_connections=60, keepalive_expiry=120.0)
        )
    return _shared_http_client


class ModelUnavailable(RuntimeError):
    pass


class BoundedLLM:
    def __init__(self, mode="balanced", deadline=None, backend=None, disabled=False, privacy_mode=False):
        self.settings = get_settings()
        self.mode = mode
        self.privacy_mode = privacy_mode or (mode == "privacy")
        self.model = self.settings.model_for(mode) if backend is not None else None
        self.deadline = deadline
        self.backend = backend
        self.disabled = disabled
        self.failure = None
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def complete(self, system, prompt, max_tokens=1500, json_mode=False):
        if self.failure:
            raise ModelUnavailable(self.failure)

        # STRICT AIR-GAPPED PRIVACY MODE: ZERO EXTERNAL CLOUD CALLS
        from app.rag.privacy_guard import is_air_gapped, call_local_llm, get_last_local_model
        if self.privacy_mode or is_air_gapped():
            local_text = call_local_llm(system, prompt, model=self.model, max_tokens=max_tokens)
            if local_text:
                used = get_last_local_model()
                if used:
                    self.model = used
                return local_text
            raise ModelUnavailable(
                "Air-Gapped Privacy Mode: Local LLM daemon (Ollama/LM Studio) did not respond. "
                "Executing deterministic in-process private evidence synthesizer."
            )

        if self.disabled or (self.backend is None and not self.settings.llm_configured):
            raise ModelUnavailable("No text model is available. Configure a valid OPENAI_API_KEY and model on the server.")
        remaining = self.deadline - time.monotonic() if self.deadline else self.settings.LLM_TIMEOUT_SECONDS
        if remaining < 1:
            raise ModelUnavailable("The request's time budget was exhausted; no additional model call was made.")
        self.calls += 1
        try:
            if self.backend is not None:
                from langchain_core.messages import SystemMessage, HumanMessage
                result = self.backend.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
                text = result.content
            else:
                groq_available = bool(self.settings.OPENAI_API_KEY and not self.settings.OPENAI_API_KEY.startswith(("your-", "changeme")))
                gemini_available = self.settings.gemini_configured
                groq_model = self.settings.OPENAI_MODEL_NAME or "qwen/qwen3.8-27b"

                def _call_groq(timeout_sec):
                    groq_candidates, actual_tokens = _groq_pool(self.mode, groq_model, max_tokens)
                    if not groq_candidates:
                        return ""
                    from openai import OpenAI
                    # Reuse IPv4 keepalive client without entering `with OpenAI(...)` (which would close shared_client)
                    oai_client = OpenAI(
                        api_key=self.settings.OPENAI_API_KEY,
                        base_url=self.settings.OPENAI_BASE_URL,
                        timeout=timeout_sec,
                        max_retries=0,
                        http_client=get_shared_client(),
                    )
                    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
                    last_err = None
                    result = None
                    used_groq_model = ""
                    res_text = ""
                    for cand_model in groq_candidates:
                        try:
                            cand_kwargs = dict(kwargs)
                            if "qwen" in cand_model:
                                cand_kwargs["reasoning_effort"] = "none"
                                cand_tokens = min(actual_tokens, 1800)
                            elif "gpt-oss" in cand_model:
                                cand_kwargs["reasoning_effort"] = "low"
                                cand_tokens = min(actual_tokens, 3200)
                            else:
                                cand_tokens = min(actual_tokens, 2400)
                            result = oai_client.chat.completions.create(
                                model=cand_model,
                                temperature=0.1,
                                max_tokens=cand_tokens,
                                timeout=timeout_sec,
                                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                                **cand_kwargs,
                            )
                            cand_text = _clean_model_text(
                                (result.choices[0].message.content or "") if result and result.choices else ""
                            )
                            finish_reason = getattr(result.choices[0], "finish_reason", None) if result and result.choices else None
                            if not cand_text:
                                logger.warning(f"Groq model {cand_model} returned empty/invisible text; trying next candidate")
                                continue
                            if finish_reason == "length" and len(cand_text) < 80 and cand_model != groq_candidates[-1]:
                                logger.warning(f"Groq model {cand_model} truncated early at {len(cand_text)} chars; trying next candidate")
                                continue
                            res_text = cand_text
                            used_groq_model = cand_model
                            break
                        except Exception as cand_err:
                            last_err = cand_err
                            if _is_auth_error(cand_err):
                                raise
                            _mark_cooldown(cand_model, cand_err)
                            logger.warning(f"Groq candidate {cand_model} failed ({cand_err}); trying next candidate")
                            continue
                    if not res_text and last_err is not None:
                        raise last_err
                    if result and result.usage:
                        self.input_tokens += result.usage.prompt_tokens
                        self.output_tokens += result.usage.completion_tokens
                    if res_text and used_groq_model:
                        self.model = used_groq_model
                    return res_text

                def _call_gemini(timeout_sec):
                    if self.mode in ("fast", "quick"):
                        output_tokens = max(min(max_tokens, 2400), 1200)
                    elif self.mode in ("research", "deep"):
                        output_tokens = max(min(max_tokens, 4096), 2400)
                    else:
                        output_tokens = max(min(max_tokens, 3600), 2000)

                    candidate_models = _gemini_pool(self.mode)
                    headers = {
                        "Content-Type": "application/json",
                        "x-goog-api-key": self.settings.GEMINI_API_KEY
                    }
                    client = get_shared_client()
                    for model_name in candidate_models:
                        try:
                            cand_out_tokens = output_tokens
                            gen_config = {
                                "temperature": 0.2,
                                "maxOutputTokens": cand_out_tokens,
                            }
                            if "3.5-flash" in model_name and "lite" not in model_name:
                                cand_out_tokens = max(output_tokens, 2400)
                                gen_config["maxOutputTokens"] = cand_out_tokens
                                gen_config["thinkingConfig"] = {"thinkingBudget": 0}
                            if json_mode:
                                gen_config["responseMimeType"] = "application/json"
                            body = {
                                "systemInstruction": {"parts": [{"text": system}]},
                                "contents": [{"parts": [{"text": prompt}]}],
                                "generationConfig": gen_config,
                            }
                            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
                            resp = client.post(url, headers=headers, json=body, timeout=timeout_sec)
                            if resp.status_code == 200:
                                data = resp.json()
                                candidates_list = data.get("candidates", [])
                                if candidates_list and "content" in candidates_list[0]:
                                    first_cand = candidates_list[0]
                                    parts = first_cand["content"].get("parts", [])
                                    res_text = _clean_model_text(
                                        "".join(p.get("text", "") for p in parts if not p.get("thought"))
                                    )
                                    finish_reason = first_cand.get("finishReason")
                                    if not res_text:
                                        logger.warning(f"Gemini model {model_name} returned empty/invisible text; continuing")
                                        continue
                                    if finish_reason == "MAX_TOKENS" and len(res_text) < 120 and model_name != candidate_models[-1]:
                                        logger.warning(
                                            f"Gemini model {model_name} hit MAX_TOKENS with {len(res_text)} chars; trying next candidate"
                                        )
                                        continue
                                    usage = data.get("usageMetadata", {})
                                    self.input_tokens += usage.get("promptTokenCount", 0)
                                    self.output_tokens += usage.get("candidatesTokenCount", 0)
                                    self.model = model_name
                                    return res_text
                            elif resp.status_code in (404, 429, 500, 502, 503):
                                _mark_cooldown(model_name, resp.status_code)
                                logger.warning(f"Model {model_name} returned {resp.status_code}; continuing to next candidate")
                                continue
                        except Exception as gem_err:
                            _mark_cooldown(model_name, gem_err)
                            continue
                    return ""

                text = ""
                if groq_available:
                    try:
                        fast_groq_timeout = min(remaining, 6.0 if self.mode in ("fast", "quick") else 16.0)
                        text = _call_groq(timeout_sec=fast_groq_timeout)
                    except Exception as e:
                        if _is_auth_error(e):
                            raise
                        logger.warning(f"Fast Groq pass skipped ({e}); attempting instant Gemini fallback")
                        text = ""

                if not text and gemini_available:
                    try:
                        rem_after_groq = self.deadline - time.monotonic() if self.deadline else remaining
                        gemini_budget = min(max(rem_after_groq, 4.0), 10.0 if self.mode in ("fast", "quick") else 22.0)
                        text = _call_gemini(timeout_sec=gemini_budget)
                    except Exception as e:
                        logger.warning(f"Gemini call failed: {e}")
                        text = ""

            text = _clean_model_text(text) if isinstance(text, str) else ""
            if not text:
                if self.backend is None:
                    self.model = None
                raise ModelUnavailable("The text model returned an empty answer.")
            return text
        except ModelUnavailable:
            if self.backend is None and not getattr(self, "failure", None):
                self.model = None
            raise
        except Exception as exc:
            if self.backend is None:
                self.model = None
            status = getattr(exc, "status_code", getattr(exc, "code", None))
            if status in (401, 403):
                message = "The model provider rejected the configured credentials. Check the server API key and account permissions."
            elif status == 429:
                message = "The model provider is rate-limited or out of quota. Check your quota or try again later."
            elif status in (400, 404):
                message = "The model provider rejected this model or request. Check model configuration."
            elif "Timeout" in type(exc).__name__:
                message = "The model provider timed out within this request's budget."
            else:
                message = f"The model provider could not be reached: {exc}"
            self.failure = message
            raise ModelUnavailable(message) from None

    def stream_complete(self, system, prompt, max_tokens=2500):
        """Yields streamed text chunks in real-time from local LLM (when air-gapped) or Groq/Gemini."""
        from app.rag.privacy_guard import is_air_gapped, stream_local_llm, get_last_local_model
        if self.privacy_mode or is_air_gapped():
            yielded = False
            for chunk in stream_local_llm(system, prompt, model=self.model, max_tokens=max_tokens):
                yielded = True
                yield chunk
            used = get_last_local_model()
            if used:
                self.model = used
            if not yielded:
                raise ModelUnavailable(
                    "Air-Gapped Privacy Mode: Local LLM daemon (Ollama/LM Studio) did not respond."
                )
            return

        groq_available = bool(self.settings.OPENAI_API_KEY and not self.settings.OPENAI_API_KEY.startswith(("your-", "changeme")))
        gemini_available = self.settings.gemini_configured
        groq_model = self.settings.OPENAI_MODEL_NAME or "qwen/qwen3.8-27b"

        if groq_available:
            stream_candidates, actual_tokens = _groq_pool(self.mode, groq_model, max_tokens)
            if stream_candidates:
                try:
                    from openai import OpenAI
                    stream_timeout = min(self.deadline - time.monotonic() if self.deadline else 8.0, 18.0 if self.mode in ("research", "deep") else 6.0)
                    oai_client = OpenAI(
                        api_key=self.settings.OPENAI_API_KEY,
                        base_url=self.settings.OPENAI_BASE_URL,
                        timeout=stream_timeout,
                        max_retries=0,
                        http_client=get_shared_client(),
                    )
                    for cand_model in stream_candidates:
                        try:
                            extra_kw = {}
                            if "qwen" in cand_model:
                                extra_kw["reasoning_effort"] = "none"
                                cand_tokens = min(actual_tokens, 1800)
                            elif "gpt-oss" in cand_model:
                                extra_kw["reasoning_effort"] = "low"
                                cand_tokens = min(actual_tokens, 3200)
                            else:
                                cand_tokens = min(actual_tokens, 2400)
                            response = oai_client.chat.completions.create(
                                model=cand_model,
                                temperature=0.1,
                                max_tokens=cand_tokens,
                                timeout=stream_timeout,
                                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                                stream=True,
                                **extra_kw,
                            )
                            streamed_any = False
                            for chunk in response:
                                if chunk.choices and chunk.choices[0].delta.content:
                                    text_piece = chunk.choices[0].delta.content
                                    streamed_any = True
                                    self.output_tokens += 1
                                    yield text_piece
                            if streamed_any:
                                self.model = cand_model
                                self.calls += 1
                                return
                        except Exception as cand_err:
                            if _is_auth_error(cand_err):
                                raise
                            _mark_cooldown(cand_model, cand_err)
                            continue
                except Exception as e:
                    logger.warning(f"Groq streaming pass skipped ({e}); falling back to Gemini streaming.")

        if gemini_available:
            candidate_models = _gemini_pool(self.mode)
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.settings.GEMINI_API_KEY
            }
            streamed_any = False
            client = get_shared_client()
            gemini_stream_timeout = 25.0 if self.mode in ("research", "deep") else (15.0 if self.mode == "balanced" else 10.0)
            for model_name in candidate_models:
                try:
                    stream_max_tokens = 2400 if self.mode in ("fast", "quick") else max(max_tokens, 2400)
                    gen_cfg = {
                        "temperature": 0.2,
                        "maxOutputTokens": stream_max_tokens,
                    }
                    if "3.5-flash" in model_name and "lite" not in model_name:
                        gen_cfg["thinkingConfig"] = {"thinkingBudget": 0}
                    body = {
                        "systemInstruction": {"parts": [{"text": system}]},
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": gen_cfg,
                    }
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:streamGenerateContent?alt=sse"
                    with client.stream("POST", url, headers=headers, json=body, timeout=gemini_stream_timeout) as response:
                        if response.status_code == 200:
                            for line in response.iter_lines():
                                if line.startswith("data: "):
                                    try:
                                        data = json.loads(line[6:])
                                        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                                        for p in parts:
                                            if p.get("thought"):
                                                continue
                                            chunk = _INVISIBLE_CHARS_RE.sub("", p.get("text", ""))
                                            if chunk:
                                                streamed_any = True
                                                yield chunk
                                    except Exception:
                                        continue
                            if streamed_any:
                                self.model = model_name
                                self.calls += 1
                                return
                        elif response.status_code in (404, 429, 500, 502, 503):
                            _mark_cooldown(model_name, response.status_code)
                except Exception as gem_err:
                    _mark_cooldown(model_name, gem_err)
                    continue

        # Final complete fallback
        text = self.complete(system, prompt, max_tokens)
        yield text

    def json(self, system, prompt, max_tokens=1000):
        text = self.complete(system, prompt, max_tokens=max_tokens, json_mode=True)
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
        result = json.loads(text)
        if not isinstance(result, dict):
            raise ValueError("Expected a JSON object.")
        return result


def call_model(llm, state, system, prompt, max_tokens=1500, json_mode=False):
    is_privacy = (
        state.get("mode") == "privacy" or
        state.get("options", {}).get("privacy_mode", False) or
        state.get("privacy_mode", False)
    )
    client = llm if isinstance(llm, BoundedLLM) else BoundedLLM(
        state.get("mode", "balanced"), state.get("deadline"), backend=llm,
        disabled=state.get("offline", False), privacy_mode=is_privacy)
    if json_mode:
        return client.json(system, prompt, max_tokens)
    return client.complete(system, prompt, max_tokens)
