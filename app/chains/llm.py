"""Request-scoped, bounded model calls with truthful failure messages and usage."""
import json
import re
import time
from app.config import get_settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")

_INVISIBLE_CHARS_RE = re.compile(r"[\u200b\u200c\u200d\ufeff]+")


def _clean_model_text(text: str) -> str:
    """Strip zero-width / BOM invisible characters and surrounding whitespace from model outputs."""
    if not isinstance(text, str):
        return ""
    return _INVISIBLE_CHARS_RE.sub("", text).strip()


import httpx

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
        self.model = self.settings.model_for(mode)
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

                groq_attempted = False
                groq_rate_limited = False

                def _call_groq(timeout_sec):
                    nonlocal groq_attempted, groq_rate_limited
                    groq_attempted = True
                    from openai import OpenAI
                    # Set max_retries=0 so OpenAI never blocks on 14-second retry-after sleeps
                    with OpenAI(api_key=self.settings.OPENAI_API_KEY, base_url=self.settings.OPENAI_BASE_URL,
                                timeout=timeout_sec, max_retries=0) as oai_client:
                        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
                        # Dynamic token bounds: qwen3.8-27b has a 1000 OTPM cap, whereas gpt-oss-120b/20b support 4000+ tokens
                        if self.mode in ("fast", "quick"):
                            actual_tokens = max(max_tokens, 850)
                            base_pool = ["qwen/qwen3.8-27b", "openai/gpt-oss-20b", "openai/gpt-oss-120b", "allam-2-7b"]
                        elif self.mode in ("research", "deep"):
                            actual_tokens = max(max_tokens, 3200)
                            base_pool = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b", "allam-2-7b"]
                        else:
                            actual_tokens = max(max_tokens, 2200)
                            base_pool = ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b", "allam-2-7b"]
                        groq_candidates = [m for m in dict.fromkeys([*base_pool, groq_model]) if "llama-3" not in m]
                        last_err = None
                        result = None
                        used_groq_model = groq_model
                        res_text = ""
                        for cand_model in groq_candidates:
                            try:
                                cand_kwargs = dict(kwargs)
                                if "qwen" in cand_model:
                                    cand_kwargs["reasoning_effort"] = "none"
                                    cand_tokens = min(actual_tokens, 850)
                                elif "gpt-oss" in cand_model:
                                    cand_kwargs["reasoning_effort"] = "low"
                                    cand_tokens = min(actual_tokens, 3600)
                                else:
                                    cand_tokens = min(actual_tokens, 2400)
                                result = oai_client.chat.completions.create(
                                    model=cand_model, temperature=0.1, max_tokens=cand_tokens,
                                    messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}], **cand_kwargs)
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
                                if "model_not_found" in str(cand_err) or "404" in str(cand_err) or "429" in str(cand_err):
                                    continue
                                raise
                        if not res_text and last_err is not None:
                            raise last_err
                    if result and result.usage:
                        self.input_tokens += result.usage.prompt_tokens
                        self.output_tokens += result.usage.completion_tokens
                    self.model = used_groq_model
                    return res_text

                def _call_gemini(timeout_sec):
                    if self.mode in ("fast", "quick"):
                        output_tokens = max(min(max_tokens, 2400), 1200)
                        candidates = ["gemini-flash-lite-latest", "gemini-3.5-flash-lite", "gemini-3.5-flash"]
                    elif self.mode in ("research", "deep"):
                        output_tokens = max(min(max_tokens, 8192), 4096)
                        candidates = ["gemini-3.5-flash", "gemini-flash-lite-latest", "gemini-3.5-flash-lite"]
                    else:
                        output_tokens = max(min(max_tokens, 4096), 2400)
                        candidates = ["gemini-3.5-flash", "gemini-flash-lite-latest", "gemini-3.5-flash-lite"]

                    candidate_models = list(dict.fromkeys([m for m in candidates if m]))
                    headers = {
                        "Content-Type": "application/json",
                        "x-goog-api-key": self.settings.GEMINI_API_KEY
                    }
                    client = get_shared_client()
                    for model_name in candidate_models:
                        try:
                            # Thinking models (e.g. gemini-3.5-flash without -lite) charge hidden reasoning tokens
                            # against maxOutputTokens, so give non-lite models plenty of token headroom to prevent
                            # mid-sentence cutoffs.
                            cand_out_tokens = output_tokens
                            if "3.5-flash" in model_name and "lite" not in model_name:
                                cand_out_tokens = max(output_tokens, 4096)
                            gen_config = {
                                "temperature": 0.2,
                                "maxOutputTokens": cand_out_tokens,
                            }
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
                            elif resp.status_code in (429, 503):
                                logger.warning(f"Model {model_name} returned {resp.status_code}; continuing to next candidate")
                                continue
                        except Exception:
                            continue
                    return ""

                text = ""
                # Fast Groq multi-model pool across all modes (gpt-oss-120b/20b for 2200-token research, qwen3.8-27b for sub-2s fast mode)
                if groq_available:
                    try:
                        fast_groq_timeout = min(remaining, 6.0 if self.mode in ("fast", "quick") else 16.0)
                        text = _call_groq(timeout_sec=fast_groq_timeout)
                    except Exception as e:
                        if "429" in str(e) or "rate_limit" in str(e):
                            groq_rate_limited = True
                        logger.warning(f"Fast Groq pass skipped ({e}); attempting instant Gemini fallback")
                        text = ""

                # Gemini fallback pass (using fast gemini-3.5-flash)
                if not text and gemini_available:
                    try:
                        gemini_budget = min(remaining, 8.0 if self.mode in ("fast", "quick") else 22.0)
                        text = _call_gemini(timeout_sec=gemini_budget)
                    except Exception as e:
                        logger.warning(f"Gemini call failed: {e}")
                        text = ""

            text = _clean_model_text(text) if isinstance(text, str) else ""
            if not text:
                raise ModelUnavailable("The text model returned an empty answer.")
            return text
        except ModelUnavailable:
            raise
        except Exception as exc:
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

        # 1. Fast Groq multi-model streaming (<1.5s TTFT, up to 2200 tokens on gpt-oss-120b/20b)
        if groq_available:
            try:
                from openai import OpenAI
                stream_timeout = min(self.deadline - time.monotonic() if self.deadline else 8.0, 18.0 if self.mode in ("research", "deep") else 6.0)
                if self.mode in ("fast", "quick"):
                    stream_candidates = ["qwen/qwen3.8-27b", "openai/gpt-oss-20b", "openai/gpt-oss-120b", "allam-2-7b"]
                else:
                    stream_candidates = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b", "allam-2-7b"]
                with OpenAI(api_key=self.settings.OPENAI_API_KEY, base_url=self.settings.OPENAI_BASE_URL,
                            timeout=stream_timeout, max_retries=0) as oai_client:
                    for cand_model in stream_candidates:
                        try:
                            extra_kw = {}
                            if "qwen" in cand_model:
                                extra_kw["reasoning_effort"] = "none"
                                cand_tokens = min(max(max_tokens, 800), 850)
                            elif "gpt-oss" in cand_model:
                                extra_kw["reasoning_effort"] = "low"
                                cand_tokens = min(max(max_tokens, 1200 if self.mode in ("fast", "quick") else 3200), 3600)
                            else:
                                cand_tokens = min(max(max_tokens, 1000 if self.mode in ("fast", "quick") else 2400), 2400)
                            response = oai_client.chat.completions.create(
                                model=cand_model, temperature=0.1, max_tokens=cand_tokens,
                                messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                                stream=True, **extra_kw)
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
                        except Exception:
                            continue
            except Exception as e:
                logger.warning(f"Groq streaming pass skipped ({e}); falling back to Gemini streaming.")

        # 2. Gemini SSE streaming (prioritize fast non-thinking Flash Lite for fast mode)
        if gemini_available:
            if self.mode in ("fast", "quick"):
                candidate_models = ["gemini-flash-lite-latest", "gemini-3.5-flash-lite", "gemini-3.5-flash"]
            else:
                candidate_models = ["gemini-3.5-flash", "gemini-flash-lite-latest", "gemini-3.5-flash-lite"]

            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.settings.GEMINI_API_KEY
            }
            streamed_any = False
            client = get_shared_client()
            gemini_stream_timeout = 25.0 if self.mode in ("research", "deep") else (15.0 if self.mode == "balanced" else 8.0)
            for model_name in candidate_models:
                try:
                    stream_max_tokens = 2400 if self.mode in ("fast", "quick") else max(max_tokens, 4096)
                    if "3.5-flash" in model_name and "lite" not in model_name:
                        stream_max_tokens = max(stream_max_tokens, 4096)
                    body = {
                        "systemInstruction": {"parts": [{"text": system}]},
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "temperature": 0.2,
                            "maxOutputTokens": stream_max_tokens,
                        }
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
                                return
                except Exception:
                    continue

        # 3. Final complete fallback
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
