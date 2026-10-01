"""Request-scoped, bounded model calls with truthful failure messages and usage."""
import json
import time
from app.config import get_settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__, "INFO")


import httpx

_shared_http_client: httpx.Client = None

def get_shared_client() -> httpx.Client:
    global _shared_http_client
    if _shared_http_client is None or _shared_http_client.is_closed:
        _shared_http_client = httpx.Client(
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
        if self.privacy_mode:
            from app.rag.privacy_guard import call_local_llm
            local_text = call_local_llm(system, prompt, model=self.model)
            if local_text:
                return local_text
            raise ModelUnavailable(
                "Air-Gapped Privacy Mode: Local LLM daemon (Ollama/LM Studio) not running. "
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
                        # Dynamic token bounds: 1200 for fast, up to 8192 for deep research
                        if self.mode in ("fast", "quick"):
                            actual_tokens = min(max_tokens, 1200)
                        elif self.mode in ("research", "deep"):
                            actual_tokens = min(max_tokens, 8192)
                        else:
                            actual_tokens = min(max_tokens, 2500)
                        result = oai_client.chat.completions.create(
                            model=groq_model, temperature=0.1, max_tokens=actual_tokens,
                            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}], **kwargs)
                    res_text = result.choices[0].message.content or ""
                    if result.usage:
                        self.input_tokens += result.usage.prompt_tokens
                        self.output_tokens += result.usage.completion_tokens
                    self.model = groq_model
                    return res_text.strip()

                def _call_gemini(timeout_sec):
                    if self.mode in ("fast", "quick"):
                        output_tokens = min(max_tokens, 1200)
                        candidates = ["gemini-flash-lite-latest", "gemini-3.5-flash"]
                    elif self.mode in ("research", "deep"):
                        output_tokens = min(max_tokens, 8192)
                        candidates = ["gemini-flash-lite-latest", self.model, "gemini-3.5-flash"]
                    else:
                        output_tokens = min(max_tokens, 2500)
                        candidates = [self.model, "gemini-flash-lite-latest", "gemini-3.5-flash"]
                    
                    candidate_models = list(dict.fromkeys([m for m in candidates if m]))
                    headers = {
                        "Content-Type": "application/json",
                        "x-goog-api-key": self.settings.GEMINI_API_KEY
                    }
                    gen_config = {
                        "temperature": 0.2,
                        "maxOutputTokens": output_tokens
                    }
                    if json_mode:
                        gen_config["responseMimeType"] = "application/json"
                    body = {
                        "systemInstruction": {"parts": [{"text": system}]},
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": gen_config
                    }
                    client = get_shared_client()
                    for model_name in candidate_models:
                        try:
                            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
                            resp = client.post(url, headers=headers, json=body, timeout=timeout_sec)
                            if resp.status_code == 200:
                                data = resp.json()
                                candidates_list = data.get("candidates", [])
                                if candidates_list and "content" in candidates_list[0]:
                                    parts = candidates_list[0]["content"].get("parts", [])
                                    res_text = "".join(p.get("text", "") for p in parts)
                                    if res_text.strip():
                                        usage = data.get("usageMetadata", {})
                                        self.input_tokens += usage.get("promptTokenCount", 0)
                                        self.output_tokens += usage.get("candidatesTokenCount", 0)
                                        self.model = model_name
                                        return res_text.strip()
                            elif resp.status_code in (429, 503):
                                logger.warning(f"Model {model_name} returned {resp.status_code}; continuing to next candidate")
                                continue
                        except Exception:
                            continue
                    return ""

                text = ""
                # Prioritize Groq for ultra-low latency (<1.5s) in Fast Mode or JSON extractions
                if groq_available and (self.mode in ("fast", "quick") or json_mode):
                    try:
                        fast_groq_timeout = min(remaining, 4.0 if self.mode in ("fast", "quick") else 20.0)
                        text = _call_groq(timeout_sec=fast_groq_timeout)
                    except Exception as e:
                        if "429" in str(e) or "rate_limit" in str(e):
                            groq_rate_limited = True
                        logger.warning(f"Fast Groq pass skipped ({e}); attempting instant Gemini fallback")
                        text = ""

                # Primary Gemini pass for Research & Balanced synthesis
                if not text and gemini_available:
                    try:
                        gemini_budget = min(remaining, 3.5 if self.mode in ("fast", "quick") else (65.0 if self.mode in ("research", "deep") else 20.0))
                        text = _call_gemini(timeout_sec=gemini_budget)
                    except Exception as e:
                        logger.warning(f"Gemini call failed: {e}")
                        text = ""

                # Groq fallback if Gemini failed and Groq wasn't already rate-limited
                if not text and groq_available and not groq_rate_limited:
                    try:
                        groq_budget = min(remaining, 45.0 if self.mode in ("research", "deep") else 15.0)
                        text = _call_groq(timeout_sec=groq_budget)
                    except Exception as e:
                        logger.warning(f"Groq fallback failed: {e}")
                        text = ""

            if not isinstance(text, str) or not text.strip():
                raise ModelUnavailable("The text model returned an empty answer.")
            return text.strip()
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
        """Yields streamed text chunks in real-time from Groq or Gemini."""
        if self.privacy_mode:
            text = self.complete(system, prompt, max_tokens)
            yield text
            return

        groq_available = bool(self.settings.OPENAI_API_KEY and not self.settings.OPENAI_API_KEY.startswith(("your-", "changeme")))
        gemini_available = self.settings.gemini_configured
        groq_model = self.settings.OPENAI_MODEL_NAME or "qwen/qwen3.8-27b"

        # 1. Fast Groq streaming (<1.5s real-time streaming)
        if groq_available and (self.mode in ("fast", "quick") or not gemini_available):
            try:
                from openai import OpenAI
                stream_timeout = min(self.deadline - time.monotonic() if self.deadline else 8.0, 60.0 if self.mode in ("research", "deep") else 8.0)
                with OpenAI(api_key=self.settings.OPENAI_API_KEY, base_url=self.settings.OPENAI_BASE_URL,
                            timeout=stream_timeout, max_retries=0) as oai_client:
                    actual_tokens = min(max_tokens, 1200) if self.mode in ("fast", "quick") else (min(max_tokens, 8192) if self.mode in ("research", "deep") else min(max_tokens, 2500))
                    response = oai_client.chat.completions.create(
                        model=groq_model, temperature=0.1, max_tokens=actual_tokens,
                        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                        stream=True)
                    streamed_any = False
                    for chunk in response:
                        if chunk.choices and chunk.choices[0].delta.content:
                            text_piece = chunk.choices[0].delta.content
                            streamed_any = True
                            self.output_tokens += 1
                            yield text_piece
                    if streamed_any:
                        self.model = groq_model
                        return
            except Exception as e:
                logger.warning(f"Groq streaming pass skipped ({e}); falling back to Gemini streaming.")

        # 2. Gemini SSE streaming
        if gemini_available:
            if self.mode in ("fast", "quick"):
                fast_pool = ["gemini-flash-lite-latest", "gemini-3.5-flash"]
            elif self.mode in ("research", "deep"):
                fast_pool = ["gemini-flash-lite-latest", self.model, "gemini-3.5-flash"]
            else:
                fast_pool = ["gemini-flash-lite-latest", self.model, "gemini-3.5-flash"]

            candidates = fast_pool + [self.model]
            candidate_models = list(dict.fromkeys([m for m in candidates if m]))

            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.settings.GEMINI_API_KEY
            }
            body = {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": min(max_tokens, 1200) if self.mode in ("fast", "quick") else (min(max_tokens, 8192) if self.mode in ("research", "deep") else max(max_tokens, 2500))
                }
            }
            streamed_any = False
            client = get_shared_client()
            gemini_stream_timeout = 65.0 if self.mode in ("research", "deep") else (20.0 if self.mode == "balanced" else 8.0)
            for model_name in candidate_models:
                try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:streamGenerateContent?alt=sse"
                    with client.stream("POST", url, headers=headers, json=body, timeout=gemini_stream_timeout) as response:
                        if response.status_code == 200:
                            for line in response.iter_lines():
                                if line.startswith("data: "):
                                    try:
                                        data = json.loads(line[6:])
                                        parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                                        for p in parts:
                                            chunk = p.get("text", "")
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
