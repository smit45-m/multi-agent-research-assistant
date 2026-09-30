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
            elif self.settings.gemini_configured:
                # Optimized candidate order: 200 OK fast models first, avoid 503/404 traps
                if self.mode in ("fast", "quick"):
                    fast_pool = ["gemini-flash-lite-latest", "gemini-3.5-flash-lite"]
                    output_tokens = min(max_tokens, getattr(self.settings, "FAST_MODE_MAX_OUTPUT_TOKENS", 280))
                    default_timeout = 3.8
                    candidates = ["gemini-flash-lite-latest", self.model, "gemini-3.5-flash-lite"]
                elif self.mode in ("research", "deep"):
                    fast_pool = ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-flash-lite-latest"]
                    output_tokens = max(max_tokens, 3000)
                    default_timeout = 18.0
                    candidates = [self.model] + fast_pool
                else:
                    fast_pool = ["gemini-flash-lite-latest", "gemini-3.5-flash-lite"]
                    output_tokens = max(max_tokens, 1500)
                    default_timeout = 8.0
                    candidates = [self.model] + fast_pool
                
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
                
                text = ""
                last_err = None
                client = get_shared_client()
                for model_name in candidate_models:
                    rem_now = self.deadline - time.monotonic() if self.deadline else self.settings.LLM_TIMEOUT_SECONDS
                    if rem_now < 1.0:
                        break
                    per_try_timeout = min(rem_now, default_timeout)
                    try:
                        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
                        resp = client.post(url, headers=headers, json=body, timeout=per_try_timeout)
                        if resp.status_code == 200:
                            data = resp.json()
                            candidates_list = data.get("candidates", [])
                            if candidates_list and "content" in candidates_list[0]:
                                parts = candidates_list[0]["content"].get("parts", [])
                                text = "".join(p.get("text", "") for p in parts)
                            usage = data.get("usageMetadata", {})
                            self.input_tokens += usage.get("promptTokenCount", 0)
                            self.output_tokens += usage.get("candidatesTokenCount", 0)
                            self.model = model_name
                            break
                        else:
                            last_err = f"{resp.status_code}: {resp.text[:120]}"
                            continue
                    except Exception as e:
                        last_err = e
                        continue
                if not text and last_err:
                    logger.warning(f"Gemini fallback exhausted; last error: {last_err}")
            else:
                from openai import OpenAI
                with OpenAI(api_key=self.settings.OPENAI_API_KEY, base_url=self.settings.OPENAI_BASE_URL,
                            timeout=min(remaining, self.settings.LLM_TIMEOUT_SECONDS), max_retries=0) as client:
                    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
                    result = client.chat.completions.create(
                        model=self.model, temperature=0.1, max_tokens=max_tokens,
                        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}], **kwargs)
                text = result.choices[0].message.content or ""
                if result.usage:
                    self.input_tokens += result.usage.prompt_tokens
                    self.output_tokens += result.usage.completion_tokens
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
        """Yields streamed text chunks in real-time from Gemini or OpenAI."""
        if not self.settings.gemini_configured:
            text = self.complete(system, prompt, max_tokens)
            yield text
            return

        if self.mode in ("fast", "quick"):
            fast_pool = ["gemini-flash-lite-latest", "gemini-3.1-flash-lite", "gemini-3.5-flash-lite"]
        elif self.mode in ("research", "deep"):
            fast_pool = ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-flash-lite-latest"]
        else:
            fast_pool = ["gemini-flash-lite-latest", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]

        candidates = [self.model] + fast_pool
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
                "maxOutputTokens": max(max_tokens, 2048)
            }
        }
        streamed_any = False
        client = get_shared_client()
        for model_name in candidate_models:
                try:
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:streamGenerateContent?alt=sse"
                    with client.stream("POST", url, headers=headers, json=body) as response:
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
                        else:
                            continue
                except Exception:
                    continue
        if not streamed_any:
            # Fallback to complete
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
