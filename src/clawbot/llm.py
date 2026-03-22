from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from openai import OpenAI

from .config import Settings


class LLMClient(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        ...


class OpenAIClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = OpenAI(api_key=settings.openai_api_key)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        try:
            resp = self._client.responses.create(
                model=self._settings.openai_model,
                temperature=self._settings.temperature,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return resp.output_text
        except Exception as exc:
            return f"LLM_PROVIDER_ERROR: OpenAI request failed: {exc}"


class GeminiClient:
    _last_request_ts: float = 0.0

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": self._settings.temperature},
        }
        body = json.dumps(payload).encode("utf-8")
        models = _build_gemini_model_candidates(
            self._settings.gemini_model,
            self._settings.gemini_fallback_models,
        )
        endpoints = ["v1beta", "v1"]
        errors: list[str] = []

        for model in models:
            for api_version in endpoints:
                endpoint = (
                    f"https://generativelanguage.googleapis.com/{api_version}/models/"
                    f"{model}:generateContent"
                    f"?key={self._settings.gemini_api_key}"
                )
                result = self._request_with_retry(endpoint, body, model=model, api_version=api_version)
                if result.startswith("LLM_PROVIDER_ERROR:"):
                    errors.append(result)

                    # 429 is usually quota/global and switching models rarely helps immediately.
                    if "HTTP 429" in result:
                        return result

                    # If model does not exist for this version, try next model quickly.
                    if "HTTP 404" in result:
                        break
                    continue

                return result

        if errors:
            last_error = errors[-1]
            return (
                "LLM_PROVIDER_ERROR: Gemini unavailable after retries and model fallback. "
                f"Last error: {last_error}"
            )
        return "LLM_PROVIDER_ERROR: Gemini returned no usable output."

    def _request_with_retry(self, endpoint: str, body: bytes, model: str, api_version: str) -> str:
        max_retries = max(0, self._settings.llm_max_retries)
        base_delay = max(0.1, self._settings.llm_retry_base_delay_seconds)
        min_interval = max(0.0, self._settings.llm_min_interval_seconds)

        for attempt in range(max_retries + 1):
            now = time.time()
            since_last = now - GeminiClient._last_request_ts
            if since_last < min_interval:
                time.sleep(min_interval - since_last)

            request = Request(
                endpoint,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(request, timeout=25) as response:
                    raw = response.read().decode("utf-8", errors="ignore")
                GeminiClient._last_request_ts = time.time()
                return _extract_gemini_text(raw)
            except HTTPError as exc:
                status = exc.code
                if status in {429, 500, 503} and attempt < max_retries:
                    time.sleep(base_delay * (2**attempt))
                    continue
                return (
                    "LLM_PROVIDER_ERROR: Gemini HTTP "
                    f"{status} for model={model} api_version={api_version}"
                )
            except URLError as exc:
                if attempt < max_retries:
                    time.sleep(base_delay * (2**attempt))
                    continue
                return f"LLM_PROVIDER_ERROR: Gemini network error: {exc}"
            except KeyboardInterrupt:
                return "LLM_PROVIDER_ERROR: Gemini request interrupted while waiting for retry."
            except Exception as exc:
                return f"LLM_PROVIDER_ERROR: Gemini request failed: {exc}"

        return "LLM_PROVIDER_ERROR: Gemini retries exhausted."


class MockClient:
    """Fallback model for local development without API access."""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        _ = system_prompt

        # Provide deterministic JSON actions to keep the loop testable offline.
        if "Return valid JSON" in user_prompt:
            if "tool_result" in user_prompt:
                return json.dumps(
                    {
                        "type": "final",
                        "content": "I completed the task using available context and generated a concise answer.",
                    }
                )
            return json.dumps(
                {
                    "type": "final",
                    "content": "Mock mode: Provide OPENAI_API_KEY or GEMINI_API_KEY in .env for full reasoning and tool use.",
                }
            )

        return "Mock response"


@dataclass
class NamedLLMClient:
    name: str
    client: LLMClient


def create_llm_client(settings: Settings) -> LLMClient:
    provider = settings.llm_provider.strip().lower()

    if provider == "openai":
        if settings.openai_api_key:
            return FailoverClient([NamedLLMClient("openai", OpenAIClient(settings))], settings)
        return MockClient()

    if provider == "gemini":
        clients: list[NamedLLMClient] = []
        if settings.gemini_api_key:
            clients.append(NamedLLMClient("gemini", GeminiClient(settings)))
        if settings.openai_api_key:
            clients.append(NamedLLMClient("openai", OpenAIClient(settings)))
        if clients:
            return FailoverClient(clients, settings)
        return MockClient()

    clients: list[NamedLLMClient] = []
    if settings.openai_api_key:
        clients.append(NamedLLMClient("openai", OpenAIClient(settings)))
    if settings.gemini_api_key:
        clients.append(NamedLLMClient("gemini", GeminiClient(settings)))
    if clients:
        return FailoverClient(clients, settings)

    return MockClient()


class FailoverClient:
    _cache: dict[str, tuple[float, str]] = {}
    _breaker_state: dict[str, dict[str, float]] = {}

    def __init__(self, clients: list[NamedLLMClient], settings: Settings) -> None:
        self._clients = clients
        self._settings = settings

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        cache_key = _make_cache_key(system_prompt, user_prompt)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        errors: list[str] = []
        candidates = self._healthy_clients()
        if not candidates:
            candidates = self._clients

        for named in candidates:
            result = named.client.complete(system_prompt, user_prompt)
            if not _is_provider_error(result):
                self._set_cached(cache_key, result)
                self._on_success(named.name)
                return result
            self._on_failure(named.name)
            errors.append(result)

        # Return strict JSON to avoid agent looping when all providers fail.
        details = " | ".join(errors[-2:]) if errors else "No provider details available."
        return json.dumps(
            {
                "type": "final",
                "content": (
                    "All configured LLM providers are temporarily unavailable. "
                    "Check API quota, model name, or key permissions and retry shortly. "
                    f"Details: {details}. Circuit-breaker state: {self._breaker_snapshot()}"
                ),
            }
        )

    def _healthy_clients(self) -> list[NamedLLMClient]:
        now = time.time()
        healthy: list[NamedLLMClient] = []
        for named in self._clients:
            state = self._breaker_state.get(named.name, {})
            open_until = float(state.get("open_until", 0.0))
            if open_until <= now:
                healthy.append(named)
        return healthy

    def _on_success(self, provider_name: str) -> None:
        self._breaker_state[provider_name] = {"failures": 0.0, "open_until": 0.0}

    def _on_failure(self, provider_name: str) -> None:
        threshold = max(1, self._settings.llm_cb_failures)
        cooldown = max(1, self._settings.llm_cb_cooldown_seconds)

        state = self._breaker_state.get(provider_name, {"failures": 0.0, "open_until": 0.0})
        failures = int(state.get("failures", 0.0)) + 1
        open_until = float(state.get("open_until", 0.0))

        if failures >= threshold:
            open_until = time.time() + cooldown
            failures = 0

        self._breaker_state[provider_name] = {
            "failures": float(failures),
            "open_until": float(open_until),
        }

    def _breaker_snapshot(self) -> str:
        now = time.time()
        parts: list[str] = []
        for named in self._clients:
            state = self._breaker_state.get(named.name, {"failures": 0.0, "open_until": 0.0})
            open_until = float(state.get("open_until", 0.0))
            remaining = max(0, int(open_until - now))
            failures = int(state.get("failures", 0.0))
            status = f"open:{remaining}s" if remaining > 0 else "closed"
            parts.append(f"{named.name}={status}/f={failures}")
        return ", ".join(parts)

    def _get_cached(self, cache_key: str) -> str | None:
        ttl = max(0, self._settings.llm_cache_ttl_seconds)
        if ttl == 0:
            return None
        item = self._cache.get(cache_key)
        if not item:
            return None
        created_at, value = item
        if (time.time() - created_at) > ttl:
            self._cache.pop(cache_key, None)
            return None
        return value

    def _set_cached(self, cache_key: str, value: str) -> None:
        if max(0, self._settings.llm_cache_ttl_seconds) == 0:
            return
        self._cache[cache_key] = (time.time(), value)


def safe_json_loads(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return {"type": "final", "content": raw.strip() or "No output produced."}


def _is_provider_error(text: str) -> bool:
    return text.startswith("LLM_PROVIDER_ERROR:")


def _build_gemini_model_candidates(primary: str, fallback_csv: str) -> list[str]:
    models: list[str] = []
    for model in [primary, *fallback_csv.split(",")]:
        cleaned = model.strip()
        if cleaned and cleaned not in models:
            models.append(cleaned)
    return models


def _extract_gemini_text(raw: str) -> str:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return raw.strip() or "No output produced by Gemini."

    candidates = parsed.get("candidates", [])
    if candidates:
        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        text = "".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()
        if text:
            return text

    return raw.strip() or "No output produced by Gemini."


def _make_cache_key(system_prompt: str, user_prompt: str) -> str:
    digest = hashlib.sha256()
    digest.update(system_prompt.encode("utf-8", errors="ignore"))
    digest.update(b"\n---\n")
    digest.update(user_prompt.encode("utf-8", errors="ignore"))
    return digest.hexdigest()
