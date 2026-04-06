from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    llm_provider: str = os.getenv("LLM_PROVIDER", "auto")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    openai_fallback_models: str = os.getenv("OPENAI_FALLBACK_MODELS", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    gemini_fallback_models: str = os.getenv(
        "GEMINI_FALLBACK_MODELS", "gemini-2.0-flash-lite,gemini-flash-latest,gemini-2.5-flash"
    )
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "mixtral-8x7b-32768")
    groq_fallback_models: str = os.getenv("GROQ_FALLBACK_MODELS", "llama2-70b-4096")
    llm_max_retries: int = int(os.getenv("LLM_MAX_RETRIES", "2"))
    llm_retry_base_delay_seconds: float = float(
        os.getenv("LLM_RETRY_BASE_DELAY_SECONDS", "1.5")
    )
    llm_min_interval_seconds: float = float(os.getenv("LLM_MIN_INTERVAL_SECONDS", "0.8"))
    llm_cb_failures: int = int(os.getenv("LLM_CB_FAILURES", "2"))
    llm_cb_cooldown_seconds: int = int(os.getenv("LLM_CB_COOLDOWN_SECONDS", "60"))
    agent_mode: str = os.getenv("AGENT_MODE", "fast")
    enable_web_research: bool = os.getenv("ENABLE_WEB_RESEARCH", "1") == "1"
    research_query_budget: int = int(os.getenv("RESEARCH_QUERY_BUDGET", "2"))
    enable_critic: bool = os.getenv("ENABLE_CRITIC", "0") == "1"
    llm_cache_ttl_seconds: int = int(os.getenv("LLM_CACHE_TTL_SECONDS", "180"))
    max_steps: int = int(os.getenv("MAX_STEPS", "6"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))


def get_settings() -> Settings:
    return Settings()
