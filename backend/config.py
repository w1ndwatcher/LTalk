# backend/config.py
"""
Centralized configuration, loaded and validated once at startup.

Why this exists: scattered os.getenv("SOME_VAR") calls throughout app.py,
guardrails.py, etc. fail silently — a missing var becomes None, and the
resulting error surfaces later, deep in some unrelated stack trace (e.g.
"api_key must be set" from inside the Groq SDK). Production code should
fail LOUDLY, at startup, with one clear message listing everything that's
missing — not force you to debug backwards from a cryptic runtime error.

Import `config` and use `config.AZURE_SEARCH_ENDPOINT` etc. — if anything
required is missing, the import itself raises with a full list of what's
wrong, before the app tries to serve a single request.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    # --- Required: no sensible default, app cannot function without these ---
    HUGGINGFACEHUB_API_TOKEN: str
    LLAMA_GROQ_KEY: str
    AZURE_SEARCH_ENDPOINT: str
    AZURE_SEARCH_KEY: str

    # --- Required only once Redis is wired in (see note below) ---
    REDIS_HOST: str = ""
    REDIS_PORT: int = 6380
    REDIS_PASSWORD: str = ""
    REDIS_SSL: bool = True

    # --- Optional, with sensible defaults ---
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    LLM_MODEL: str = "openai/gpt-oss-120b"
    GUARDRAIL_GROQ_MODEL: str = "llama-3.1-8b-instant"
    AZURE_SEARCH_INDEX_NAME: str = "learntrail-qa-index"
    RETRIEVAL_K: int = 3
    RELEVANCE_THRESHOLD: float = 0.3
    MAX_QUESTION_LENGTH: int = 1000
    CACHE_SIMILARITY_THRESHOLD: float = 0.92
    RATE_LIMIT_ASK: str = "20 per minute"
    RATE_LIMIT_FEEDBACK: str = "60 per minute"

    # --- Azure OpenAI (used by the eval harness, and later Phase 4/5) ---
    AZURE_OPENAI_ENDPOINT: str = ""
    AZURE_OPENAI_KEY: str = ""
    AZURE_OPENAI_DEPLOYMENT: str = "gpt-4.1-mini"
    AZURE_OPENAI_API_VERSION: str = "2024-10-21"

    # --- LangSmith (optional — /feedback degrades to a clear error without it) ---
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "learntrail"

    # --- Frontend origin for CORS — see app.py; do not default this to "*" ---
    FRONTEND_ORIGIN: str = "http://localhost:3000"

    # --- Environment marker (changes a few behaviors — see app.py) ---
    ENV: str = "development"  # "development" | "production"

    @property
    def is_production(self) -> bool:
        return self.ENV.lower() == "production"

    @property
    def redis_configured(self) -> bool:
        return bool(self.REDIS_HOST and self.REDIS_PASSWORD)


def _load() -> Config:
    required = {
        "HUGGINGFACEHUB_API_TOKEN": os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        "LLAMA_GROQ_KEY": os.getenv("LLAMA_GROQ_KEY"),
        "AZURE_SEARCH_ENDPOINT": os.getenv("AZURE_SEARCH_ENDPOINT"),
        "AZURE_SEARCH_KEY": os.getenv("AZURE_SEARCH_KEY"),
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise EnvironmentError(
            "Missing required environment variable(s): " + ", ".join(missing) +
            ". Set these in .env (dev) or the platform's secret/env config (prod) "
            "before starting the app."
        )

    return Config(
        **required,
        REDIS_HOST=os.getenv("REDIS_HOST", ""),
        REDIS_PORT=int(os.getenv("REDIS_PORT", 6380)),
        REDIS_PASSWORD=os.getenv("REDIS_PASSWORD", ""),
        REDIS_SSL=os.getenv("REDIS_SSL", "true").lower() == "true",
        EMBEDDING_MODEL=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2"),
        LLM_MODEL=os.getenv("LLM_MODEL", "openai/gpt-oss-120b"),
        GUARDRAIL_GROQ_MODEL=os.getenv("GUARDRAIL_GROQ_MODEL", "llama-3.1-8b-instant"),
        AZURE_SEARCH_INDEX_NAME=os.getenv("AZURE_SEARCH_INDEX_NAME", "learntrail-qa-index"),
        RETRIEVAL_K=int(os.getenv("RETRIEVAL_K", 3)),
        RELEVANCE_THRESHOLD=float(os.getenv("RELEVANCE_THRESHOLD", 0.3)),
        MAX_QUESTION_LENGTH=int(os.getenv("MAX_QUESTION_LENGTH", 1000)),
        CACHE_SIMILARITY_THRESHOLD=float(os.getenv("CACHE_SIMILARITY_THRESHOLD", 0.92)),
        RATE_LIMIT_ASK=os.getenv("RATE_LIMIT_ASK", "20 per minute"),
        RATE_LIMIT_FEEDBACK=os.getenv("RATE_LIMIT_FEEDBACK", "60 per minute"),
        AZURE_OPENAI_ENDPOINT=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        AZURE_OPENAI_KEY=os.getenv("AZURE_OPENAI_KEY", ""),
        AZURE_OPENAI_DEPLOYMENT=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1-mini"),
        AZURE_OPENAI_API_VERSION=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        LANGCHAIN_TRACING_V2=os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true",
        LANGCHAIN_API_KEY=os.getenv("LANGCHAIN_API_KEY", ""),
        LANGCHAIN_PROJECT=os.getenv("LANGCHAIN_PROJECT", "learntrail"),
        FRONTEND_ORIGIN=os.getenv("FRONTEND_ORIGIN", "http://localhost:3000"),
        ENV=os.getenv("ENV", "development"),
    )


config = _load()