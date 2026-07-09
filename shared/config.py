"""Shared configuration for all microservices."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""

    # Single-company deployment: this instance serves exactly one company.
    # ORG_ID is a fixed internal identifier (used for ChromaDB chunk scoping).
    ORG_ID: str = os.getenv("ORG_ID", "company")

    # LLM provider
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")

    # Ollama settings
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral:7b")

    # Anthropic settings
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str = os.getenv(
        "ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"
    )

    # ChromaDB settings
    CHROMADB_PATH: str = os.getenv("CHROMADB_PATH", "/data/chromadb")

    # Microservice URLs
    AS1_URL: str = os.getenv("AS1_URL", "http://localhost:8001")
    AS2_URL: str = os.getenv("AS2_URL", "http://localhost:8002")
    AS3_URL: str = os.getenv("AS3_URL", "http://localhost:8003")
    AGA_URL: str = os.getenv("AGA_URL", "http://localhost:8004")
    AIU_URL: str = os.getenv("AIU_URL", "http://localhost:8005")
    ORCHESTRATOR_URL: str = os.getenv("ORCHESTRATOR_URL", "http://localhost:8000")

    # Timeouts. AS agents evaluate 10-30+ requirements each with one LLM
    # call per requirement — on CPU inference (Ollama without GPU) a single
    # call can take 1-2 minutes, so the agent timeout must be generous.
    AS_TIMEOUT: int = int(os.getenv("AS_TIMEOUT", "3600"))
    AGA_TIMEOUT: int = int(os.getenv("AGA_TIMEOUT", "600"))
    AIU_TIMEOUT: int = int(os.getenv("AIU_TIMEOUT", "300"))

    # LLM temperature — always 0 for reproducibility
    LLM_TEMPERATURE: float = 0.0

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def get_llm(temperature: float = 0.0) -> Any:
    """
    LLM factory.
    Returns ChatOllama or ChatAnthropic based on LLM_PROVIDER env var.
    Always uses temperature=0 for deterministic output.
    """
    settings = get_settings()

    if settings.LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if not settings.ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set but LLM_PROVIDER=anthropic"
            )
        return ChatAnthropic(
            model=settings.ANTHROPIC_MODEL,
            api_key=settings.ANTHROPIC_API_KEY,
            temperature=temperature,
        )
    else:
        # Default to Ollama
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=temperature,
        )
