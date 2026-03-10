"""Factory for creating LLM provider instances."""

from __future__ import annotations

from typing import Any

from ragmax.core.config import LLMConfig
from ragmax.core.exceptions import ConfigError
from ragmax.core.protocols import LLMProvider


def create_llm(config: LLMConfig | None = None, **kwargs: Any) -> LLMProvider:
    """Instantiate an LLMProvider from config or keyword arguments."""
    if config is None:
        config = LLMConfig(**kwargs)

    provider = config.provider

    if provider == "openai":
        from ragmax.generation.openai import OpenAILLM

        return OpenAILLM(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            **config.extra,
        )
    elif provider == "anthropic":
        from ragmax.generation.anthropic import AnthropicLLM

        return AnthropicLLM(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            **config.extra,
        )
    elif provider == "google":
        from ragmax.generation.google import GoogleLLM

        return GoogleLLM(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            **config.extra,
        )
    else:
        raise ConfigError(f"Unknown LLM provider: {provider!r}")
