"""Anthropic Claude LLM provider."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from ragmax.core.exceptions import GenerationError
from ragmax.core.utils import require_dependency


class AnthropicLLM:
    """Generate text using Anthropic's Claude models.

    Supports Claude Sonnet 4, Claude Opus 4, Haiku, and other variants.
    Converts OpenAI-style messages to Anthropic's format.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 2048,
        temperature: float = 0.1,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        anthropic = require_dependency("anthropic", "anthropic")
        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._extra = kwargs

    def _convert_messages(
        self, messages: list[dict[str, str]]
    ) -> tuple[str | None, list[dict[str, str]]]:
        """Split a system message (if any) from user/assistant messages."""
        system = None
        converted: list[dict[str, str]] = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                converted.append(msg)
        return system, converted

    async def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        try:
            system, msgs = self._convert_messages(messages)
            params: dict[str, Any] = {
                "model": self.model,
                "messages": msgs,
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
            }
            if system:
                params["system"] = system
            response = await self._client.messages.create(**params)
            return response.content[0].text
        except Exception as exc:
            raise GenerationError(f"Anthropic generation failed: {exc}") from exc

    async def generate_stream(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> AsyncIterator[str]:
        try:
            system, msgs = self._convert_messages(messages)
            params: dict[str, Any] = {
                "model": self.model,
                "messages": msgs,
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
            }
            if system:
                params["system"] = system

            async with self._client.messages.stream(**params) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as exc:
            raise GenerationError(f"Anthropic streaming failed: {exc}") from exc
