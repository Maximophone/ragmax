"""OpenAI LLM provider."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from ragmax.core.exceptions import GenerationError
from ragmax.core.utils import require_dependency


class OpenAILLM:
    """Generate text using OpenAI chat completions.

    Supports GPT-4o, GPT-4o-mini, o1, o3-mini, and compatible models.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        openai = require_dependency("openai", "openai")
        self._client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._extra = kwargs

    async def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        try:
            params = {
                "model": self.model,
                "messages": messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                **self._extra,
            }
            response = await self._client.chat.completions.create(**params)
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise GenerationError(f"OpenAI generation failed: {exc}") from exc

    async def generate_stream(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> AsyncIterator[str]:
        try:
            params = {
                "model": self.model,
                "messages": messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "stream": True,
                **self._extra,
            }
            response = await self._client.chat.completions.create(**params)
            async for chunk in response:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
        except Exception as exc:
            raise GenerationError(f"OpenAI streaming failed: {exc}") from exc
