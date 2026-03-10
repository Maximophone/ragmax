"""Google Gemini LLM provider."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from ragmax.core.exceptions import GenerationError
from ragmax.core.utils import require_dependency


class GoogleLLM:
    """Generate text using Google Gemini models.

    Supports Gemini 3 Flash Preview, 2.5 Pro, and other variants.
    """

    def __init__(
        self,
        model: str = "gemini-3-flash-preview",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        api_key: str | None = None,
        **kwargs: Any,
    ) -> None:
        genai = require_dependency("google.genai", "google")
        from google.genai import types

        client_kwargs: dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        self._client = genai.Client(**client_kwargs)
        self._types = types
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._extra = kwargs

    def _build_contents(self, messages: list[dict[str, str]]) -> list[Any]:
        """Convert OpenAI-style messages to Gemini Content objects."""
        contents = []
        for msg in messages:
            role = msg["role"]
            if role == "system":
                # Gemini handles system via system_instruction in config
                continue
            gemini_role = "user" if role == "user" else "model"
            contents.append(
                self._types.Content(
                    role=gemini_role,
                    parts=[self._types.Part(text=msg["content"])],
                )
            )
        return contents

    def _get_system_instruction(self, messages: list[dict[str, str]]) -> str | None:
        for msg in messages:
            if msg["role"] == "system":
                return msg["content"]
        return None

    async def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        try:
            contents = self._build_contents(messages)
            config_kwargs: dict[str, Any] = {
                "temperature": kwargs.get("temperature", self.temperature),
                "max_output_tokens": kwargs.get("max_tokens", self.max_tokens),
            }
            system = self._get_system_instruction(messages)
            if system:
                config_kwargs["system_instruction"] = system

            response = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=self._types.GenerateContentConfig(**config_kwargs),
            )
            return response.text or ""
        except Exception as exc:
            raise GenerationError(f"Google generation failed: {exc}") from exc

    async def generate_stream(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> AsyncIterator[str]:
        try:
            contents = self._build_contents(messages)
            config_kwargs: dict[str, Any] = {
                "temperature": kwargs.get("temperature", self.temperature),
                "max_output_tokens": kwargs.get("max_tokens", self.max_tokens),
            }
            system = self._get_system_instruction(messages)
            if system:
                config_kwargs["system_instruction"] = system

            for chunk in self._client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=self._types.GenerateContentConfig(**config_kwargs),
            ):
                if chunk.text:
                    yield chunk.text
        except Exception as exc:
            raise GenerationError(f"Google streaming failed: {exc}") from exc
