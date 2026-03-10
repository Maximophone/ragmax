"""OpenAI embedding provider."""

from __future__ import annotations

from ragmax.core.exceptions import EmbeddingError
from ragmax.core.utils import require_dependency, batched


class OpenAIEmbedder:
    """Generate embeddings via the OpenAI API.

    Supports ``text-embedding-3-small``, ``text-embedding-3-large``,
    and ``text-embedding-ada-002``.  The ``dimensions`` parameter enables
    Matryoshka truncation on v3 models.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
        batch_size: int = 64,
        api_key: str | None = None,
    ) -> None:
        openai = require_dependency("openai", "openai")
        self._client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
        self._dimensions = dimensions
        self.batch_size = batch_size

    @property
    def dimension(self) -> int:
        defaults = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        if self._dimensions is not None:
            return self._dimensions
        return defaults.get(self.model, 1536)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        all_embeddings: list[list[float]] = []
        try:
            for batch in batched(texts, self.batch_size):
                kwargs: dict = {"input": batch, "model": self.model}
                if self._dimensions is not None:
                    kwargs["dimensions"] = self._dimensions
                response = await self._client.embeddings.create(**kwargs)
                all_embeddings.extend([d.embedding for d in response.data])
        except Exception as exc:
            raise EmbeddingError(f"OpenAI embedding failed: {exc}") from exc
        return all_embeddings
