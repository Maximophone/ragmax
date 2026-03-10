"""Anthropic Voyage embedding provider.

Anthropic partners with Voyage AI for embeddings.  This provider uses
the Voyage API (``voyageai`` package) which provides high-quality
embeddings optimised for retrieval.
"""

from __future__ import annotations

from ragmax.core.exceptions import EmbeddingError
from ragmax.core.utils import require_dependency, batched


class AnthropicEmbedder:
    """Generate embeddings via the Voyage AI API (Anthropic's recommended embedder).

    Models: ``voyage-3-large``, ``voyage-3``, ``voyage-3-lite``,
    ``voyage-code-3``.
    """

    def __init__(
        self,
        model: str = "voyage-3",
        batch_size: int = 64,
        api_key: str | None = None,
    ) -> None:
        voyageai = require_dependency("voyageai", "anthropic")
        self._client = voyageai.AsyncClient(api_key=api_key)
        self.model = model
        self.batch_size = batch_size

    @property
    def dimension(self) -> int:
        defaults = {
            "voyage-3-large": 2048,
            "voyage-3": 1024,
            "voyage-3-lite": 512,
            "voyage-code-3": 1024,
        }
        return defaults.get(self.model, 1024)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        all_embeddings: list[list[float]] = []
        try:
            for batch in batched(texts, self.batch_size):
                result = await self._client.embed(batch, model=self.model)
                all_embeddings.extend(result.embeddings)
        except Exception as exc:
            raise EmbeddingError(f"Voyage embedding failed: {exc}") from exc
        return all_embeddings
