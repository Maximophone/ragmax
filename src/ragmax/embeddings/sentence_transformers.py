"""Local sentence-transformers embedder for offline / self-hosted use."""

from __future__ import annotations

import numpy as np

from ragmax.core.exceptions import EmbeddingError
from ragmax.core.utils import require_dependency, batched


class SentenceTransformerEmbedder:
    """Embed using a local sentence-transformers model.

    Runs entirely on-device — no API calls.  Good for
    development and air-gapped environments.
    """

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 128,
    ) -> None:
        st = require_dependency("sentence_transformers", "rerankers")
        self._model = st.SentenceTransformer(model, device=device)
        self.batch_size = batch_size

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            all_embeddings: list[list[float]] = []
            for batch in batched(texts, self.batch_size):
                embs = self._model.encode(batch, convert_to_numpy=True)
                if isinstance(embs, np.ndarray):
                    all_embeddings.extend(embs.tolist())
                else:
                    all_embeddings.extend([e.tolist() for e in embs])
            return all_embeddings
        except Exception as exc:
            raise EmbeddingError(f"SentenceTransformer embedding failed: {exc}") from exc
