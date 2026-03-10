"""Google Gemini embedding provider.

Supports both ``gemini-embedding-001`` (text-only) and
``gemini-embedding-2-preview`` (multimodal, Matryoshka MRL).
"""

from __future__ import annotations

from typing import Any

from ragmax.core.exceptions import EmbeddingError
from ragmax.core.utils import require_dependency, batched


class GoogleEmbedder:
    """Generate embeddings via the Google GenAI (Gemini) API.

    Parameters
    ----------
    model : str
        Model identifier.  ``"gemini-embedding-2-preview"`` is recommended
        for best quality and Matryoshka dimension flexibility.
    dimensions : int | None
        Output dimensionality via Matryoshka Representation Learning.
        Recommended values: 3072 (full), 1536, 768.  Only supported on
        ``gemini-embedding-2-preview`` and ``gemini-embedding-001``.
    task_type : str
        Task hint for the model.  Use ``"RETRIEVAL_DOCUMENT"`` when
        embedding documents and ``"RETRIEVAL_QUERY"`` when embedding
        queries.  Other values: ``SEMANTIC_SIMILARITY``,
        ``CLASSIFICATION``, ``CLUSTERING``, ``CODE_RETRIEVAL_QUERY``,
        ``QUESTION_ANSWERING``, ``FACT_VERIFICATION``.
    batch_size : int
        Maximum texts per API call.
    api_key : str | None
        Google AI API key.  Falls back to ``GOOGLE_API_KEY`` env var.
    """

    TASK_TYPES = frozenset({
        "RETRIEVAL_DOCUMENT",
        "RETRIEVAL_QUERY",
        "SEMANTIC_SIMILARITY",
        "CLASSIFICATION",
        "CLUSTERING",
        "CODE_RETRIEVAL_QUERY",
        "QUESTION_ANSWERING",
        "FACT_VERIFICATION",
    })

    def __init__(
        self,
        model: str = "gemini-embedding-2-preview",
        dimensions: int | None = None,
        task_type: str = "RETRIEVAL_DOCUMENT",
        batch_size: int = 64,
        api_key: str | None = None,
    ) -> None:
        genai = require_dependency("google.genai", "google")
        self._genai = genai
        from google.genai import types as genai_types

        self._types = genai_types

        client_kwargs: dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        self._client = genai.Client(**client_kwargs)

        self.model = model
        self._dimensions = dimensions
        self.task_type = task_type
        self.batch_size = batch_size

    @property
    def dimension(self) -> int:
        if self._dimensions is not None:
            return self._dimensions
        defaults = {
            "gemini-embedding-2-preview": 3072,
            "gemini-embedding-001": 768,
        }
        return defaults.get(self.model, 3072)

    def _build_config(self, task_type: str | None = None) -> Any:
        """Build EmbedContentConfig with optional dimension & task_type."""
        kwargs: dict[str, Any] = {}
        if self._dimensions is not None:
            kwargs["output_dimensionality"] = self._dimensions
        tt = task_type or self.task_type
        if tt:
            kwargs["task_type"] = tt
        return self._types.EmbedContentConfig(**kwargs) if kwargs else None

    async def embed(
        self,
        texts: list[str],
        task_type: str | None = None,
    ) -> list[list[float]]:
        """Embed a list of texts.

        Parameters
        ----------
        texts : list[str]
            Texts to embed.
        task_type : str | None
            Override the default task_type for this call.  Useful for
            switching between ``RETRIEVAL_QUERY`` and ``RETRIEVAL_DOCUMENT``.
        """
        all_embeddings: list[list[float]] = []
        config = self._build_config(task_type)

        try:
            for batch in batched(texts, self.batch_size):
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "contents": batch,
                }
                if config is not None:
                    kwargs["config"] = config
                result = self._client.models.embed_content(**kwargs)
                all_embeddings.extend([e.values for e in result.embeddings])
        except Exception as exc:
            raise EmbeddingError(f"Google embedding failed: {exc}") from exc

        return all_embeddings

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single query using RETRIEVAL_QUERY task type."""
        result = await self.embed([query], task_type="RETRIEVAL_QUERY")
        return result[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents using RETRIEVAL_DOCUMENT task type."""
        return await self.embed(texts, task_type="RETRIEVAL_DOCUMENT")
