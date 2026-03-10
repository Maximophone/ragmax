"""Cross-encoder reranker using sentence-transformers.

Cross-encoders process (query, passage) pairs jointly through a
transformer, producing much more accurate relevance scores than
bi-encoders.  Typical latency: 200-400ms for 20 passages.
"""

from __future__ import annotations

from ragmax.core.exceptions import RerankingError
from ragmax.core.models import RerankedResult, SearchResult
from ragmax.core.utils import require_dependency


class CrossEncoderReranker:
    """Rerank using a cross-encoder model from sentence-transformers.

    Default model: ``cross-encoder/ms-marco-MiniLM-L-6-v2``
    (fast, good quality for English).

    For multilingual: ``cross-encoder/mmarco-mMiniLMv2-L12-H384-v1``.
    """

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
    ) -> None:
        st = require_dependency("sentence_transformers", "rerankers")
        self._model = st.CrossEncoder(model, device=device)

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = 5,
    ) -> list[RerankedResult]:
        if not results:
            return []
        try:
            pairs = [(query, r.chunk.content) for r in results]
            scores = self._model.predict(pairs)

            reranked = [
                RerankedResult(
                    chunk=r.chunk,
                    score=float(scores[i]),
                    original_rank=i,
                    reranker_score=float(scores[i]),
                )
                for i, r in enumerate(results)
            ]
            reranked.sort(key=lambda x: x.reranker_score, reverse=True)
            return reranked[:top_k]
        except Exception as exc:
            raise RerankingError(f"Cross-encoder reranking failed: {exc}") from exc
