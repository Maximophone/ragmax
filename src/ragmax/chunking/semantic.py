"""Semantic chunker — split on embedding similarity breakpoints.

Groups consecutive sentences by semantic similarity.  When the
cosine similarity between consecutive sentence embeddings drops
below a threshold, a new chunk boundary is created.  This produces
more coherent chunks than fixed-size approaches.
"""

from __future__ import annotations

import numpy as np

from ragmax.core.models import Chunk, Document
from ragmax.core.protocols import Embedder


class SemanticChunker:
    """Split documents on semantic boundaries using embedding similarity.

    Parameters
    ----------
    embedder : Embedder
        The embedder to use for sentence-level embeddings.
    threshold : float
        Cosine similarity threshold below which a split is created.
        Lower values = larger chunks.  Default 0.5.
    max_sentences : int
        Maximum sentences per chunk even if similarity stays high.
    """

    def __init__(
        self,
        embedder: Embedder | None = None,
        threshold: float = 0.5,
        max_sentences: int = 15,
    ) -> None:
        self.embedder = embedder
        self.threshold = threshold
        self.max_sentences = max_sentences
        self._import_re()

    def _import_re(self) -> None:
        import re

        self._sentence_re = re.compile(r"(?<=[.!?])\s+")

    def _split_sentences(self, text: str) -> list[str]:
        return [s.strip() for s in self._sentence_re.split(text) if s.strip()]

    def chunk(self, document: Document) -> list[Chunk]:
        """Synchronous wrapper — embeddings must be pre-computed or sync."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Can't await in sync context with running loop
            return self._chunk_fallback(document)
        return asyncio.run(self._chunk_async(document))

    async def _chunk_async(self, document: Document) -> list[Chunk]:
        sentences = self._split_sentences(document.content)
        if not sentences:
            return []

        if self.embedder is None:
            return self._chunk_fallback(document)

        embeddings = await self.embedder.embed(sentences)
        emb_array = np.array(embeddings)

        # Compute cosine similarities between consecutive sentences
        groups: list[list[int]] = [[0]]
        for i in range(1, len(sentences)):
            sim = self._cosine_sim(emb_array[i - 1], emb_array[i])
            if sim < self.threshold or len(groups[-1]) >= self.max_sentences:
                groups.append([i])
            else:
                groups[-1].append(i)

        chunks: list[Chunk] = []
        for idx, group in enumerate(groups):
            text = " ".join(sentences[i] for i in group)
            if text.strip():
                chunks.append(
                    Chunk(
                        content=text,
                        document_id=document.id,
                        index=idx,
                        metadata={
                            **document.metadata,
                            "chunker": "semantic",
                            "sentence_indices": group,
                        },
                    )
                )
        return chunks

    def _chunk_fallback(self, document: Document) -> list[Chunk]:
        """Fallback to sentence-based chunking without embeddings."""
        sentences = self._split_sentences(document.content)
        chunks: list[Chunk] = []
        for idx in range(0, len(sentences), self.max_sentences):
            group = sentences[idx : idx + self.max_sentences]
            text = " ".join(group)
            if text.strip():
                chunks.append(
                    Chunk(
                        content=text,
                        document_id=document.id,
                        index=len(chunks),
                        metadata={**document.metadata, "chunker": "semantic_fallback"},
                    )
                )
        return chunks

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        return float(dot / norm) if norm > 0 else 0.0
