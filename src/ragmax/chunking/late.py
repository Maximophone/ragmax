"""Late chunking — Jina AI's approach from the book (Ch. 3).

Embed the full document first (preserving cross-chunk context in the
embeddings), then chunk afterwards.  Each chunk's embedding is derived
by mean-pooling the relevant token embeddings from the full-document
embedding pass.

This preserves long-range contextual information that would be lost
with traditional chunk-then-embed approaches.
"""

from __future__ import annotations

import numpy as np

from ragmax.core.models import Chunk, Document


class LateChunker:
    """Late chunking: embed full document, then chunk and pool.

    Parameters
    ----------
    chunk_size : int
        Target chunk size in characters.
    chunk_overlap : int
        Overlap between chunks in characters.

    Note: This is a simplified implementation.  Full late chunking requires
    access to per-token embeddings from the model (not all APIs expose this).
    When per-token embeddings are not available, this falls back to standard
    chunking with the embedding stored on each chunk for downstream pooling.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, document: Document) -> list[Chunk]:
        """Split document into chunks with late-chunking metadata."""
        text = document.content
        if not text.strip():
            return []

        chunks: list[Chunk] = []
        start = 0
        idx = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            # Try to break at a sentence boundary
            segment = text[start:end]
            last_period = segment.rfind(". ")
            if last_period > self.chunk_size // 2:
                end = start + last_period + 2
                segment = text[start:end]

            if segment.strip():
                chunks.append(
                    Chunk(
                        content=segment,
                        document_id=document.id,
                        index=idx,
                        metadata={
                            **document.metadata,
                            "chunker": "late",
                            "char_start": start,
                            "char_end": end,
                            "total_doc_length": len(text),
                        },
                    )
                )
                idx += 1
            new_start = end - self.chunk_overlap
            if new_start <= start:
                new_start = end  # Guarantee forward progress
            start = new_start

        return chunks

    @staticmethod
    def pool_embeddings(
        token_embeddings: np.ndarray,
        chunk_token_ranges: list[tuple[int, int]],
    ) -> list[list[float]]:
        """Pool per-token embeddings into per-chunk embeddings.

        Parameters
        ----------
        token_embeddings : np.ndarray
            Shape (num_tokens, dim) — per-token embeddings for the full document.
        chunk_token_ranges : list[tuple[int, int]]
            (start, end) token indices for each chunk.

        Returns
        -------
        list[list[float]]
            One embedding per chunk via mean pooling.
        """
        pooled: list[list[float]] = []
        for start, end in chunk_token_ranges:
            chunk_embs = token_embeddings[start:end]
            if len(chunk_embs) > 0:
                mean = np.mean(chunk_embs, axis=0)
                mean = mean / np.linalg.norm(mean)
                pooled.append(mean.tolist())
            else:
                pooled.append([0.0] * token_embeddings.shape[1])
        return pooled
